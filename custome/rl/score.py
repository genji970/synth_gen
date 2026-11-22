"""One implementation of code verification, PCA distance, rewards and plots.
VERL's reward callback is a thin HTTP proxy to this CPU-side service.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
import os
import re
import threading
import time
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen
from uuid import uuid4

import numpy as np

from custome.rl.core import PrivateDataset, Progress, env, files
from custome.rl.data_load import DataLoad, Sample


def code_only(text):
    match = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.I | re.S)
    return match.group(1).strip() if match else text.strip()


class CodeVerifier:
    """Uses existing remote Podman client; no generated code runs on the host."""

    @staticmethod
    def dependencies(*sources):
        allowed = {x.split("==")[0].replace("-", "_"): x for x in env("RL_ALLOWED_PIP").split()}
        imported = set()
        for source in sources:
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(a.name.split(".")[0] for a in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.split(".")[0])
        packages = sorted({allowed[x] for x in imported if x in allowed})
        if not packages:
            return ""
        return ("import subprocess,sys\nsubprocess.run([sys.executable,'-m','pip','install','--quiet',"
                "'--disable-pip-version-check',*" + repr(packages) + "],check=True,timeout=120)\n")

    @staticmethod
    def apps_tests(io):
        tests = json.loads(io) if isinstance(io, str) else io
        if not isinstance(tests, dict) or not tests.get("inputs"):
            return ""
        return '''
import subprocess,sys,pytest
_CASES = %r
_NAME = %r
@pytest.mark.parametrize('i',range(len(_CASES)))
def test_apps(i):
    inp, expected = _CASES[i]
    if _NAME:
        scope = {'__name__':'candidate_module'}
        exec(compile(_SOURCE,'candidate.py','exec'),scope)
        func = scope.get(_NAME) or getattr(scope['Solution'](),_NAME)
        got = func(*(inp if isinstance(inp,list) else [inp]))
        assert got == expected
    else:
        proc = subprocess.run([sys.executable,'-c',_SOURCE],input=str(inp),
                              text=True,capture_output=True,timeout=15)
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == str(expected).strip()
''' % (list(zip(tests["inputs"], tests["outputs"])), tests.get("fn_name"))

    @staticmethod
    def result(*, passed=False, status="unverified", error_message="", error_types=None,
               summary=None, pytest_exit_code=None, test_code="", coverage=None,
               pytest_report=None, pytest_stdout="", pytest_stderr=""):
        """One canonical validation result for all success and failure paths."""
        return dict(
            passed=bool(passed), status=status, error_message=error_message,
            error_types=list(error_types or []), pytest_summary=summary or {},
            pytest_exit_code=pytest_exit_code, test_code=test_code or "",
            coverage=coverage or {}, pytest_report=pytest_report or {},
            pytest_stdout=str(pytest_stdout or ""), pytest_stderr=str(pytest_stderr or ""),
        )

    @staticmethod
    def generated_test_issue(code, test_code):
        """Reject definitely invalid / self-oracling tests, without judging answers.

        This does NOT prove tests are mathematically correct. Original code
        and test are left unchanged in the JSONL audit trail.
        """
        try:
            code_tree = ast.parse(code)
            test_tree = ast.parse(test_code)
        except SyntaxError as exc:
            return f"Python syntax error in code or test: {exc}"
        candidate_names = {node.name for node in code_tree.body
                           if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        test_functions = [node for node in ast.walk(test_tree)
                          if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                          and node.name.startswith("test_")]
        if not test_functions:
            return "No discoverable pytest/unittest test_* functions; top-level assertions are not valid tests"
        if any(isinstance(stmt, ast.Assert) for stmt in test_tree.body):
            return "Top-level assert executes during collection, outside a pytest test function"
        # A generated test without any assertions can pass without verifying
        # anything; reject it even if pytest reports test_* as passed.
        checks = [node for fn in test_functions for node in ast.walk(fn)
                  if isinstance(node, ast.Assert)
                  or (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                      and node.func.attr.startswith("assert"))
                  or (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                      and node.func.attr == "raises")]
        if not checks:
            return "Test has no meaningful assert, unittest assertion or pytest.raises check"
        # A generated test must not replace the implementation being checked.
        test_declared = {node.name for node in test_tree.body
                         if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        collisions = candidate_names & test_declared
        if collisions:
            return "Test redefines predicted function(s): " + ", ".join(sorted(collisions))
        if any(isinstance(node, ast.Name) and node.id == "pytest"
               for node in ast.walk(test_tree)):
            has_pytest = any(isinstance(node, ast.Import)
                             and any(alias.name == "pytest" and (alias.asname is None or alias.asname == "pytest")
                                     for alias in node.names)
                             for node in test_tree.body)
            if not has_pytest:
                return "Test uses pytest without 'import pytest'"

        def candidate_called(node):
            return any(isinstance(item, ast.Call)
                       and isinstance(item.func, ast.Name)
                       and item.func.id in candidate_names
                       for item in ast.walk(node))

        # The test must actually check the candidate, not simply `assert True`
        # or assertions about unrelated Python operations.
        if candidate_names and not any(candidate_called(fn) for fn in test_functions):
            return "Generated test never calls the model-predicted function"
        for fn in test_functions:
            # A test cannot compute its own ground truth by calling the very
            # function it purports to check, e.g. expect=f(x); assert f(x)==expect.
            oracle_vars = set()
            for stmt in fn.body:
                if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                    value = stmt.value
                    if value is not None and candidate_called(value):
                        targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                        for target in targets:
                            if isinstance(target, ast.Name):
                                name = target.id
                                # Any expected/gold/reference label derived
                                # from candidate() is invalid by construction.
                                if re.search(r"(?:expect|gold|oracle|reference|ground.?truth)", name, re.I):
                                    return "Test derives expected/reference answer from model prediction: " + name
                                oracle_vars.add(name)
                if isinstance(stmt, ast.Assert) and isinstance(stmt.test, ast.Compare):
                    cmp = stmt.test
                    names = {x.id for x in ast.walk(cmp) if isinstance(x, ast.Name)}
                    if candidate_called(cmp) and oracle_vars & names:
                        return "Test uses model prediction as its own expected answer (self-referential oracle)"
                    if isinstance(cmp.left, ast.Name) and len(cmp.comparators) == 1 and isinstance(cmp.comparators[0], ast.Name):
                        if cmp.left.id in oracle_vars and cmp.comparators[0].id in oracle_vars:
                            return "Test compares two values both computed by model prediction"
                    # Direct assert f(x)==f(x) is tautological.
                    if isinstance(cmp.left, ast.Call) and len(cmp.comparators) == 1 and \
                       isinstance(cmp.comparators[0], ast.Call) and \
                       ast.dump(cmp.left, include_attributes=False) == ast.dump(cmp.comparators[0], include_attributes=False):
                        return "Test compares an expression with itself (tautological assertion)"
        return ""

    def verify(self, code, test_code=None, apps_io=None):
        test_code = test_code or (self.apps_tests(apps_io) if apps_io else "")
        if not test_code:
            return self.result(status="no_tests", error_message="No executable generated tests",
                               error_types=["NoTests"], test_code=test_code)
        if not apps_io:
            try:
                ast.parse(code)
            except SyntaxError as exc:
                return self.result(status="invalid_code", error_message=f"Prediction syntax error: {exc}",
                                   error_types=["SyntaxError"], test_code=test_code)
            issue = self.generated_test_issue(code, test_code)
            if issue:
                return self.result(status="invalid_test", error_message=issue,
                                   error_types=["InvalidTest"], test_code=test_code)
        from custome.podman_generating import run_pytest
        script = (f"_SOURCE={code!r}\n" + test_code if apps_io else
                  f"exec(compile({code!r},'candidate.py','exec'),globals())\n" + test_code)
        script = self.dependencies(code, test_code) + script
        try:
            report, out, err, exit_code = run_pytest(script, timeout=300)
            report = report or {}
            summary = report.get("summary", {}) if isinstance(report, dict) else {}
            passed = (exit_code == 0 and summary.get("passed", 0) > 0
                      and summary.get("failed", 0) == 0 and summary.get("error", 0) == 0
                      and summary.get("errors", 0) == 0)
            failures = [str(t.get("call", {}).get("longrepr") or t.get("setup", {}).get("longrepr") or t)
                        for t in report.get("tests", []) if t.get("outcome") != "passed"] if isinstance(report, dict) else []
            failures += [str(t.get("longrepr", t)) for t in report.get("collectors", [])
                         if t.get("outcome") == "failed"] if isinstance(report, dict) else []
            message = "\n".join(failures) if failures else ("" if passed else (err or out or "Test failure"))
            kinds = sorted(set(re.findall(r"\b[A-Za-z_]+(?:Error|Exception|Failure)\b", message)))
            if not passed and not kinds:
                kinds = ["PytestFailure"]
            coverage = {}
            if passed and env("RL_CHECK_COVERAGE") == "1":
                from custome.podman_generating import run_test_coverage
                result = run_test_coverage(script, timeout=300)
                coverage = (result[0] or {}).get("totals") if result else {}
            return self.result(
                passed=passed, status="passed" if passed else "failed", error_message=message,
                error_types=kinds, summary=summary, pytest_exit_code=exit_code,
                test_code=test_code, coverage=coverage, pytest_report=report,
                pytest_stdout=out, pytest_stderr=err,
            )
        except Exception as error:
            return self.result(status="environment_error", error_message=str(error),
                               error_types=[type(error).__name__], test_code=test_code)


class FeatureSpace:
    """ONE frozen Code-SFT encoder shared by all historical tasks + new code."""

    def __init__(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from custome.rl.core import require_model_lineage
        model = require_model_lineage(env("RL_FEATURE_MODEL"))
        started = time.monotonic()
        Progress.log("pca-encoder", "loading frozen encoder", model=model,
                     device=env("RL_FEATURE_DEVICE"))
        with Progress.watch("pca-encoder", "still loading frozen encoder", model=model):
            self.tokenizer = AutoTokenizer.from_pretrained(model)
            self.tokenizer.padding_side = "right"
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                model, torch_dtype="auto", low_cpu_mem_usage=True
            ).to(env("RL_FEATURE_DEVICE")).eval()
        Progress.log("pca-encoder", "frozen encoder loaded", elapsed_s=round(time.monotonic()-started,1))

    def embed_many(self, texts):
        """Batched embeddings; use the last NON-padding token in each prompt."""
        import torch
        results = []
        size = max(1, env("RL_PCA_EMBED_BATCH_SIZE", int))
        started = time.monotonic()
        Progress.log("pca-embed", "embedding reference samples", total=len(texts), batch_size=size)
        for i in range(0, len(texts), size):
            Progress.pulse("pca-embed", "progress", "embedding batch",
                           processed=i, total=len(texts), elapsed_s=round(time.monotonic()-started,1))
            group = texts[i:i + size]
            encoded = self.tokenizer(
                group, return_tensors="pt", padding=True, truncation=True,
                max_length=env("RL_MAX_FEATURE_TOKENS", int),
            ).to(self.model.device)
            with torch.inference_mode():
                hidden = self.model(**encoded, output_hidden_states=True).hidden_states[-1]
                indices = encoded["attention_mask"].long().sum(1) - 1
                # Right padding is forced so last non-padding index is valid.
                selected = hidden[torch.arange(len(group), device=hidden.device), indices]
                results.extend(selected.float().cpu().numpy())
        Progress.log("pca-embed", "reference embeddings complete", count=len(results),
                     elapsed_s=round(time.monotonic()-started,1))
        return results

    @lru_cache(maxsize=16384)
    def embed(self, text):
        return self.embed_many([text])[0]


class BatchPCA:
    """Single PCA implementation for each saved subtask reference collection."""

    def __init__(self, references, embed):
        self.fit_vectors(np.stack([embed(value) for value in references]))

    def fit_vectors(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or len(x) < 4:
            raise ValueError("PCA3 requires at least four training references")
        self.mean = x.mean(0)
        # randomized SVD saves substantial time on 10% of a full task dataset.
        if len(x) > 32 and x.shape[1] > 32:
            from sklearn.decomposition import PCA
            self.axis = PCA(n_components=3, svd_solver="randomized", random_state=42).fit(x).components_
        else:
            self.axis = np.linalg.svd(x - self.mean, full_matrices=False)[2][:3]
        self.refs = (x - self.mean) @ self.axis.T
        self.scales = {}
        # Avoid an O(N^2) allocation for thousands of selected task examples.
        sample = self.refs[np.linspace(0, len(self.refs) - 1, min(len(self.refs), 256), dtype=int)]
        for n in (1, 2, 3):
            y = sample[:, :n]
            distance = np.linalg.norm(y[:, None, :] - y[None, :, :], axis=-1)
            positive = distance[np.triu_indices(len(y), 1)]
            self.scales[n] = max(float(np.median(positive)), 1e-9)
        return self

    @classmethod
    def from_vectors(cls, vectors):
        obj = cls.__new__(cls)
        return obj.fit_vectors(vectors)

    def score_vector(self, vector):
        y = (np.asarray(vector, dtype=np.float64) - self.mean) @ self.axis.T
        ratios, rewards, points = {}, {}, {}
        for n in (1, 2, 3):
            ratio = float(np.min(np.linalg.norm(self.refs[:, :n] - y[:n], axis=1)) / self.scales[n])
            ratios[f"d{n}"] = ratio
            rewards[f"d{n}"] = 1 / (1 + ratio)
            points[n] = (self.refs[:, :n].tolist(), y[:n].tolist())
        return rewards, ratios, points

    def score(self, code, embed):
        return self.score_vector(embed(code))

    @staticmethod
    def accepts(ratios):
        return all(ratios.get(f"d{n}", float("inf")) <= env(f"RL_THRESHOLD_D{n}", float)
                   for n in (1, 2, 3))


class TaskPCABank:
    """Persistent per-task 10%-sampled PCA reference banks.

    A common FROZEN encoder projects historical MCQA, math, code and new
    generated Python into matching hidden-state spaces. Each task has its own
    PCA axes, centering and scale. A candidate is compared independently to
    every PRIOR SFT task. One eligible task passes iff all 1/2/3D thresholds
    pass for that same task. Policy any/all is configurable.
    """

    @staticmethod
    def path():
        return files.output / "pca" / "task_banks.npz"

    @staticmethod
    def manifest_path():
        return files.output / "pca" / "task_banks_manifest.json"

    def __init__(self, banks, tasks, encoder_model):
        self.banks = banks
        self.tasks = list(tasks)
        self.encoder_model = str(encoder_model)

    @classmethod
    def build(cls, encoder):
        source = DataLoad()
        tasks = env("RL_SFT_TASKS").split()
        arrays, counts = {}, {}
        for task_index, task in enumerate(tasks, 1):
            Progress.log("pca-bank", "fitting historical task PCA", task=task,
                         index=task_index, total=len(tasks))
            selection = files.read(source.pca_selection_path(task))
            if selection is None:
                raise FileNotFoundError(f"Missing SFT selection for {task}; run SFT first")
            records = selection["records"]
            embeddings = encoder.embed_many([r["pca_text"] for r in records])
            bank = BatchPCA.from_vectors(np.stack(embeddings))
            for key, value in (("mean", bank.mean), ("axis", bank.axis), ("refs", bank.refs),
                               ("scales", np.array([bank.scales[n] for n in (1, 2, 3)]))):
                arrays[f"{task}__{key}"] = value
            counts[task] = len(records)
            print(f"[prior PCA] task={task} selected={len(records)} "
                  f"observed={selection['observed_unique']} fraction={selection['fraction']}", flush=True)
        Progress.log("pca-bank", "saving task PCA reference bank", tasks=len(tasks))
        path = cls.path()
        path.parent.mkdir(parents=True, exist_ok=True)
        import tempfile
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as tmp:
            temp = Path(tmp.name)
        try:
            np.savez_compressed(temp, **arrays)
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)
        files.atomic(cls.manifest_path(), {
            "tasks": tasks, "counts": counts, "fraction": env("RL_PCA_REFERENCE_FRACTION", float),
            "encoder_model": env("RL_FEATURE_MODEL"),
            "feature_text": "USER\n{canonical_prompt}\nASSISTANT\n{reference_answer_or_generated_code}",
        })
        return cls.load()

    @classmethod
    def load(cls):
        info = files.read(cls.manifest_path())
        if not info or not cls.path().exists():
            raise FileNotFoundError("PCA task bank not built; run 'python -m custome.rl.run pca_build MODEL'")
        if info["encoder_model"] != env("RL_FEATURE_MODEL"):
            raise ValueError("PCA encoder model differs from current RL_FEATURE_MODEL")
        result = {}
        with np.load(cls.path(), allow_pickle=False) as saved:
            for task in info["tasks"]:
                bank = BatchPCA.__new__(BatchPCA)
                bank.mean = saved[f"{task}__mean"].copy()
                bank.axis = saved[f"{task}__axis"].copy()
                bank.refs = saved[f"{task}__refs"].copy()
                scale = saved[f"{task}__scales"]
                bank.scales = {n: float(scale[n - 1]) for n in (1, 2, 3)}
                result[task] = bank
        return cls(result, info["tasks"], info["encoder_model"])

    def prior_tasks(self):
        target = env("RL_GRPO_TASK")
        if target not in self.tasks:
            raise ValueError(f"GRPO task {target} is not in learned task order {self.tasks}")
        n = self.tasks.index(target) + (env("RL_PCA_INCLUDE_CURRENT_TASK", int) != 0)
        eligible = self.tasks[:n]
        if not eligible:
            raise ValueError("No earlier task PCA reference is available")
        return eligible

    def score(self, code, embed):
        vector = embed(code)  # Encode candidate ONCE, score against all tasks.
        per_task = {}
        eligible = self.prior_tasks()
        for task in eligible:
            rewards, ratios, points = self.banks[task].score_vector(vector)
            per_task[task] = {"distance_ratios": ratios, "matched": BatchPCA.accepts(ratios),
                              "rewards": rewards, "points": points}
        policy = env("RL_PCA_MATCH_POLICY")
        if policy not in ("any", "all"):
            raise ValueError("RL_PCA_MATCH_POLICY must be any or all")
        matched = (any(x["matched"] for x in per_task.values()) if policy == "any"
                   else all(x["matched"] for x in per_task.values()))
        eligible_matches = [t for t in eligible if per_task[t]["matched"]] or eligible
        nearest = min(eligible_matches, key=lambda t: sum(per_task[t]["distance_ratios"].values()))
        chosen = per_task[nearest]
        details = {task: {"distance_ratios": data["distance_ratios"], "matched": data["matched"]}
                   for task, data in per_task.items()}
        return chosen["rewards"], chosen["distance_ratios"], chosen["points"], {
            "pca_match_task": nearest, "pca_matched": matched,
            "pca_match_policy": policy, "pca_by_task": details,
            "pca_reference_tasks": eligible,
        }


class Scatter:
    """Only circular dots; blue=reference, orange=candidate."""

    def plot(self, step, identifier, points, rewards):
        if env("RL_VISUALIZE") != "1" or step % max(1, env("RL_VIS_EVERY", int)):
            return
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        folder = files.track() / "visualizations"
        folder.mkdir(parents=True, exist_ok=True)
        for n, (refs, candidate) in points.items():
            refs, candidate = np.array(refs), np.array(candidate)
            fig = plt.figure(figsize=(5, 4))
            ax = fig.add_subplot(111, projection="3d") if n == 3 else fig.add_subplot(111)
            if n == 3:
                ax.scatter(*refs.T, c="#286db1", marker="o", label="prior task")
                ax.scatter(*candidate, c="#e68c3b", marker="o", label="generated")
            else:
                ax.scatter(refs[:, 0], refs[:, 1] if n == 2 else np.zeros(len(refs)),
                           c="#286db1", marker="o", label="prior task")
                ax.scatter(candidate[0], candidate[1] if n == 2 else 0,
                           c="#e68c3b", marker="o", label="generated")
            ax.legend(); fig.tight_layout()
            fig.savefig(folder / f"step{step:05d}_{identifier}_{n}d.png", dpi=125)
            plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 3))
        for i, (name, value) in enumerate(rewards.items()):
            ax.scatter(i, value, c="#286db1" if name.startswith("d") else "#e68c3b", marker="o")
        ax.set_xticks(range(len(rewards)), list(rewards)); ax.set_ylim(-.02, 1.02)
        fig.tight_layout(); fig.savefig(folder / f"step{step:05d}_{identifier}_rewards.png", dpi=125)
        plt.close(fig)


class ScoreEngine:
    """One verifier, one frozen encoder and persisted PRIOR task PCA banks."""

    def __init__(self):
        Progress.log("reward", "initializing reward service (encoder + prior task PCA)")
        self.verifier, self.encoder, self.plotter = CodeVerifier(), FeatureSpace(), Scatter()
        self.bank = TaskPCABank.load()
        self._scored = 0
        Progress.log("reward", "reward engine ready", tasks=self.bank.tasks)
        self.lock = threading.RLock()  # Protect inference on one frozen encoder.

    def evaluate(self, payload):
        started = time.monotonic()
        info = payload.get("extra_info") or {}
        code = code_only(payload["code"])
        # Remote pytest can run on multiple threads at once. Only the frozen
        # encoder/PCA cache is serialized, not the much slower Podman calls.
        test = self.verifier.verify(code, info.get("test_code"), info.get("apps_io"))
        rewards = {f"d{n}": 0. for n in (1, 2, 3)}
        ratios, points = {}, {}
        # Same canonical prompt + answer format that built SFT reference banks.
        question = info.get("question") or info.get("problem") or ""
        if not question:
            raise ValueError("Question is required for task PCA candidate embedding")
        prompt = Sample(task="code", kind="code", question=question).prompt
        candidate_text = Sample.pca_text_for(prompt, code)
        with self.lock:
            values, ratios, points, pca_meta = self.bank.score(candidate_text, self.encoder.embed)
            rewards.update(values)
        rewards["exec"] = float(test["passed"])
        rewards["err"] = 1 / (1 + len(test["error_types"]))
        weights = {key: env(f"RL_WEIGHT_{key.upper()}", float) for key in rewards}
        if min(weights.values()) < 0 or sum(weights.values()) <= 0:
            raise ValueError("Reward weights must be nonnegative and sum to >0")
        total = sum(weights[k] * rewards[k] for k in rewards) / sum(weights.values())
        uid = uuid4().hex[:12]
        from datetime import datetime, timezone
        source = payload.get("source", "grpo")
        source_record = info.get("source_record") or {}
        original_question = (info.get("question") or info.get("problem")
                             or source_record.get("question") or "")
        sample_fields = {
            "id": uid, "source": source, "track": env("RL_TRACK", default="shared"),
            "step": info.get("batch_id"), "seed": info.get("seed"),
            "question": original_question,
            "generation_prompt": source_record.get("generation_prompt", ""),
            "generator_model": source_record.get("generator_model", ""),
            "source_record": source_record,
            "code": code, "raw_output": payload.get("raw_output", code),
            "apps_io": info.get("apps_io"), "test_code": test.get("test_code"),
            "passed": test["passed"], "status": test["status"],
            "error_message": test["error_message"], "error_types": test["error_types"],
            "pytest_summary": test["pytest_summary"],
            "pytest_report": test["pytest_report"],
            "pytest_stdout": test["pytest_stdout"],
            "pytest_stderr": test["pytest_stderr"],
            "pytest_exit_code": test.get("pytest_exit_code"),
            "coverage": test.get("coverage"),
            "distance_ratios": ratios, "reward_components": rewards,
            "reward_total": total,
            "metadata": {**dict(source_record.get("metadata") or {}), **pca_meta},
            "reference_step": "prior_sft_task_pca",
        }
        # Carry the full original source record (including synth verifier errors,
        # raw benchmark fields and generator snapshot) through reward scoring.
        sample_fields = {**source_record, **sample_fields}
        sample_fields.pop("schema_version", None)  # Revalidate computed fields.
        mapper = DataLoad()
        row = (mapper.synth(sample_fields) if source == "synth_gen"
               else mapper.rollout(sample_fields)).to_dict()
        if payload.get("record", True):
            PrivateDataset("generated", env("RL_TRACK", default="shared")).append([row])
        if points:
            self.plotter.plot(int(info.get("batch_id", 0)), uid, points, rewards)
        if not payload.get("record", True):
            from custome.rl.data_load import Sample
            row = {key: row[key] for key in Sample.verification_fields}
        self._scored += 1
        every = max(1, env("RL_PROGRESS_REWARD_EVERY", int))
        if self._scored == 1 or self._scored % every == 0:
            Progress.log("reward", "code candidates scored", count=self._scored,
                         source=source, last_score=round(total,4), last_test_passed=bool(test["passed"]),
                         last_elapsed_s=round(time.monotonic()-started,1))
        return {"score": total, "acc": rewards["exec"], "record": row}


class ScoreHTTP(BaseHTTPRequestHandler):
    scorer = None

    def respond(self, status, body):
        raw = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers(); self.wfile.write(raw)

    def do_GET(self):
        self.respond(200, {"ok": True}) if self.path == "/health" else self.respond(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/score" or self.headers.get("Authorization") != "Bearer " + env("RL_SERVICE_TOKEN"):
            self.respond(401, {"error": "unauthorized"}); return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size < 2_000_000:
                self.respond(413, {"error": "bad size"}); return
            result = self.scorer.evaluate(json.loads(self.rfile.read(size)))
            self.respond(200, result)
        except Exception as error:
            self.respond(500, {"error": f"{type(error).__name__}: {error}"})


def serve():
    if not os.environ.get("RL_SERVICE_TOKEN"):
        raise RuntimeError("RL_SERVICE_TOKEN missing")
    ScoreHTTP.scorer = ScoreEngine()
    Progress.log("reward", "HTTP scoring service listening", bind=env("RL_SERVICE_BIND"),
                 port=env("RL_SERVICE_PORT", int))
    ThreadingHTTPServer((env("RL_SERVICE_BIND"),
                         env("RL_SERVICE_PORT", int)), ScoreHTTP).serve_forever()


def call_service(payload):
    request = Request(env("RL_SERVICE_URL") + "/score", data=json.dumps(payload).encode(), method="POST",
                      headers={"Authorization": "Bearer " + env("RL_SERVICE_TOKEN"),
                               "Content-Type": "application/json"})
    with urlopen(request, timeout=900) as response:
        return json.load(response)


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """VERL reward.custom_reward_function.name: no local model per Ray worker."""
    response = call_service({"source": "grpo", "code": solution_str,
                             "raw_output": solution_str, "extra_info": extra_info or {}})
    return {"score": response["score"], "acc": response["acc"],
            **{f"reward_{key}": val for key, val in response["record"]["reward_components"].items()}}


if __name__ == "__main__":
    serve()
