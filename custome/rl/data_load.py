"""The only dataset-to-schema and schema-to-training-prompt conversion module.

Raw benchmark rows and synth_gen/GRPO outputs all become ``Sample`` records.
The same immutable prompt template is used by SFT, GRPO, and evaluation.

JSONL keeps structured dictionaries; Parquet JSON-encodes nested columns so
all datasets, including rows with missing answers, have a stable Arrow schema.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, ClassVar

from custome.rl.core import Progress, files, env


def _json(value):
    return json.dumps(value, ensure_ascii=False, default=str)


def prompt_content(value):
    """Extract ONLY user-visible text, never repr() of chat/message objects.

    Handles plain strings, LangChain BaseMessage/ChatPromptValue, common
    role/content dictionaries, and message lists. Other objects fail closed.
    Used by all synth_gen LLM calls; Sample.prompt is already a string.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if "content" in value:
            return prompt_content(value["content"])
        if "messages" in value:
            return prompt_content(value["messages"])
    if isinstance(value, (list, tuple)):
        return "\n".join(prompt_content(message) for message in value)
    # LangChain messages expose .content; PromptValue exposes .messages.
    if hasattr(value, "content"):
        return prompt_content(value.content)
    if hasattr(value, "messages"):
        return prompt_content(value.messages)
    raise TypeError(f"LLM prompt must contain text, not {type(value).__name__}")


def boxed_answer(text):
    """Extract the last balanced \boxed{...} answer; shared with evaluator."""
    k = text.rfind("\\boxed{")
    if k < 0:
        return text.strip().splitlines()[-1].strip()
    begin = k + 7
    depth = 1
    for i in range(begin, len(text)):
        depth += (text[i] == "{") - (text[i] == "}")
        if depth == 0:
            return text[begin:i].replace(" ", "").replace("\\left", "").replace("\\right", "")
    return text.strip()


def strict_math_answer(solution):
    """Gold MATH labels MUST come from a balanced boxed final answer."""
    k = solution.rfind("\\boxed{")
    if k < 0:
        return ""
    start, depth = k + len("\\boxed{"), 1
    for end in range(start, len(solution)):
        if solution[end] == "{":
            depth += 1
        elif solution[end] == "}":
            depth -= 1
            if depth == 0:
                answer = solution[start:end].strip()
                if not answer or "\n" in answer:
                    return ""
                return answer.replace(" ", "").replace("\\left", "").replace("\\right", "")
    return ""  # An unmatched box is not a trustworthy label.


def problem_leak_reason(question):
    """Recognize solution CODE embedded in a synthetic *question*.

    This intentionally rejects code skeletons too: synthetic Code GRPO
    problems must be natural-language specifications, not executable source.
    No model answer is ever removed from a question heuristically; an invalid
    synthetic problem must be regenerated or rejected with its trace saved.
    """
    text = str(question or "")
    if not text.strip():
        return "empty generated problem"
    if re.search(r"(?:ChatPromptValue\s*\(|HumanMessage\s*\(|messages=\[HumanMessage)", text):
        return "serialized chat object embedded in generated question"
    if re.search(r"(?m)^\s*```(?:python|py)\b", text, flags=re.I):
        return "Python code fence embedded in generated question"
    # These are the executable structures seen in failing RunPod smoke tests.
    if re.search(r"(?m)^\s*(?:async\s+def|def|class)\s+\w+\s*[(:]", text):
        return "Python function/class definition embedded in generated question"
    if re.search(r"(?m)^\s*(?:from\s+\S+\s+import|import\s+\w+|return\b|yield\b|assert\b|@\w+|if\s+__name__\s*==)", text):
        return "Python implementation/test statement embedded in generated question"
    if re.search(r"(?im)^\s*(?:solution|answer|final answer|correct answer|reference solution|answer key|worked solution|reasoning)\s*:", text):
        return "answer or solution section embedded in generated question"
    if re.search(r"(?i)\b(?:the answer is|final answer is|the correct answer is)\b", text):
        return "stated final answer embedded in generated question"
    # A synthetic problem is deliberately generated WITHOUT a worked example.
    # Example computations can reveal gold outputs or an entire solution before
    # the policy has predicted anything, even without the word 'answer'.
    if re.search(r"(?im)^\s*(?:\*\*)?(?:worked example|example|sample walkthrough)(?:\*\*)?\s*:", text):
        return "worked/example answer section embedded in synthetic question"
    # 'e.g. 10^9+7' is not a specified modulus: it leaves multiple legitimate
    # correct implementations, so any single numerical test oracle is suspect.
    if re.search(r"(?i)\b(?:modulo|modulus|prime number)\b[^\n]{0,100}\b(?:e\.g\.|for example\b|such as\b)", text):
        return "unspecified example-only modulus in generated problem"
    return ""


def _contained_reference_code(question, references):
    """Reject exact benchmark reference-code copy in the problem statement."""
    question_compact = re.sub(r"\s+", "", question)
    for code in references:
        compact = re.sub(r"\s+", "", str(code))
        if len(compact) >= 24 and compact in question_compact:
            return True
    return False


def _parse(value, default):
    if value is None:
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return default
    return value


@dataclass
class Sample:
    """Canonical, versioned schema for original and generated data."""

    schema_version: int = 1
    task: str = ""
    kind: str = ""
    source: str = ""
    split: str = "train"
    source_id: str = ""
    question: str = ""
    choices: dict[str, str] = field(default_factory=dict)
    answer: str = ""
    reference: str = ""  # Gold final label or gold code; not a MATH derivation.
    sft_solution: str = ""  # Deliberate MATH assistant target (full worked solution).
    references: list[str] = field(default_factory=list)
    generated_code: str = ""
    generation_prompt: str = ""
    origin_source_id: str = ""
    generator_model: str = ""
    synth_gen_passed: bool | None = None
    synth_gen_failed_verifier: str = ""
    synth_gen_error: str = ""
    synth_verification_success: bool | None = None
    verification_history: list[dict] = field(default_factory=list)
    generation_history: list[dict] = field(default_factory=list)
    test_code: str = ""
    apps_io: str = ""
    seed: str = ""
    passed: bool | None = None
    status: str = "unverified"
    success_reason: str = ""
    failure_stage: str = ""
    failure_reason: str = ""
    error_message: str = ""
    error_types: list[str] = field(default_factory=list)
    pytest_summary: dict = field(default_factory=dict)
    pytest_report: dict = field(default_factory=dict)
    pytest_stdout: str = ""
    pytest_stderr: str = ""
    pytest_exit_code: int | None = None
    coverage: dict = field(default_factory=dict)
    accepted: bool | None = None
    rejection_reason: str = ""
    step: int = -1
    attempt: int = -1
    seed_slot: int = -1
    reference_step: str = ""
    policy_snapshot_step: int = -1
    distance_ratios: dict[str, float] = field(default_factory=dict)
    reward_components: dict[str, float] = field(default_factory=dict)
    reward_total: float | None = None
    raw_output: str = ""
    raw_source: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    prompt: str = ""

    verification_fields: ClassVar[tuple[str, ...]] = (
        "passed", "status", "error_message", "error_types", "pytest_summary",
        "pytest_report", "pytest_stdout", "pytest_stderr", "pytest_exit_code",
        "coverage", "test_code", "distance_ratios", "reward_components", "reward_total", "metadata",
    )

    nested_fields: ClassVar[tuple[str, ...]] = (
        "choices", "references", "error_types", "distance_ratios",
        "reward_components", "raw_source", "metadata", "pytest_summary", "coverage",
        "pytest_report", "verification_history", "generation_history",
    )

    def __post_init__(self):
        if self.kind not in ("mcqa", "short_answer", "code"):
            raise ValueError(f"Invalid kind: {self.kind}")
        if self.task not in ("math", "social", "code"):
            raise ValueError(f"Invalid task: {self.task}")
        if not self.question.strip():
            raise ValueError("Missing question; refusing malformed training sample")
        if self.kind == "mcqa":
            if set(self.choices) != {"A", "B", "C"} or self.answer not in self.choices:
                raise ValueError("Social IQa requires A/B/C choices and an A/B/C gold label")
            if not all(str(value).strip() for value in self.choices.values()):
                raise ValueError("Social IQa has an empty choice")
        if self.task == "math":
            # Upgrade cached old Parquet rows with a reasoning-containing
            # reference: reasoning stays ONLY in the SFT assistant target.
            if self.reference and self.reference != self.answer:
                extracted = strict_math_answer(self.reference)
                if extracted != self.answer:
                    raise ValueError("MATH reference contains an unverified or non-boxed answer")
                self.sft_solution = self.sft_solution or self.reference
                self.reference = extracted
            if not self.answer or not self.reference:
                raise ValueError("MATH requires a boxed final-answer gold label")
        if self.passed not in (None, True, False):
            raise ValueError("passed must be None, True or False")
        if self.accepted not in (None, True, False):
            raise ValueError("accepted must be None, True or False")
        self.prompt = self.build_prompt()

    def assert_prompt_integrity(self):
        """Fail closed before any SFT/GRPO/eval model sees a leaked answer."""
        if any(token in self.prompt for token in ("HumanMessage(", "ChatPromptValue(", "messages=[HumanMessage")):
            raise ValueError("serialized chat object in model prompt")
        if self.task == "math":
            if "\\boxed{" in self.question or re.search(
                r"(?im)^\s*(?:solution|final answer|correct answer|answer key)\s*:", self.question
            ):
                raise ValueError("MATH question contains an explicit answer/solution marker")
        if self.kind == "mcqa" and re.search(
            r"(?im)^\s*(?:correct answer|gold label|answer key)\s*:\s*[ABC]\s*$", self.question
        ):
            raise ValueError("Social IQa question explicitly contains the gold answer label")
        if self.kind == "code" and self.source == "synth_gen":
            reason = problem_leak_reason(self.question)
            if reason:
                raise ValueError(f"synthetic question leaks code/answer: {reason}")
        if self.kind == "code" and self.source == "APPS":
            if _contained_reference_code(self.question, self.references):
                raise ValueError("APPS question contains an entire official reference solution")
        return True

    def build_prompt(self):
        if self.kind == "mcqa":
            choices = "\n".join(f"{k}. {v}" for k, v in self.choices.items())
            allowed = ", ".join(self.choices)
            return f"{self.question}\n\nChoices:\n{choices}\n\nAnswer with one option letter ({allowed})."
        if self.kind == "short_answer":
            return self.question + "\n\nSolve the problem and state the final answer clearly."
        return (self.question + "\n\nWrite runnable Python code only. "
                "Do not include Markdown fences or explanations.")

    @staticmethod
    def pca_text_for(prompt, answer):
        """Same prompt/answer representation for MCQA, math and code PCA.

        A fixed encoder embeds this text for both stored SFT references and
        generated GRPO code; no per-task ad-hoc text templates.
        """
        return f"USER:\n{prompt}\nASSISTANT:\n{answer}"

    def pca_text(self):
        return self.pca_text_for(self.prompt, self.training_answer)

    @property
    def training_answer(self):
        if self.kind == "code":
            return self.generated_code or self.reference or self.answer
        if self.kind == "mcqa":
            return self.answer
        if self.task == "math":
            return self.sft_solution or self.answer
        return self.reference or self.answer

    @property
    def code(self):
        return self.generated_code or self.reference or self.answer

    def to_dict(self):
        return asdict(self)

    def to_storage(self):
        row = self.to_dict()
        for key in self.nested_fields:
            row[key] = _json(row[key])
        return row

    @classmethod
    def from_dict(cls, row):
        if isinstance(row, cls):
            return row
        allowed = {f.name for f in fields(cls)}
        data = {k: v for k, v in row.items() if k in allowed}
        for key in cls.nested_fields:
            if key in data:
                default = [] if key in ("references", "error_types", "verification_history", "generation_history") else {}
                data[key] = _parse(data[key], default)
        return cls(**data)

    def to_sft(self):
        self.assert_prompt_integrity()
        if not self.training_answer:
            raise ValueError("No reference answer for SFT")
        # All canonical fields are preserved beside the VERL messages column.
        return {
            **self.to_storage(),
            "messages": [
                {"role": "user", "content": self.prompt},
                {"role": "assistant", "content": self.training_answer},
            ],
        }

    def reward_reference(self):
        """Small stable link to the full local training JSONL record.

        Do not copy megabytes of raw synth_gen chat and pytest traces into
        every vLLM rollout. The accepted_training.jsonl file holds originals.
        """
        keys = (
            "source_id", "source", "question", "seed", "generator_model",
            "policy_snapshot_step", "origin_source_id", "synth_gen_passed",
            "synth_gen_failed_verifier", "synth_gen_error", "accepted",
        )
        return {key: getattr(self, key) for key in keys}

    def to_verl(self, step, split="train"):
        self.assert_prompt_integrity()
        import torch
        # Only raw_prompt is model-visible. Tests/expected outputs stay in
        # extra_info for the external reward verifier, never in raw_prompt.
        return {
            "raw_prompt": [{"role": "user", "content": self.prompt}],
            "data_source": self.task,
            # The configured GRPO task is Code and uses external execution
            # reward; no gold/reference is needed anywhere in the actor batch.
            "reward_model": {"ground_truth": ""},
            "extra_info": {
                "index": self.source_id, "question": self.question,
                "problem": self.question, "seed": self.seed,
                "apps_io": self.apps_io, "test_code": self.test_code,
                "batch_id": step,
                # Distances are measured against saved PRIOR TASK PCA banks,
                # not against this or the previous GRPO mini-batch.
                "split": split, "benchmark": self.source,
                "source_record": self.reward_reference(),
            },
            "index": self.source_id,
            "dummy_tensor": torch.zeros(1, dtype=torch.uint8),
            "tools_kwargs": {}, "interaction_kwargs": {},
        }


class Mapper:
    """Shared source-preserving conversion; subclasses only map dataset-specific fields."""

    task: ClassVar[str]
    kind: ClassVar[str]
    source: ClassVar[str]

    def convert(self, row, index=0, split="train"):
        original = dict(row)
        fields = self.extract(original)
        return Sample(
            task=self.task, kind=self.kind, source=self.source,
            split=split, source_id=str(original.get("id", index)),
            raw_source=original, **fields,
        )

    def extract(self, row):
        raise NotImplementedError


class MathMapper(Mapper):
    task, kind, source = "math", "short_answer", "MATH"

    def extract(self, row):
        solution = str(row["solution"])
        # Final gold label is separate from the deliberately supervised SFT
        # assistant response (which contains the dataset's full worked solution).
        gold = strict_math_answer(solution)
        if not gold:
            raise ValueError("MATH row has no reliable boxed final answer")
        return {"question": str(row["problem"]),
                "answer": gold, "reference": gold, "sft_solution": solution}


class SocialMapper(Mapper):
    task, kind, source = "social", "mcqa", "SocialIQA"

    def extract(self, row):
        question = f"{row['context']}\n{row['question']}"
        try:
            label = int(row["label"])
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid Social IQa gold label") from exc
        if label not in (1, 2, 3):
            raise ValueError("Social IQa gold label must be 1, 2 or 3")
        return {"question": question, "choices": {
            "A": str(row["answerA"]), "B": str(row["answerB"]), "C": str(row["answerC"]),
        }, "answer": "ABC"[label - 1]}


class CodeMapper(Mapper):
    task, kind, source = "code", "code", "APPS"

    def extract(self, row):
        solutions = _parse(row.get("solutions"), [])
        if not isinstance(solutions, list):
            solutions = []
        references = [str(s) for s in solutions if s]
        return {
            "question": str(row["question"]), "answer": references[0] if references else "",
            "reference": references[0] if references else "",
            "references": references,
            "apps_io": _json(_parse(row.get("input_output"), {})),
        }


class GeneratedMapper(Mapper):
    task, kind, source = "code", "code", "synth_gen"

    def convert(self, row, index=0, split="train"):
        if row.get("schema_version") == 1 and row.get("kind") == "code":
            return Sample.from_dict(row)
        original = dict(row)
        source = original.get("source", self.source)
        code = str(original.get("generated_code") or original.get("code") or "")
        status = str(original.get("status", "unverified"))
        rejected = str(original.get("rejection_reason") or "")
        passed = original.get("passed")
        reason = str(original.get("error_message") or original.get("synth_gen_error") or "")
        failed_verifier = str(original.get("synth_gen_failed_verifier") or "")
        failure_stage = str(original.get("failure_stage") or (
            "generation" if status == "generation_error" else
            "pca_filter" if rejected == "distance" else
            "deduplication" if rejected == "duplicate" else
            "synth_gen_verifier" if rejected == "synth_gen_verifier" else
            "synth_gen_verifier" if failed_verifier and not original.get("synth_gen_passed") else
            "code_verification" if passed is False else ""
        ))
        upstream = original.get("raw_source") or original.get("source_record") or {}
        if not isinstance(upstream, dict):
            upstream = {"unparsed_raw_source": str(upstream)}
        return Sample(
            task="code", kind="code", source=source, split=split,
            source_id=str(original.get("id") or index),
            origin_source_id=str(original.get("origin_source_id") or upstream.get("source_id", "")),
            generation_prompt=str(original.get("generation_prompt") or ""),
            generator_model=str(original.get("generator_model") or ""),
            synth_gen_passed=original.get("synth_gen_passed"),
            synth_gen_failed_verifier=str(original.get("synth_gen_failed_verifier") or ""),
            synth_gen_error=str(original.get("synth_gen_error") or ""),
            synth_verification_success=original.get("synth_verification_success"),
            verification_history=list(original.get("verification_history") or []),
            generation_history=list(original.get("generation_history") or []),
            question=str(original.get("question") or original.get("problem") or
                         "UNAVAILABLE: code generation failed before a problem was produced"),
            answer=code, reference=code, references=[code] if code else [],
            generated_code=code,
            test_code=str(original.get("test_code") or ""),
            apps_io=str(original.get("apps_io") or ""),
            passed=passed, status=status,
            success_reason=(str(original.get("success_reason") or "pytest passed") if passed is True else ""),
            failure_stage=failure_stage,
            failure_reason=str(original.get("failure_reason") or rejected or reason),
            error_message=reason,
            error_types=list(original.get("error_types") or []),
            pytest_summary=dict(original.get("pytest_summary") or {}),
            pytest_report=dict(original.get("pytest_report") or {}),
            pytest_stdout=str(original.get("pytest_stdout") or ""),
            pytest_stderr=str(original.get("pytest_stderr") or ""),
            pytest_exit_code=original.get("pytest_exit_code"),
            coverage=dict(original.get("coverage") or {}),
            accepted=original.get("accepted"), rejection_reason=rejected,
            seed=str(original.get("seed") or ""),
            step=int(original.get("step") if original.get("step") is not None else
                     original.get("batch_id") if original.get("batch_id") is not None else -1),
            attempt=int(original.get("attempt", -1) if original.get("attempt") is not None else -1),
            seed_slot=int(original.get("seed_slot", -1) if original.get("seed_slot") is not None else -1),
            reference_step=str(original.get("reference_step") or ""),
            policy_snapshot_step=int(original.get("policy_snapshot_step", -1)),
            distance_ratios=dict(original.get("distance_ratios") or {}),
            reward_components=dict(original.get("reward_components") or {}),
            reward_total=original.get("reward_total"),
            raw_output=str(original.get("raw_output") or ""),
            raw_source=dict(upstream),
            # Preserve explicit metadata AND previously unseen source fields.
            # Discarding the explicit metadata field silently would erase
            # independent-verifier agreement flags and other audit evidence.
            metadata={
                **dict(original.get("metadata") or {}),
                **{k: v for k, v in original.items() if k not in (
                    {f.name for f in fields(Sample)} | {
                        "id", "code", "problem", "batch_id", "source_record",
                        "rejection_reason", "error_message", "raw_source",
                    })},
            },
        )

class DataLoad:
    """One facade for every benchmark and generated code; no parallel parsers."""

    names = ("math", "social", "code")
    math_subsets = (
        "algebra", "counting_and_probability", "geometry", "intermediate_algebra",
        "number_theory", "prealgebra", "precalculus",
    )

    def __init__(self):
        self.mappers = {"math": MathMapper(), "social": SocialMapper(),
                        "code": CodeMapper(), "synth": GeneratedMapper()}

    def math(self, row, index=0, split="train"):
        return self.mappers["math"].convert(row, index, split)

    def social(self, row, index=0, split="train"):
        return self.mappers["social"].convert(row, index, split)

    def code(self, row, index=0, split="train"):
        return self.mappers["code"].convert(row, index, split)

    def synth(self, row, index=0, split="train"):
        return self.mappers["synth"].convert(row, index, split)

    # RL rollout samples are also generated code, differentiated by source.
    def rollout(self, row, index=0, split="train"):
        return self.synth({**row, "source": "grpo"}, index, split)

    @classmethod
    def read(cls, task, split="train"):
        import pandas as pd
        if task not in cls.names:
            raise ValueError(f"Unknown task: {task}")
        path = files.data / f"{task}_{'eval' if split in ('val','test','eval') else 'train'}_raw.parquet"
        return [Sample.from_dict(record) for record in pd.read_parquet(path).to_dict("records")]

    @staticmethod
    def pca_trace_path(task, rank):
        return files.output / "pca" / "sft_trace" / f"{task}_rank_{rank}.json"

    @staticmethod
    def pca_selection_path(task):
        return files.output / "pca" / "selections" / f"{task}.json"

    @classmethod
    def consolidate_pca_trace(cls, task):
        """Merge selected unique examples from all SFT ranks, with audit info.

        Only ACTUALLY INDEXED train examples are eligible; distinct ranks can
        sample the same source ID, so each source is kept once.
        """
        if task not in cls.names:
            raise ValueError(f"Unknown SFT task for PCA: {task}")
        total = env("RL_NNODES", int) * env("RL_GPUS_PER_NODE", int)
        rows, fallback, observed = {}, {}, set()
        for rank in range(total):
            path = cls.pca_trace_path(task, rank)
            trace = files.read(path)
            if trace is None:
                raise FileNotFoundError(f"Missing SFT PCA trace: {path}")
            if trace.get("task") != task:
                raise ValueError(f"Wrong task in PCA trace: {path}")
            observed.update(str(i) for i in trace.get("observed_ids", []))
            for record in trace.get("selected", []):
                rows[str(record["source_id"])] = record
            for record in trace.get("fallback", []):
                fallback[str(record["source_id"])] = record
        minimum = max(4, env("RL_PCA_MIN_REFERENCE_SAMPLES", int))
        if len(rows) < minimum:
            for key, record in sorted(fallback.items()):
                rows.setdefault(key, record)
                if len(rows) >= minimum:
                    break
        if len(rows) < minimum:
            raise ValueError(f"Task {task}: only {len(rows)} indexed PCA references; "
                             f"need >= {minimum}. Increase SFT steps or train batch.")
        selection = {
            "task": task, "fraction": env("RL_PCA_REFERENCE_FRACTION", float),
            "observed_unique": len(observed), "selected_count": len(rows),
            "records": [rows[k] for k in sorted(rows)],
            "source": "actually_indexed_sft_training_data",
        }
        files.atomic(cls.pca_selection_path(task), selection)
        return selection

    def download(self):
        from datasets import load_dataset, concatenate_datasets
        for split in ("train", "test"):
            math_parts = []
            for subset in self.math_subsets:
                Progress.log("data-load", "downloading MATH subset", subset=subset, split=split)
                with Progress.watch("data-load", "MATH subset download still running", subset=subset, split=split):
                    math_parts.append(load_dataset("EleutherAI/hendrycks_math", subset, split=split))
            math = concatenate_datasets(math_parts)
            Progress.log("data-load", "downloading Social IQa", split=split)
            with Progress.watch("data-load", "Social IQa download still running", split=split):
                social = load_dataset("baber/social_i_qa", split="validation" if split == "test" else "train")
            Progress.log("data-load", "downloading APPS", split=split)
            with Progress.watch("data-load", "APPS download still running", split=split):
                code = load_dataset("codeparrot/apps", split=split)
            yield split, {"math": math, "social": social, "code": code}

    def prepare(self):
        from datasets import Dataset
        files.data.mkdir(parents=True, exist_ok=True)
        Progress.log("data-load", "downloading and preparing source datasets", tasks=self.names)
        for split, datasets in self.download():
            dest = "eval" if split == "test" else "train"
            Progress.log("data-load", "download split ready", split=dest)
            for task in self.names:
                convert = getattr(self, task)
                total = len(datasets[task])
                Progress.log("data-load", "mapping source dataset", task=task, split=dest, total=total)
                rows = []
                skipped = 0
                rejected_rows = []
                for i, raw in enumerate(datasets[task]):
                    try:
                        row = convert(raw, i, dest)
                        row.assert_prompt_integrity()
                        rows.append(row)
                    except ValueError as error:
                        # Never train on leaked labels; keep an auditable count.
                        skipped += 1
                        rejected_rows.append({"index": i, "source_id": str(raw.get("id", i)),
                                              "reason": str(error)})
                        if skipped <= 5:
                            Progress.log("data-load", "SKIP invalid/leaked source row",
                                         task=task, split=dest, index=i, reason=str(error))
                    if i == 0 or (i+1) % 1000 == 0:
                        Progress.log("data-load", "mapping progress", task=task, split=dest,
                                     processed=i+1, total=total)
                if not rows:
                    raise ValueError(f"{task}/{dest}: every row rejected by leakage/integrity checks")
                audit = files.output / "dataset_audit" / f"{task}_{dest}_rejected.json"
                files.atomic(audit, rejected_rows)
                Dataset.from_list([row.to_storage() for row in rows]).to_parquet(
                    str(files.data / f"{task}_{dest}_raw.parquet")
                )
                Dataset.from_list([row.to_sft() for row in rows if row.training_answer]).to_parquet(
                    str(files.data / f"{task}_{dest}.parquet")
                )
                Progress.log("data-load", "dataset prepared with leakage gate",
                             task=task, split=dest, kept=len(rows), skipped=skipped)
                print("[data_load]", task, dest, len(rows), "skipped", skipped, flush=True)
        # RL reads the same normalized code train/eval data, no second mapping.

    @classmethod
    def audit_cached(cls):
        """Check *every* cached train/eval sample before SFT/eval/GRPO."""
        import pandas as pd
        failures = []
        totals = {}
        for task in cls.names:
            for split in ("train", "eval"):
                path = files.data / f"{task}_{split}_raw.parquet"
                if not path.exists():
                    raise FileNotFoundError(f"Prepare datasets before audit: {path}")
                table = pd.read_parquet(path)
                errors = 0
                for i, raw in enumerate(table.to_dict("records")):
                    try:
                        sample = Sample.from_dict(raw)
                        sample.assert_prompt_integrity()
                        if sample.training_answer:
                            sample.to_sft()
                    except (ValueError, TypeError) as error:
                        errors += 1
                        if errors <= 5:
                            Progress.log("prompt-audit", "BLOCKED cached row", task=task,
                                         split=split, index=i, id=raw.get("source_id"), reason=str(error))
                totals[f"{task}/{split}"] = {"checked": len(table), "invalid": errors}
                Progress.log("prompt-audit", "cached dataset checked", task=task,
                             split=split, checked=len(table), invalid=errors)
                if errors:
                    failures.append((task, split, errors))
        files.atomic(files.output / "dataset_audit" / "cached_report.json", totals)
        if failures:
            raise ValueError(f"BLOCKED: invalid/leaked cached rows: {failures}; rerun prepare")
        return totals


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 2 and sys.argv[1] == "audit":
        DataLoad.audit_cached()
    elif len(sys.argv) == 1:
        DataLoad().prepare()
    else:
        raise SystemExit("Usage: python -m custome.rl.data_load [audit]")
