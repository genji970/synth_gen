"""Run the exact RL synth_gen -> independent pytest -> Sample -> JSONL path.

No SFT, GRPO, PCA, or HF upload is run unless --upload-private is given.
This is the public smoke-test entrypoint; custome.main delegates to it.

Run after sourcing custome/config.sh and custome/rl/rl_config.sh:
    python -m custome.sample_test --seed 'Fibonacci with a twist.' --samples 1
"""
from __future__ import annotations

import argparse
import json
import os
import re
import traceback
from datetime import datetime
from concurrent.futures import wait, FIRST_COMPLETED
from pathlib import Path
import time

from custome.rl.core import PrivateDataset, Progress, env, files
from custome.rl.data_load import DataLoad, Sample, problem_leak_reason
from custome.rl.score import CodeVerifier


class SampleDiagnosis:
    """Single source for human-facing verdicts and short structured audit.

    This class NEVER decides training acceptance. It interprets existing
    synth_gen and Podman outputs without erasing original evidence.
    """

    @staticmethod
    def _clean(text):
        return re.sub(r"\x1b\[[0-9;]*m", "", str(text or ""))

    @classmethod
    def _failure_evidence(cls, record):
        evidence = []
        report = record.get("pytest_report") or {}
        for col in report.get("collectors", []):
            if col.get("outcome") == "failed":
                evidence.append(cls._clean(col.get("longrepr", "")))
        for test in report.get("tests", []):
            if test.get("outcome") != "passed":
                for field in ("setup", "call", "teardown"):
                    phase = test.get(field) or {}
                    if isinstance(phase, dict) and phase.get("longrepr"):
                        evidence.append(cls._clean(phase["longrepr"]))
        if not evidence:
            evidence.append(cls._clean(record.get("error_message") or ""))
        # Short, actionable excerpts; the full report stays in generated_code.jsonl.
        snippets = []
        for block in evidence:
            lines = block.splitlines()
            errors = [line.strip() for line in lines if
                      re.search(r"(?:^E\s+|Error:|Exception:|AssertionError|ModuleNotFoundError|NameError|Failed:|ImportError)", line)]
            if not errors:
                errors = [line.strip() for line in lines if line.strip()]
            for line in errors[:5]:
                if line and line not in snippets:
                    snippets.append(line[:450])
            if len(snippets) >= 6:
                break
        return snippets[:6]

    @classmethod
    def summarize(cls, record, *, index):
        pytest = record.get("pytest_summary") or {}
        summary = (record.get("pytest_report") or {}).get("summary") or {}
        if not pytest:
            pytest = summary
        passed_count = int(pytest.get("passed", 0) or 0)
        failed_count = int(pytest.get("failed", 0) or 0)
        error_count = int(pytest.get("error", pytest.get("errors", 0)) or 0)
        collected = max(int(pytest.get("collected", 0) or 0),
                        passed_count + failed_count)
        exit_code = record.get("pytest_exit_code")
        synth_passed = record.get("synth_gen_passed")
        podman_passed = record.get("passed") is True
        status = record.get("status") or "unknown"
        test_code = record.get("test_code") or ""
        evidence = cls._failure_evidence(record) if not podman_passed else []
        if status == "prompt_leak" or (record.get("question") and problem_leak_reason(record["question"])):
            kind, label = "prompt_leak", "문제에 정답 코드/풀이가 포함됨 — 학습에서 제외"
            action = "문제 생성 프롬프트 또는 재생성 횟수를 수정. 해당 질문을 학습에 사용하지 말 것"
        elif status == "environment_error":
            kind, label = "environment_error", "생성 워커/Podman/검증 환경 오류"
            action = "아래 예외와 워커 상세 로그를 확인. 코드의 정답 여부는 아직 판단 불가"
        elif status == "invalid_test":
            kind, label = "invalid_test", "생성된 테스트 자체가 유효하지 않음 — 코드 정오 미판정"
            action = "아래 테스트 생성 오류를 수정. 이 샘플은 학습에 채택하지 않음"
        elif status == "invalid_code":
            kind, label = "invalid_code", "모델 예측 코드의 Python 문법 오류"
            action = "모델 예측 코드의 SyntaxError를 수정. 문제 및 테스트 품질과 별개로 코드가 실행 불가"
        elif not test_code.strip():
            kind, label = "no_tests", "테스트 코드 없음"
            action = "테스트 생성 단계를 확인하고, 테스트가 만들어지기 전에는 학습에 사용하지 말 것"
        elif not podman_passed and (collected == 0 or exit_code == 2) and passed_count + failed_count == 0:
            kind, label = "collection_error", "pytest 수집/임포트 실패 (테스트 미실행)"
            action = "아래 오류와 생성된 테스트의 import·데코레이터를 먼저 수정. 코드 정오 판정 불가"
        elif not podman_passed and (failed_count > 0 or collected > 0):
            kind, label = "test_failed", "pytest 테스트 실행 후 실패"
            action = "생성 코드와 테스트의 기대값이 문제 설명과 일치하는지 비교"
        elif podman_passed and synth_passed is False:
            kind, label = "synth_verifier_failed", "독립 pytest 통과 / synth_gen 검증 실패"
            action = "원본 synth_gen 검증 실패 원인을 별도로 확인"
        elif podman_passed:
            kind, label = "passed", "synth_gen 및 독립 pytest 확인" if synth_passed else "독립 pytest 통과"
            action = "독립 pytest 통과; 별도의 PCA·학습 채택 판정은 아직 수행하지 않음"
        else:
            kind, label = "unknown_failure", "검증 실패 (상세 확인 필요)"
            action = "원본 pytest 보고서와 검증 이력을 확인"
        quality_warnings = []
        question = str(record.get("question") or "")
        if re.search(r"(?m)^\s*(?:def\s+|class\s+|import\s+|from\s+\S+\s+import\s+|@pytest\b|```(?:python)?)", question):
            quality_warnings.append("문제(question)에 구현 코드 또는 import가 포함됨: 정답 누출 가능성 확인")
        if "HumanMessage(" in question or "ChatPromptValue(" in question:
            quality_warnings.append("문제에 LangChain 객체의 문자열 표현이 포함됨: 프롬프트 생성 확인")
        if re.search(r"\bpytest\.", test_code) and not re.search(r"(?m)^\s*(?:import pytest|from pytest import)", test_code):
            quality_warnings.append("테스트가 pytest를 사용하지만 import pytest가 없음")
        predicted = str(record.get("generated_code") or "")
        if re.search(r"(?m)^\s*(?:from langchain(?:\.|\s)|import langchain\b)", predicted + "\n" + test_code):
            quality_warnings.append("코드/테스트에 LangChain 의존성 포함: 일반 알고리즘 문제에 불필요할 수 있음")
        if kind == "passed" and synth_passed is True:
            outcome = "검증 통과 (PCA 미검사)"
        elif kind in ("collection_error", "environment_error", "no_tests", "invalid_test"):
            outcome = "정답 여부 미판정"
        else:
            outcome = "검증 실패"
        return {
            "sample_number": int(index), "source_id": record.get("source_id"),
            "seed": record.get("seed", ""), "question": question,
            "predicted_code": record.get("generated_code", ""),
            "generated_test_code": test_code,
            "generator_model": record.get("generator_model", ""),
            "synth_gen_result": {
                "passed": synth_passed,
                "failed_verifier": record.get("synth_gen_failed_verifier") or "",
                "error": record.get("synth_gen_error") or "",
            },
            "independent_pytest": {
                "passed": podman_passed, "status": status,
                "exit_code": exit_code, "collected": collected,
                "passed_tests": passed_count, "failed_tests": failed_count,
                "errors": error_count, "failure_evidence": evidence,
            },
            "diagnosis": {
                "category": kind, "label": label, "outcome": outcome,
                "next_action": action, "quality_warnings": quality_warnings,
            },
            "pca_checked": False, "training_executed": False,
            "accepted_for_training": False,
            "full_record_file": str(files.code_records("ours")),
        }

    @staticmethod
    def render(info, total=None):
        n = info["sample_number"]
        suffix = f"/{total}" if total is not None else ""
        header = f"{'='*16} SAMPLE {n}{suffix} {'='*16}"
        g = info["synth_gen_result"]
        p = info["independent_pytest"]
        d = info["diagnosis"]
        lines = ["", header,
                 f"[1] SEED: {info['seed']}",
                 "[2] QUESTION / PROBLEM:", info["question"] or "(생성 실패)",
                 "[3] MODEL PREDICTION / CODE:", info["predicted_code"] or "(코드 없음)",
                 "[4] GENERATED PYTEST:", info["generated_test_code"] or "(테스트 없음)",
                 "[5] VERDICTS:",
                 f"  synth_gen: {'PASS' if g['passed'] is True else 'FAIL' if g['passed'] is False else 'NOT RUN'}"
                 + (f" | failed_verifier={g['failed_verifier']}" if g["failed_verifier"] else ""),
                 (f"  Podman pytest: SKIPPED | status={p['status']} | invalid input before execution"
                  if p['status'] in ('invalid_test', 'invalid_code', 'prompt_leak', 'no_tests')
                  else f"  Podman pytest: {'PASS' if p['passed'] else 'FAIL'}"
                       f" | collected={p['collected']} passed={p['passed_tests']} failed={p['failed_tests']}"
                       f" errors={p['errors']} exit_code={p['exit_code']}"),
                 f"  FINAL: {d['outcome']} — {d['label']}",
                 "[6] FAILURE REASON / EVIDENCE:"]
        if g["passed"] is False:
            synth_error = SampleDiagnosis._clean(g["error"]).strip() or "(실패 사유 기록 없음)"
            if len(synth_error) > 1500:
                synth_error = synth_error[:1500] + " ... [remaining text in raw JSONL]"
            lines.append("  synth_gen: " + synth_error)
        if p["failure_evidence"]:
            lines.extend(f"  pytest: {message}" for message in p["failure_evidence"])
        if g["passed"] is not False and not p["failure_evidence"]:
            lines.append("  (보고된 실패 오류 없음)")
        lines += ["[7] WHAT TO FIX / CHECK:", f"  {d['next_action']}"]
        lines += [f"  WARNING: {x}" for x in d["quality_warnings"]]
        lines += ["  *수동 테스트: PCA 미검사 / RL 학습 미실행*",
                  f"  Full raw JSONL: {info['full_record_file']}", ""]
        return "\n".join(lines)


class CodeGenerationSmokeTest:
    """One reusable, training-equivalent verification/data recording path.

    Uses the SAME CodeGenerationPool from main.py, CodeVerifier from score.py,
    DataLoad mapper and PrivateDataset used by AsyncProducer. There is no
    second synth_gen, pytest, or data-schema implementation here.
    """

    def __init__(self, *, seed, samples, use_latest=False,
                 upload_private=False, strict=False, verbose=False):
        self.seed = seed
        self.samples = samples
        self.use_latest = use_latest
        self.upload_private = upload_private
        self.strict = strict
        self.verbose = verbose
        # Isolate repeated smoke-test runs, even when RL_RUN_ID is reused.
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_pid{os.getpid()}"
        self.session_dir = files.track("ours") / "smoke_reports" / self.session_id
        self.summary_path = self.session_dir / "samples.jsonl"
        self.report_path = self.session_dir / "samples_readable.txt"
        if self.samples < 1:
            raise ValueError("--samples must be >= 1")
        # Prevent an unnoticed mix of model families across stages.
        if env("RL_MODEL") != env("RL_BASE_MODEL"):
            raise ValueError("RL_MODEL must equal RL_BASE_MODEL from custome/config.sh")

    def model_snapshot(self):
        if self.use_latest:
            pointer = files.track("ours") / "policy_latest.json"
            value = files.read(pointer)
            if not value or not value.get("path"):
                raise FileNotFoundError(f"No completed actor snapshot: {pointer}")
            model_path = Path(str(value["path"]))
            if not model_path.is_dir() or not (model_path / "config.json").is_file():
                raise FileNotFoundError(f"Actor snapshot has not finished publishing: {model_path}")
            return {"path": str(model_path), "policy_step": int(value["policy_step"])}
        # No hardcoded 7B or secondary synth_gen model. A completed --latest
        # actor remains the same model family, but uses fine-tuned weights.
        return {"path": env("RL_BASE_MODEL"), "policy_step": -1}

    @staticmethod
    def check_mapping(sample: Sample, independent: dict):
        """Fail visibly if the code, test, verdict, or prompt was mismapped."""
        assert sample.kind == sample.task == "code", "Wrong canonical data task"
        assert sample.source == "synth_gen", "Wrong source mapper"
        assert sample.generated_code == independent["generated_code"], "Generated code mismatch"
        assert sample.test_code == independent["test_code"], "Test code mismatch"
        assert sample.passed is independent["passed"], "Pytest verdict mismatch"
        assert sample.pytest_exit_code == independent["pytest_exit_code"], "Pytest exit code mismatch"
        assert sample.pytest_report == independent["pytest_report"], "Pytest report mismatch"
        assert sample.pytest_stdout == independent["pytest_stdout"], "Pytest stdout mismatch"
        assert sample.pytest_stderr == independent["pytest_stderr"], "Pytest stderr mismatch"
        assert sample.question in sample.prompt, "Generated problem missing in model prompt"
        assert "Write runnable Python code only" in sample.prompt, "Wrong code prompt template"
        assert sample.accepted is False, "Unfiltered sample must never enter GRPO"
        assert sample.metadata["smoke_test"]["independent_pytest_passed"] is sample.passed
        if sample.passed:
            summary = sample.pytest_summary
            assert sample.test_code.strip(), "Reported success without actual tests"
            assert sample.generated_code.strip(), "Reported success without code"
            assert sample.pytest_exit_code == 0, "Reported success with non-zero exit code"
            assert int(summary.get("passed", 0)) > 0, "Reported success with no passing test"
            assert int(summary.get("failed", 0)) == 0, "Reported success despite failing test"
            assert int(summary.get("error", 0)) == 0, "Reported success despite pytest error"
        # Actual JSONL-to-schema round trip, including the full pytest report.
        restored = Sample.from_dict(sample.to_storage())
        assert restored.generated_code == sample.generated_code
        assert restored.test_code == sample.test_code
        assert restored.pytest_report == sample.pytest_report
        assert restored.verification_history == sample.verification_history
        return True

    def verify_and_record(self, generated, verifier, dataset):
        original_code = str(generated.get("code") or "")
        original_test = str(generated.get("test_code") or "")
        # Reuse EXACTLY the verifier called for RL candidates, not a mock or
        # a new local exec() implementation. Exceptions are recorded as failures.
        started = time.monotonic()
        if self.verbose:
            Progress.log("smoke", "independent Podman verification started", code_chars=len(original_code), test_chars=len(original_test))
        leak = problem_leak_reason(generated.get("problem") or generated.get("question") or "")
        if leak:
            # A leaked solution is a bad QUESTION, not a model prediction
            # failure. Save it, show the precise reason, skip expensive Podman.
            result = CodeVerifier.result(status="prompt_leak", error_message=leak,
                                         error_types=["PromptLeak"], test_code=original_test)
        else:
            with Progress.watch("smoke", "still running independent Podman pytest"):
                result = verifier.verify(original_code, original_test)
        if self.verbose:
            Progress.log("smoke", "independent Podman verification complete", passed=result.get("passed"), status=result.get("status"), elapsed_s=round(time.monotonic()-started,1))
        generated = dict(generated)
        generated.update(result)
        synth_result = generated.get("synth_gen_passed")
        # Two collection failures must NEVER count as agreement on correctness.
        def executed_tests(summary):
            return int(summary.get("passed", 0) or 0) + int(summary.get("failed", 0) or 0) > 0
        synth_executed = any(
            r.get("event") == "pytest_execution" and
            executed_tests((r.get("pytest_report") or {}).get("summary", {}))
            for r in generated.get("verification_history") or []
        )
        independent_executed = executed_tests(result.get("pytest_summary") or {})
        consistent = (bool(synth_result) == bool(result["passed"])
                      if synth_result is not None and synth_executed and independent_executed else None)
        metadata = dict(generated.get("metadata") or {})
        metadata["smoke_test"] = {
            "synth_vs_independent_pytest_agree": consistent,
            "synth_gen_passed": synth_result,
            "independent_pytest_passed": result["passed"],
            "pca_checked": False,
            "training_executed": False,
        }
        generated["metadata"] = metadata
        generated["accepted"] = False  # PCA has not been run in this smoke test.
        generated["rejection_reason"] = "manual_smoke_test_only"
        generated["success_reason"] = ""  # Verified != PCA-qualified.
        generated["failure_stage"] = (
            "problem_generation" if leak else
            "generated_test" if result.get("status") == "invalid_test" else
            "predicted_code" if result.get("status") == "invalid_code" else
            "synth_gen_verifier" if generated.get("synth_gen_passed") is False else
            "independent_pytest" if not result["passed"] else
            "pca_not_run"
        )
        generated["failure_reason"] = (
            leak if leak else
            result.get("error_message") if result.get("status") in ("invalid_test", "invalid_code") else
            generated.get("synth_gen_error") if generated.get("synth_gen_passed") is False else
            result.get("error_message") if not result["passed"] else
            "No PCA reference batch in standalone smoke test"
        )
        # Preserve the per-execution matched input and status, even if synth_gen
        # changed its own tests during the repair loop.
        history = list(generated.get("verification_history") or [])
        history.append({
            "verifier": "IndependentRLCodeVerifier",
            "event": "final_pytest_execution",
            "code": original_code,
            "test_code": original_test,
            "passed": result["passed"],
            "status": result["status"],
            "pytest_report": result.get("pytest_report", {}),
            "pytest_exit_code": result.get("pytest_exit_code"),
            "pytest_stdout": result.get("pytest_stdout", ""),
            "pytest_stderr": result.get("pytest_stderr", ""),
            "error_message": result.get("error_message", ""),
        })
        generated["verification_history"] = history
        if self.verbose:
            Progress.log("smoke", "normalizing and checking schema")
        sample = DataLoad().synth(generated)
        self.check_mapping(sample, {"generated_code": original_code, **result})
        normalized = sample.to_dict()
        dataset.append([normalized])
        if self.verbose:
            Progress.log("smoke", "sample saved", passed=normalized["passed"], jsonl=dataset.path)
        return normalized

    def print_result(self, record, index):
        """Print exactly one readable sample audit; keep raw history in JSONL."""
        info = SampleDiagnosis.summarize(record, index=index)
        report = SampleDiagnosis.render(info, total=self.samples)
        print(report, flush=True)
        # Write the SAME audit to a readable file, not just to terminal history.
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        with self.report_path.open("a", encoding="utf-8") as handle:
            handle.write(report + "\n")
        files.append(self.summary_path, [info])
        return info

    def run(self):
        from custome.main import CodeGenerationPool

        started = time.monotonic()
        print(f"[smoke] samples={self.samples}; each sample shows question / prediction / tests / verdict / failure", flush=True)
        snapshot = self.model_snapshot()
        print(f"[model] canonical base: {env('RL_BASE_MODEL')}", flush=True)
        print(f"[model] sampling weights: {snapshot['path']}", flush=True)
        print(f"[config] processes={env('RL_SYNTH_PROCESSES', int, '1')}, "
              f"in_flight={env('RL_SYNTH_BATCH_SIZE', int, '2')}, "
              f"samples={self.samples}", flush=True)
        verifier = CodeVerifier()
        dataset = PrivateDataset("generated", "ours")
        records = []
        record_index = {}
        pending_limit = max(1, env("RL_SYNTH_BATCH_SIZE", int))
        # Reuse one generator process pool; each worker caches the model weights.
        worker_log_dir = None if self.verbose else str(self.session_dir / "workers")
        with CodeGenerationPool(worker_log_dir=worker_log_dir) as pool:
            for offset in range(0, self.samples, pending_limit):
                size = min(pending_limit, self.samples - offset)
                jobs = {
                    pool.submit(snapshot, {"reference_step": "standalone_smoke_test"},
                                -1, offset + i + 1, self.seed): offset + i + 1
                    for i in range(size)
                }
                pending = set(jobs)
                print(f"[progress] dispatched={size} completed={len(records)}/{self.samples}", flush=True)
                while pending:
                    done, pending = wait(pending, timeout=Progress.interval(), return_when=FIRST_COMPLETED)
                    if not done:
                        print(f"[progress] generating={len(pending)} completed={len(records)}/{self.samples} "
                              f"elapsed={round(time.monotonic()-started)}s", flush=True)
                    for future in done:
                        idx = jobs[future]
                        print(f"[progress] sample={idx}/{self.samples} generated; running independent Podman pytest...", flush=True)
                        try:
                            generated = future.result()
                            normalized = self.verify_and_record(generated, verifier, dataset)
                        except Exception as exc:
                            # Keep an actionable per-sample report even when a worker
                            # crashes or the data mapper refuses the generated record.
                            print(f"[error] sample={idx} unhandled: {type(exc).__name__}: {exc}", flush=True)
                            normalized = {
                                "source_id": None, "seed": self.seed,
                                "question": "(generation/verification exception)",
                                "generated_code": "", "test_code": "",
                                "synth_gen_passed": False,
                                "synth_gen_failed_verifier": type(exc).__name__,
                                "synth_gen_error": f"{type(exc).__name__}: {exc}",
                                "passed": False, "status": "environment_error",
                                "error_message": f"{type(exc).__name__}: {exc}",
                                "pytest_exit_code": None, "pytest_summary": {},
                                "pytest_report": {},
                            }
                            self.session_dir.mkdir(parents=True, exist_ok=True)
                            with (self.session_dir / "exceptions.log").open("a", encoding="utf-8") as fout:
                                fout.write(f"SAMPLE {idx}\n" + traceback.format_exc() + "\n")
                        self.print_result(normalized, idx)
                        records.append(normalized)
                        record_index[idx] = normalized
                        print(f"[progress] completed={len(records)}/{self.samples} sample={idx}", flush=True)
        if self.upload_private:
            Progress.log("smoke", "uploading accumulated JSONL to PRIVATE Hugging Face dataset")
            uploaded = dataset.upload(flush=True)
            print(f"\n[HF private upload] new records uploaded: {uploaded}", flush=True)
        print("\n================ FINAL SUMMARY (sorted by sample number) ================", flush=True)
        summary_by_id = [SampleDiagnosis.summarize(row, index=i) for i, row in sorted(record_index.items())]
        # Live terminal prints as each future finishes; saved text is reordered.
        self.report_path.write_text(
            "\n".join(SampleDiagnosis.render(info, total=self.samples)
                      for info in summary_by_id) + "\n", encoding="utf-8")
        for summary in summary_by_id:
            p = summary["independent_pytest"]
            print(f"SAMPLE {summary['sample_number']}: {summary['diagnosis']['outcome']} "
                  f"| {summary['diagnosis']['category']} "
                  f"| passed={p['passed_tests']} failed={p['failed_tests']} collected={p['collected']}", flush=True)
        print(f"[readable report] {self.report_path}", flush=True)
        print(f"[concise JSONL] {self.summary_path}", flush=True)
        print(f"[raw history JSONL] {dataset.path}", flush=True)
        if worker_log_dir:
            print(f"[worker details] {worker_log_dir}/worker_<pid>.log", flush=True)
        print(f"[verified] independent pytest pass: {sum(bool(r.get('passed')) for r in records)}/{len(records)}", flush=True)
        print(f"[elapsed] {round(time.monotonic()-started)}s", flush=True)
        if self.strict and not all(r["passed"] and r.get("synth_gen_passed") for r in records):
            raise RuntimeError("--strict: one or more generated examples failed synth_gen or independent pytest")
        return records


def main():
    parser = argparse.ArgumentParser(description="Test the REAL RL synthesis path, without starting training")
    parser.add_argument("--seed", default="Fibonacci with a twist.")
    parser.add_argument("--samples", type=int, default=env("RL_SYNTH_TEST_SAMPLES", int))
    parser.add_argument("--latest", action="store_true", help="Use latest complete GRPO actor snapshot")
    parser.add_argument("--upload-private", action="store_true", help="Upload locally saved JSONL to a PRIVATE HF dataset")
    parser.add_argument("--strict", action="store_true", help="Fail if any synth_gen/pytest check fails")
    parser.add_argument("--verbose", action="store_true", help="Print original interleaved worker/verifier logs in terminal (default: saved separately)")
    args = parser.parse_args()
    if not __import__("os").environ.get("RL_OUT"):
        parser.error("Source custome/config.sh then custome/rl/rl_config.sh first")
    CodeGenerationSmokeTest(
        seed=args.seed, samples=args.samples,
        use_latest=args.latest, upload_private=args.upload_private, strict=args.strict,
        verbose=args.verbose,
    ).run()


if __name__ == "__main__":
    main()
