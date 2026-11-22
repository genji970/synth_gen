"""Shared synth_gen entrypoint used by both the GRPO producer and manual tests.

The training pipeline imports ``CodeGenerationPool`` from this module; it never
re-implements synth_gen's generator, verifiers, test authoring or fix loop.
Each spawn worker owns an inference-only snapshot of the SAME trained GRPO actor.

Standalone smoke test (does not launch training):
  python -m custome.sample_test --seed 'Fibonacci with a twist.'
"""
from __future__ import annotations

import ast
import gc
import re
import multiprocessing as mp
import os
import traceback
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from uuid import uuid4

from custome.rl.core import Progress, env, files
from custome.rl.data_load import prompt_content, problem_leak_reason


def synth_problem_instruction(seed):
    """Text-only topical seed, never a reference code/solution or chat repr."""
    return (
        "Create ONE original, self-contained Python programming problem. "
        "Write the problem statement in natural-language prose ONLY. "
        "Clearly specify the exact function name, its purpose, inputs, outputs, "
        "indexing convention, initial conditions, all numeric constants, and edge cases. "
        "Every expected output must be mathematically well-defined; give a finite "
        "stopping rule for iterative algorithms. Never write ambiguous constants "
        "such as 'e.g. modulo 1000000007' or contradictory conditions. "
        "Do not include examples with worked answers or reference outputs. "
        "Do not write Python source code, a function definition, a code skeleton, "
        "an algorithm implementation, tests, expected answers, a solution, "
        "worked reasoning, or markdown code fences. "
        "Do not repeat source code from the seed. "
        "Return ONLY the problem statement.\n\n"
        "TOPIC SEED (inspiration only, NOT an answer):\n" + prompt_content(seed)
    )



def _prose_function_signatures(problem):
    """Convert signature-only fenced snippets to prose; never permit implementations."""
    def convert(match):
        source = match.group(1).strip()
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return match.group(0)
        if len(tree.body) != 1 or not isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
            return match.group(0)
        fn = tree.body[0]
        body = fn.body
        if len(body) != 1 or not (
            isinstance(body[0], ast.Pass)
            or isinstance(body[0], ast.Expr) and (
                isinstance(body[0].value, ast.Constant) and (
                    isinstance(body[0].value.value, str) or body[0].value.value is Ellipsis
                )
            )
        ):
            return match.group(0)
        signature = source.splitlines()[0].strip()
        if not signature.startswith(("def ", "async def ")) or not signature.endswith(":"):
            return match.group(0)
        return "Function signature: " + signature[:-1]
    return re.sub(r"```(?:python|py)\s*\n(.*?)```", convert, problem, flags=re.I | re.S)


def synth_pytest_instruction(problem, predicted_code, signature="", feedback=""):
    """Build pytest instructions from the problem and signature only."""
    return (
        "Programming language: Python. Write executable Python pytest tests for the problem below. "
        "[PROBLEM SPECIFICATION / SOURCE OF TRUTH]\n"
        + prompt_content(problem).strip()
        + "\n\n[EXPECTED FUNCTION SIGNATURE]\n"
        + prompt_content(signature).strip()
        + "\n\n[TASK]\n"
        "Return ONLY runnable Python pytest test code (no prose or Markdown fences).\n"
        "The implementation already exists in the test runtime as a global function. "
        "Do NOT implement it, stub it, mock it, or import a second copy. "
        "Only define functions whose names start with test_. "
        "Write assertions calling that existing global function directly.\n"
        "Rules:\n"
        "1. Test the implementation against the PROBLEM requirements, not against its own output.\n"
        "2. Derive expected answers independently from the problem specification. "
        "NEVER use a call to the predicted function to compute expected values, "
        "reference values, gold labels, or test oracles.\n"
        "3. Do NOT copy, redefine, patch, or reimplement the predicted function or class.\n"
        "4. Define at least one discoverable `def test_...():` function with meaningful assertions.\n"
        "5. Import `pytest` if you use `pytest` decorators, `pytest.raises`, or other pytest APIs.\n"
        "6. Write ordinary, relevant boundary cases only if the problem specifies their behavior.\n"
        "7. Do not assume an example output, indexing convention, or formula not supported by the problem.\n"
        "8. Every test must call the predicted function and compare against an independent expected result.\n"
        "9. Do not include top-level assertions or run tests directly at import time.\n"
        + ("\n[PREVIOUS INVALID TEST FEEDBACK]\n" + prompt_content(feedback) +
           "\nWrite a NEW valid test instead.\n" if feedback else "")
    )


def generate_pytest_for_prediction(problem, predicted_code, llm, signature="",
                                    call_model=None, attempts=None, audit=None):
    """Use the current actor to generate tests from PROBLEM + SIGNATURE.

    Reject syntactically invalid, vacuous, self-oracling, or otherwise clearly
    unsound generated tests before ever executing them in Podman. The checker
    cannot establish mathematical correctness of arbitrary test expectations.
    """
    from custome.rl.score import CodeVerifier, code_only
    if not prompt_content(problem).strip() or not prompt_content(predicted_code).strip():
        raise ValueError("InvalidGeneratedTest: a nonempty problem and predicted code are required")
    if call_model is None:
        from synth_gen.generation.LLM import call_llm
        call_model = lambda prompt: call_llm(prompt, llm=llm, use_cache=False)
    if attempts is None:
        attempts = env("RL_SYNTH_TEST_RETRIES", int, "3")
    if not (1 <= attempts <= 10):
        raise ValueError("RL_SYNTH_TEST_RETRIES must be between 1 and 10")
    feedback = ""
    last = "no tests produced"
    for i in range(attempts):
        prompt = synth_pytest_instruction(problem, predicted_code, signature, feedback)
        raw_response = prompt_content(call_model(prompt))
        test_code = code_only(raw_response)
        last = CodeVerifier.generated_test_issue(prompt_content(predicted_code), test_code)
        if audit is not None:
            audit({
                "verifier": "GeneratedPytestInputGuard",
                "event": "pytest_generation_attempt",
                "attempt": i + 1,
                "problem": prompt_content(problem),
                "code": prompt_content(predicted_code),
                "generation_prompt": prompt,
                "raw_response": raw_response,
                "test_code": test_code,
                "passed": not bool(last),
                "error": last,
            })
        if not last:
            return test_code
        Progress.log("synth-test", "generated pytest rejected BEFORE Podman",
                     attempt=i + 1, max_attempts=attempts, issue=last)
        feedback = last
    raise ValueError(f"InvalidGeneratedTest: after {attempts} attempts: {last}")


def _visible_gpu_count():
    """Number of CUDA devices visible AFTER config.sh's CUDA_VISIBLE_DEVICES mask."""
    import torch
    return int(torch.cuda.device_count()) if torch.cuda.is_available() else 0


def _worker_cuda_plan(worker_ordinal=None, gpu_count=None, processes=None, budget=None):
    """Round-robin GPU allocation; divide each GPU's PyTorch budget by its users.

    This limits PyTorch's caching allocator, not all GPU allocations and not
    the physical device's memory. No CPU fallback is permitted.
    """
    if processes is None:
        processes = env("RL_SYNTH_PROCESSES", int)
    if gpu_count is None:
        gpu_count = _visible_gpu_count()
    if budget is None:
        budget = env("RL_SYNTH_GPU_MEMORY_FRACTION", float)
    if not 0 < budget <= 1:
        raise ValueError("RL_SYNTH_GPU_MEMORY_FRACTION must be in (0, 1]")
    if processes < 1:
        raise ValueError("RL_SYNTH_PROCESSES must be >= 1")
    if gpu_count < 1:
        raise RuntimeError(
            "No CUDA GPU visible to synth_gen. Check CUDA_VISIBLE_DEVICES in "
            "custome/config.sh, NVIDIA drivers, and torch.cuda.is_available(). "
            "CPU fallback is intentionally disabled."
        )
    if worker_ordinal is None:
        identity = mp.current_process()._identity
        worker_ordinal = (identity[0] - 1) if identity else 0
    # Number of workers on the selected CUDA device (some devices may be idle).
    slot = int(worker_ordinal) % processes
    gpu = slot % gpu_count
    workers_on_gpu = len(range(gpu, processes, gpu_count))
    return f"cuda:{gpu}", budget / workers_on_gpu, workers_on_gpu


def _worker_device():
    """Each spawn worker picks a CUDA device from the visible GPU inventory."""
    return _worker_cuda_plan()[0]


class VerifiedCodeGenerator:
    """One and only adapter around the original PythonVerifiedSynthDataGenerator.

    Reuses synth_gen's existing correction loop, seed source, verifiers, and
    PythonTestBasedVerifier. Only the LLM adapter and question seed prompt are
    customized; the model is the frozen snapshot of the training actor.
    """

    def __init__(self, device=None):
        # A worker owns one full actor replica. Respect a shared allocator
        # budget when multiple workers use one physical GPU.
        if device is not None and not str(device).startswith("cuda:"):
            raise ValueError("synth_gen runs on CUDA only; CPU fallback is disabled")
        self.device, self.gpu_fraction, self.workers_on_gpu = _worker_cuda_plan()
        if device is not None and device != self.device:
            raise ValueError("Explicit CUDA device conflicts with automatic worker allocation")
        import torch
        device_index = int(self.device.split(":", 1)[1])
        torch.cuda.set_device(device_index)
        torch.cuda.set_per_process_memory_fraction(self.gpu_fraction, device=device_index)
        total_memory = torch.cuda.get_device_properties(device_index).total_memory
        Progress.log("synth-gpu", "worker allocated GPU VRAM budget",
                     device=self.device, processes_sharing_gpu=self.workers_on_gpu,
                     allocator_fraction=round(self.gpu_fraction, 4),
                     allocator_limit_gib=round(total_memory*self.gpu_fraction/1024**3, 2),
                     gpu_total_gib=round(total_memory/1024**3, 2))
        self.model = None
        self.tokenizer = None
        self.path = None
        self.policy_step = -1
        self.verification_history = []
        Progress.log("synth-worker", "initializing generator", device=self.device)
        self.generator = self._make_generator()
        Progress.log("synth-worker", "generator initialized", verifiers=len(self.generator.verifiers))

    def _make_generator(self):
        from langchain_core.language_models.chat_models import BaseChatModel
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult
        from synth_gen.generators import PythonVerifiedSynthDataGenerator
        import synth_gen.verification.python_verifiers as pv
        from custome.podman_generating import run_pytest, run_test_coverage

        # Preserve the existing WSL/Podman backend, but retain every pytest
        # execution, including tests internally rewritten by synth_gen.
        owner = self

        def audited_pytest(script, *args, **kwargs):
            Progress.log("synth-pytest", "starting remote Podman pytest", source_chars=len(script))
            started = time.monotonic()
            try:
                active = getattr(owner, "_active_test_record", None)
                if active is not None:
                    from custome.rl.score import CodeVerifier
                    # Exactly the test embedded in the script must have passed
                    # the same static checks as the final independent verifier.
                    current_code = prompt_content(getattr(active, "code", "") or "")
                    current_test = prompt_content(getattr(active, "test_code", "") or "")
                    expected_suffix = "\n\n\n" + current_test
                    if not script.endswith(expected_suffix):
                        raise ValueError("InvalidGeneratedTest: executed tests differ from the audited test_code")
                    issue = CodeVerifier.generated_test_issue(current_code, current_test)
                    if issue:
                        raise ValueError("InvalidGeneratedTest: " + issue)
                result = run_pytest(script, *args, **kwargs)
                report, stdout, stderr, exit_code = result
                Progress.log("synth-pytest", "pytest returned", exit_code=exit_code, elapsed_s=round(time.monotonic()-started, 1))
                owner.verification_history.append({
                    "verifier": "PythonTestBasedVerifier",
                    "event": "pytest_execution",
                    "executed_source": str(script),
                    "pytest_report": report or {},
                    "pytest_stdout": str(stdout or ""),
                    "pytest_stderr": str(stderr or ""),
                    "pytest_exit_code": exit_code,
                })
                return result
            except BaseException as exc:
                Progress.log("synth-pytest", "pytest environment error", error=repr(exc), elapsed_s=round(time.monotonic()-started, 1))
                owner.verification_history.append({
                    "verifier": "PythonTestBasedVerifier",
                    "event": "pytest_environment_error",
                    "executed_source": str(script),
                    "error": f"{type(exc).__name__}: {exc}",
                })
                raise

        pv.run_pytest = audited_pytest
        pv.run_test_coverage = run_test_coverage

        class CurrentActorChat(BaseChatModel):
            @property
            def _llm_type(self):
                return "current_grpo_actor_snapshot"

            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                import torch
                if owner.model is None:
                    raise RuntimeError("Load the GRPO actor snapshot before generating")
                dialogs = [{
                    "role": "assistant" if m.type == "ai" else
                            "system" if m.type == "system" else "user",
                    "content": ("Programming language: Python. Generate only Python code; "
                                "all test code must use Python pytest.\n" + prompt_content(m))
                               if m.type == "human" else prompt_content(m),
                } for m in messages]
                Progress.log("synth-llm", "tokenizing request", step=getattr(owner, "current_step", "?"), attempt=getattr(owner, "current_attempt", "?"), message_count=len(dialogs))
                encoded = owner.tokenizer.apply_chat_template(
                    dialogs, add_generation_prompt=True,
                    return_tensors="pt", return_dict=True,
                )
                ids = encoded["input_ids"].to(owner.model.device)
                attention_mask = encoded["attention_mask"].to(owner.model.device)
                Progress.log("synth-llm", "generating tokens", prompt_tokens=ids.shape[1], max_new_tokens=env("RL_SYNTH_NEW_TOKENS", int))
                started = time.monotonic()
                with Progress.watch("synth-llm", "model still generating tokens", step=getattr(owner,"current_step","?"), attempt=getattr(owner,"current_attempt","?")):
                    with torch.inference_mode():
                        output = owner.model.generate(
                            input_ids=ids,
                            attention_mask=attention_mask,
                            max_new_tokens=env("RL_SYNTH_NEW_TOKENS", int),
                            do_sample=True,
                            temperature=env("RL_SYNTH_TEMP", float),
                            top_p=env("RL_SYNTH_TOP_P", float),
                            pad_token_id=(owner.tokenizer.pad_token_id if owner.tokenizer.pad_token_id is not None
                                          else owner.tokenizer.eos_token_id),
                        )
                text = owner.tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True)
                Progress.log("synth-llm", "generation completed", new_tokens=int(output.shape[1]-ids.shape[1]), elapsed_s=round(time.monotonic()-started, 1))
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

        class ProblemGenerator(PythonVerifiedSynthDataGenerator):
            def generate_prompt(self, seed, generation):
                seed_text = prompt_content(seed)
                Progress.log("synth", "generating NEW text-only problem", seed_chars=len(seed_text))
                attempts = env("RL_SYNTH_PROBLEM_RETRIES", int, "4")
                if attempts < 1:
                    raise ValueError("RL_SYNTH_PROBLEM_RETRIES must be >= 1")
                reason = "no generated problem"
                for i in range(attempts):
                    # No previous model outputs/answers are passed back to the
                    # problem generator. Even an invalid code question is not
                    # used as a subsequent seed.
                    text = prompt_content(self.call_llm(
                        synth_problem_instruction(seed_text),
                        extract_triple_backtick=False,
                    )).strip()
                    text = _prose_function_signatures(text)
                    reason = problem_leak_reason(text)
                    if not reason:
                        generation.problem = text
                        owner.rejected_problem_text = ""
                        Progress.log("synth", "clean problem generated", chars=len(text),
                                     generation_attempt=i+1)
                        return self.convert_prompt(text, generation)
                    owner.rejected_problem_text = text
                    Progress.log("synth", "REJECT generated problem containing code/answer",
                                 reason=reason, generation_attempt=i+1, allowed_attempts=attempts)
                raise ValueError(
                    f"Generated problem rejected after {attempts} attempts: {reason}"
                )

            def _generate_response(self, generation):
                # Upstream PythonVerifiedSynthDataGenerator passes
                # str(generation.prompt), which serializes ChatPromptValue into
                # messages=[HumanMessage(content=...)]. Never send that repr.
                from langchain_core.messages import AIMessage
                from synth_gen.generation.parsing import extract_source_code
                from synth_gen.generation.programming import generate_programming_solution
                problem = prompt_content(generation.problem or generation.prompt)
                reason = problem_leak_reason(problem)
                if reason:
                    raise ValueError(f"Refusing solution generation from leaked question: {reason}")
                Progress.log("synth", "generating Python solution", problem_chars=len(problem))
                solution = prompt_content(generate_programming_solution(
                    problem=problem, llm=self.llm, programming_language="python",
                ))
                generation.code = extract_source_code(solution)
                generation.response = AIMessage(content=solution)
                Progress.log("synth", "solution generated", code_chars=len(generation.code or ""))
                return generation

            def _fix_response(self, generation, verification):
                # The inherited fix path already takes message.content, not
                # str(ChatPromptValue). Keep synth_gen's original repair logic.
                Progress.log("synth", "repairing failed Python code", verifier=verification.verifier_name, error=str(verification.error_message)[:160])
                result = super()._fix_response(generation, verification)
                Progress.log("synth", "repair response received", repaired=result is not None)
                return result

        generator = ProblemGenerator()
        generator.seed_source = None  # Every RL seed comes from the previous input batch.
        generator.verbose = bool(env("RL_SYNTH_VERBOSE", int))
        generator.set_llm(CurrentActorChat())
        # Instrument existing synth_gen verifiers rather than reimplementing them.
        # Capture the *actual* code, test and result for every verification/fix round.
        for verifier in generator.verifiers:
            # Upstream Verifier.message_to_string() returns str(ChatPromptValue)
            # when there are multiple messages. Preserve only their contents
            # for test generation/repair and code extraction too.
            verifier.message_to_string = prompt_content
            original_check = verifier.check
            name = type(verifier).__name__

            def traced_check(record, check=original_check, verifier_name=name):
                started = time.monotonic()
                Progress.log("synth-verify", "checking", verifier=verifier_name)
                try:
                    if verifier_name == "PythonTestBasedVerifier":
                        # Generate tests from the problem and signature only.
                        owner._active_test_record = record
                        if not getattr(record, "test_code", None):
                            from synth_gen.generation.LLM_based_parsing import extract_programming_function_signature
                            predicted_code = prompt_content(getattr(record, "code", "") or "")
                            problem = prompt_content(getattr(record, "problem", "") or getattr(record, "prompt", ""))
                            if not getattr(record, "signature", None):
                                record.signature = extract_programming_function_signature(
                                    predicted_code, programming_language="python")
                            record.test_code = generate_pytest_for_prediction(
                                problem, predicted_code, generator.llm,
                                signature=prompt_content(record.signature or ""),
                                audit=owner.verification_history.append,
                            )
                            owner.verification_history.append({
                                "verifier": "GeneratedPytestInputGuard", "event": "pytest_generated",
                                "problem": problem, "code": predicted_code,
                                "test_code": record.test_code, "passed": True,
                                "prompt_basis": "problem_and_signature",
                            })
                        result = check(record, llm=generator.llm)
                    else:
                        result = check(record)
                except BaseException as exc:
                    Progress.log("synth-verify", "verifier exception", verifier=verifier_name, error=repr(exc))
                    owner.verification_history.append({
                        "verifier": verifier_name, "passed": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "code": str(getattr(record, "code", "") or ""),
                        "test_code": str(getattr(record, "test_code", "") or ""),
                        "exception": True,
                    })
                    raise
                passed, error = (result[0], result[1]) if isinstance(result, tuple) else (
                    getattr(result, "is_verified", False), getattr(result, "error_message", None)
                )
                Progress.log("synth-verify", "check finished", verifier=verifier_name, passed=bool(passed), elapsed_s=round(time.monotonic()-started, 1), error=str(error or "")[:140])
                owner.verification_history.append({
                    "verifier": verifier_name, "passed": bool(passed),
                    "error": str(error or ""),
                    "code": str(getattr(record, "code", "") or ""),
                    "test_code": str(getattr(record, "test_code", "") or ""),
                    "raw_response": str(getattr(getattr(record, "response", None), "content", "") or ""),
                })
                return result

            verifier.check = traced_check
        return generator

    def load_actor(self, snapshot):
        """Reload only after a fully merged actor snapshot was atomically published."""
        from custome.rl.core import require_model_lineage
        path = require_model_lineage(str(snapshot["path"]))
        if self.path == path:
            Progress.log("synth-model", "reusing cached actor weights", checkpoint=path, policy_step=self.policy_step)
            return False
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        started = time.monotonic()
        Progress.log("synth-model", "loading actor snapshot", checkpoint=path, device=self.device, policy_step=snapshot.get("policy_step"))
        if self.model is not None:
            del self.model
            self.model = None
            self.tokenizer = None
            gc.collect()
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
        with Progress.watch("synth-model", "still loading actor model", checkpoint=path, device=self.device):
            new_tokenizer = AutoTokenizer.from_pretrained(path)
            new_model = AutoModelForCausalLM.from_pretrained(
                path, torch_dtype="auto", low_cpu_mem_usage=True,
            ).to(self.device).eval()
        self.tokenizer, self.model = new_tokenizer, new_model
        self.path = path
        self.policy_step = int(snapshot["policy_step"])
        Progress.log("synth-model", "actor snapshot ready", elapsed_s=round(time.monotonic()-started, 1), policy_step=self.policy_step)
        return True

    def generate(self, snapshot, anchor, step, attempt, seed):
        self.current_step, self.current_attempt = step, attempt
        started = time.monotonic()
        Progress.log("synth", "starting sample", step=step, attempt=attempt, seed=str(seed)[:100])
        self.load_actor(snapshot)
        seed = prompt_content(seed)
        query = seed + "\nProduce a different problem, not a paraphrase."
        self.verification_history = []
        self._active_test_record = None
        self.rejected_problem_text = ""
        try:
            records = self.generator.generate_training_data(seed=query)
            if not records:
                raise ValueError("synth_gen returned zero generations")
        except Exception as error:
            Progress.log("synth", "sample generation FAILED", step=step, attempt=attempt, error=f"{type(error).__name__}: {error}", elapsed_s=round(time.monotonic()-started,1))
            # Preserve partial tests/codes from synth_gen even when it raises
            # before returning a VerifiedProgrammingGeneration object.
            previous_code = next((r["code"] for r in reversed(self.verification_history)
                                  if r.get("code")), "")
            previous_tests = next((r["test_code"] for r in reversed(self.verification_history)
                                   if r.get("test_code")), "")
            invalid_generated_test = "InvalidGeneratedTest" in str(error)
            previous_answer = next((r["raw_response"] for r in reversed(self.verification_history)
                                    if r.get("raw_response") and
                                    r.get("verifier") != "GeneratedPytestInputGuard"), "")
            return {
                "id": uuid4().hex,
                "time": datetime.now(timezone.utc).isoformat(),
                "source": "synth_gen", "track": os.environ.get("RL_TRACK", "ours"),
                "step": step, "attempt": attempt, "seed": seed,
                # Preserve the rejected raw problem for audit, NEVER use it
                # for the policy training prompt.
                "problem": self.rejected_problem_text or
                           "UNAVAILABLE: synth_gen terminated before yielding a problem",
                "code": previous_code,
                "test_code": previous_tests,
                "raw_output": previous_answer,
                "generation_prompt": synth_problem_instruction(query),
                "generator_model": self.path, "policy_snapshot_step": self.policy_step,
                "synth_gen_passed": False,
                "synth_verification_success": False,
                "synth_gen_failed_verifier": (
                    "GeneratedPytestInputGuard" if invalid_generated_test else "GenerationException"
                ),
                "synth_gen_error": f"{type(error).__name__}: {error}",
                "status": "invalid_test" if invalid_generated_test else "generation_error",
                "passed": False, "accepted": False,
                "failure_stage": "generated_test" if invalid_generated_test else "synth_gen_generation",
                "failure_reason": str(error),
                "error_message": str(error), "error_types": [type(error).__name__],
                "verification_history": list(self.verification_history),
                "generation_history": [],
                "raw_source": {
                    "original_seed": seed, "full_seed_prompt": query,
                    "reference_step": anchor.get("reference_step", ""),
                    "exception_traceback": traceback.format_exc(),
                },
            }
        # synth_gen may return intermediate failed revisions as separate records.
        # Keep every returned revision, select a successfully verified one if possible.
        item = next((r for r in records if getattr(r, "is_correct", False)), records[-1])
        generation_history = [{
            "revision": i, "code": str(getattr(r, "code", "") or ""),
            "raw_response": str(getattr(getattr(r, "response", None), "content", "") or ""),
            "test_code": str(getattr(r, "test_code", "") or ""),
            "is_correct": bool(getattr(r, "is_correct", False)),
            "verification_success": getattr(r, "verification_success", None),
            "failed_verifier": str(getattr(r, "failed_verifier_name", "") or ""),
            "failed_error": str(getattr(r, "failed_verifier_error_message", "") or ""),
        } for i, r in enumerate(records)]
        original_chat = getattr(item, "chat", None)
        messages = getattr(original_chat, "messages", []) or []
        source = {
            "original_seed": seed,
            "full_seed_prompt": query,
            "generation_chat": [{
                "role": getattr(m, "type", type(m).__name__),
                "content": prompt_content(m),
            } for m in messages],
            "synth_verifications": str(getattr(item, "verifications", None)),
            "generator_input_prompt": prompt_content(getattr(item, "prompt", "")),
            "reference_step": anchor.get("reference_step", ""),
        }
        failed_name = getattr(item, "failed_verifier_name", None)
        failed_error = getattr(item, "failed_verifier_error_message", None)
        prompt_messages = getattr(getattr(item, "prompt", None), "messages", []) or []
        problem = getattr(item, "problem", None) or (
            prompt_content(prompt_messages[-1])
            if prompt_messages else query
        )
        Progress.log("synth", "sample generated", step=step, attempt=attempt, verified=bool(getattr(item, "is_correct", False)), test_chars=len(str(getattr(item,"test_code","") or "")), elapsed_s=round(time.monotonic()-started,1))
        return {
            "id": uuid4().hex,
            "time": datetime.now(timezone.utc).isoformat(),
            "source": "synth_gen", "track": os.environ.get("RL_TRACK", "ours"),
            "step": step, "attempt": attempt, "seed": seed,
            "problem": problem,
            "code": getattr(item, "code", None) or "",
            "test_code": getattr(item, "test_code", None) or "",
            "raw_output": str(getattr(getattr(item, "response", None), "content", "")),
            "generation_prompt": synth_problem_instruction(query),
            "generator_model": self.path,
            "policy_snapshot_step": self.policy_step,
            "synth_gen_passed": bool(getattr(item, "is_correct", False)
                                          and getattr(item, "verification_success", True) is not False),
            "synth_verification_success": getattr(item, "verification_success", None),
            "verification_history": list(self.verification_history),
            "generation_history": generation_history,
            "synth_gen_failed_verifier": str(failed_name or ""),
            "synth_gen_error": str(failed_error or ""),
            "raw_source": source,
        }


_synth_worker = None


def _generate_in_worker(snapshot, anchor, step, attempt, seed):
    """Top-level spawn-safe entrypoint; worker model persists across tasks."""
    global _synth_worker
    if _synth_worker is None:
        _synth_worker = VerifiedCodeGenerator()
    return _synth_worker.generate(snapshot, anchor, step, attempt, seed)


def _redirect_worker_output(log_dir):
    """Keep each spawned worker's unfiltered logs, without interleaving them.

    Called by ProcessPoolExecutor ONLY for manual smoke testing. The actual
    training pipeline retains its normal console logging.
    """
    if not log_dir:
        return
    import sys
    from pathlib import Path
    folder = Path(log_dir)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"worker_{os.getpid()}.log"
    sys.stdout.flush()
    sys.stderr.flush()
    fd = os.open(str(target), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.dup2(fd, 1)
        os.dup2(fd, 2)
    finally:
        if fd not in (1, 2):
            os.close(fd)
    # One line per Progress.log, and all synth_gen print()s are persisted.
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass


class CodeGenerationPool:
    """Persistent, spawn-based generator pool for asynchronous GRPO production.

    RL_SYNTH_PROCESSES = number of CUDA model replicas, spread over visible GPUs.
    RL_SYNTH_BATCH_SIZE = max simultaneously scheduled generation attempts;
    the outer AsyncProducer handles per-seed retries and test/PCA filtering.
    """

    def __init__(self, worker_log_dir=None):
        count = env("RL_SYNTH_PROCESSES", int)
        if count < 1:
            raise ValueError("RL_SYNTH_PROCESSES must be >= 1")
        start = env("RL_SYNTH_START_METHOD")
        if start not in {"spawn", "forkserver"}:
            raise ValueError("Use spawn/forkserver to avoid CUDA fork hazards")
        gpu_count = _visible_gpu_count()
        # Validate GPU availability before launching any child processes.
        _, _, _ = _worker_cuda_plan(worker_ordinal=0, gpu_count=gpu_count, processes=count)
        allocation = {f"cuda:{i}": len(range(i, count, gpu_count)) for i in range(gpu_count)
                      if len(range(i, count, gpu_count)) > 0}
        Progress.log("synth-pool", "starting CUDA worker pool", processes=count,
                     visible_gpus=gpu_count, assignments=allocation,
                     vram_total_fraction=env("RL_SYNTH_GPU_MEMORY_FRACTION", float),
                     in_flight=env("RL_SYNTH_BATCH_SIZE", int), start_method=start)
        pool_kwargs = {"max_workers": count, "mp_context": mp.get_context(start)}
        if worker_log_dir is not None:
            pool_kwargs.update(initializer=_redirect_worker_output,
                               initargs=(worker_log_dir,))
        self.pool = ProcessPoolExecutor(**pool_kwargs)
        self.closed = False

    def submit(self, snapshot, anchor, step, attempt, seed):
        if self.closed:
            raise RuntimeError("CodeGenerationPool already closed")
        return self.pool.submit(_generate_in_worker, snapshot, anchor, step, attempt, seed)

    def close(self):
        if not self.closed:
            self.closed = True
            Progress.log("synth-pool", "shutting down pool")
            self.pool.shutdown(wait=True, cancel_futures=True)
            Progress.log("synth-pool", "pool stopped")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def main():
    """Backward-compatible CLI: delegate to the ONE sample test implementation."""
    from custome.sample_test import main as sample_main
    sample_main()


if __name__ == "__main__":
    main()
