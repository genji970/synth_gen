
import os
from datetime import datetime, timezone
from uuid import uuid4


def search(seed):
    if os.environ.get("RL_WEB_SEARCH") != "1":
        return ""
    try:
        from ddgs import DDGS
        hits = list(DDGS().text(
            seed,
            max_results=int(os.environ["RL_WEB_RESULTS"]),
        ))
        return "\n".join(
            (x.get("title", "") + " " + x.get("body", ""))[:500]
            for x in hits
        )
    except Exception as exc:
        print("[web lookup unavailable]", exc)
        return ""


def main():
    os.environ.pop("LLM", None)

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from synth_gen.generators import PythonVerifiedSynthDataGenerator
    from synth_gen.generation.parsing import (
        extract_triple_backticked_string,
    )
    from synth_gen.verification.python_verifiers import (
        PythonTestBasedVerifier,
    )
    from custome.rl import hub, sandbox

    name = os.environ["RL_MODEL"]
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()

    class ChosenChat(BaseChatModel):
        @property
        def _llm_type(self):
            return "chosen_hf_llm"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            chat = [
                {
                    "role": (
                        "assistant" if m.type == "ai"
                        else "system" if m.type == "system"
                        else "user"
                    ),
                    "content": str(m.content),
                }
                for m in messages
            ]
            prompt = tok.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tok(prompt, return_tensors="pt").to(model.device)

            with torch.inference_mode():
                seq = model.generate(
                    **inputs,
                    max_new_tokens=int(os.environ["RL_MAX_NEW_TOKENS"]),
                    do_sample=True,
                    temperature=float(os.environ["RL_GEN_TEMPERATURE"]),
                    top_p=float(os.environ["RL_GEN_TOP_P"]),
                )

            reply = tok.decode(
                seq[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            return ChatResult(generations=[
                ChatGeneration(message=AIMessage(content=reply))
            ])

    chat = ChosenChat()

    class Generator(PythonVerifiedSynthDataGenerator):
        def _generate_prompt(self, seed=None):
            prompt = (
                "Create a challenging self-contained Python function "
                "programming problem inspired by this seed. "
                "No solution; put the problem in triple backticks.\n"
                + str(seed)
            )
            result = self.call_llm(prompt)
            return extract_triple_backticked_string(result) or result

    gen = Generator()
    gen.set_llm(chat)

    for verifier in gen.verifiers:
        if isinstance(verifier, PythonTestBasedVerifier):
            original = verifier.check
            verifier.check = (
                lambda generation, fn=original:
                fn(generation, llm=chat)
            )

    for seed in os.environ["RL_CODE_SEEDS"].split("|"):
        for _ in range(
            int(os.environ["RL_SYNTHGEN_SAMPLES_PER_SEED"])
        ):
            enrichment = search(seed)

            try:
                full_seed = (
                    seed + "\n" + enrichment if enrichment else seed
                )
                items = gen.generate_training_data(seed=full_seed)

                for item in items:
                    code = item.code or (
                        item.response.content if item.response else ""
                    )
                    tests = getattr(item, "test_code", None)
                    report = sandbox.verify(code, test_code=tests)

                    rec = {
                        "id": uuid4().hex[:12],
                        "time": datetime.now(timezone.utc).isoformat(),
                        "source": "synth_gen",
                        "seed": seed,
                        "backbone": name,
                        "search_context": enrichment,
                        "problem": (
                            item.prompt.messages[-1].content
                            if item.prompt else ""
                        ),
                        "raw_output": (
                            item.response.content if item.response else ""
                        ),
                        "code": code,
                        "test_code": tests,
                        "passed": report["passed"],
                        "status": report["status"],
                        "error_message": report["error_message"],
                        "error_types": report["error_types"],
                        "pytest_summary": report["summary"],
                        "pytest_exit_code": report.get("pytest_exit_code"),
                        "coverage": report.get("coverage"),
                        "synth_gen_passed": item.is_correct,
                        "synth_gen_failed_verifier": getattr(
                            item, "failed_verifier_name", None
                        ),
                        "synth_gen_error": getattr(
                            item, "failed_verifier_error_message", None
                        ),
                        "reward_components": None,
                        "reward_total": None,
                    }
                    hub.append([rec])
                    print("saved:", seed, rec["passed"])

            except Exception as exc:
                hub.append([{
                    "id": uuid4().hex[:12],
                    "time": datetime.now(timezone.utc).isoformat(),
                    "source": "synth_gen",
                    "seed": seed,
                    "backbone": name,
                    "problem": None,
                    "code": None,
                    "test_code": None,
                    "passed": False,
                    "status": "generation_error",
                    "error_message": str(exc),
                    "error_types": [type(exc).__name__],
                }])
                print("generation failure:", repr(exc))


if __name__ == "__main__":
    main()
