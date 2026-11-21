
import json
import os
from pathlib import Path
from datasets import Dataset, concatenate_datasets, load_dataset

MATH_SUBSETS = (
    "algebra", "counting_and_probability", "geometry",
    "intermediate_algebra", "number_theory",
    "prealgebra", "precalculus",
)


def parse_json(s, default):
    try:
        return json.loads(s) if isinstance(s, str) else (s or default)
    except (ValueError, TypeError):
        return default


def build():
    root = Path(os.environ["RL_DATA"])
    root.mkdir(parents=True, exist_ok=True)

    math = {
        split: concatenate_datasets([
            load_dataset("EleutherAI/hendrycks_math", name, split=split)
            for name in MATH_SUBSETS
        ])
        for split in ("train", "test")
    }
    social = {
        split: load_dataset(
            "baber/social_i_qa",
            split="validation" if split == "test" else "train",
        )
        for split in ("train", "test")
    }
    code = {
        split: load_dataset("codeparrot/apps", split=split)
        for split in ("train", "test")
    }

    train, ev = {}, {}

    for split, dest in (("train", train), ("test", ev)):
        dest["math"] = [
            dict(task="math", prompt=r["problem"],
                 answer=r["solution"], apps_io="")
            for r in math[split]
        ]
        dest["social"] = [
            dict(
                task="social",
                prompt=(
                    f"{r['context']}\n{r['question']}\n"
                    f"A. {r['answerA']}\n"
                    f"B. {r['answerB']}\n"
                    f"C. {r['answerC']}\n"
                    "Reply with one letter."
                ),
                answer="ABC"[int(r["label"]) - 1],
                apps_io="",
            )
            for r in social[split]
        ]
        dest["code"] = [
            dict(
                task="code",
                prompt=r["question"],
                answer=next(iter(parse_json(r["solutions"], [])), ""),
                apps_io=r["input_output"] or "",
            )
            for r in code[split]
        ]
        if split == "train":
            dest["code"] = [r for r in dest["code"] if r["answer"]]

    synthetic = Path(os.environ["RL_OUT"]) / "generated_code.jsonl"

    if synthetic.exists():
        for line in synthetic.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("passed") and r.get("source") == "synth_gen":
                train["code"].append({
                    "task": "code",
                    "prompt": r["problem"],
                    "answer": r["code"],
                    "apps_io": "",
                    "test_code": r.get("test_code", ""),
                    "seed": r.get("seed", ""),
                })

    for task in ("math", "social", "code"):
        sft = [
            {
                "messages": [
                    {"role": "user", "content": r["prompt"]},
                    {"role": "assistant", "content": r["answer"]},
                ]
            }
            for r in train[task]
        ]
        Dataset.from_list(sft).to_parquet(
            str(root / f"{task}_train.parquet")
        )

        valid = [
            {
                "messages": [
                    {"role": "user", "content": r["prompt"]},
                    {"role": "assistant", "content": r["answer"]},
                ]
            }
            for r in ev[task] if r["answer"]
        ]
        Dataset.from_list(valid).to_parquet(
            str(root / f"{task}_eval.parquet")
        )
        Dataset.from_list(ev[task]).to_parquet(
            str(root / f"{task}_eval_raw.parquet")
        )
        print(task, len(train[task]), len(ev[task]))

    rl = []
    bs = int(os.environ["RL_GRPO_BATCH"])

    for i in range(0, len(train["code"]), bs):
        block = train["code"][i:i + bs]
        if len(block) < bs:
            break

        references = [x["answer"] for x in block]

        for r in block:
            rl.append({
                "data_source": "code",
                "prompt": [{
                    "role": "user",
                    "content": r["prompt"] + "\nOutput Python code only.",
                }],
                "reward_model": {"ground_truth": ""},
                "extra_info": {
                    "seed": r.get("seed", ""),
                    "problem": r["prompt"],
                    "benchmark": "APPS" if r["apps_io"] else "synthetic",
                    "reference_batch": references,
                    "test_code": r.get("test_code", ""),
                    "apps_io": r["apps_io"],
                    "batch_id": i // bs,
                    "split": "train",
                },
            })

    Dataset.from_list(rl).to_parquet(
        str(root / "code_rl_train.parquet")
    )

    val = []
    for i, r in enumerate(ev["code"]):
        val.append({
            "data_source": "code",
            "prompt": [{
                "role": "user",
                "content": r["prompt"] + "\nOutput Python code only.",
            }],
            "reward_model": {"ground_truth": ""},
            "extra_info": {
                "seed": "",
                "problem": r["prompt"],
                "benchmark": "APPS",
                "reference_batch": [],
                "test_code": "",
                "apps_io": r["apps_io"],
                "batch_id": i,
                "split": "eval",
            },
        })

    Dataset.from_list(val).to_parquet(
        str(root / "code_rl_eval.parquet")
    )


if __name__ == "__main__":
    build()
