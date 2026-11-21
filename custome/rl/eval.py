
import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support


def boxed(s):
    k = s.rfind("\\boxed{")
    if k < 0:
        return s.strip().splitlines()[-1].strip()

    start = k + 7
    depth = 1

    for i in range(start, len(s)):
        depth += (s[i] == "{") - (s[i] == "}")
        if depth == 0:
            return (
                s[start:i]
                .replace(" ", "")
                .replace("\\left", "")
                .replace("\\right", "")
            )
    return s.strip()


def label(s):
    matches = re.findall(
        r"(?<![A-Za-z])[ABC](?![A-Za-z])",
        s.upper(),
    )
    return matches[-1] if matches else "?"


def continual(history, now, stage):
    tasks = ["math", "social", "code"]
    scores = now["accuracy"]
    old = [x["accuracy"] for x in history]
    avg = float(np.mean(list(scores.values())))

    if not old:
        return {
            "average_accuracy": avg,
            "average_forgetting": 0.,
            "bwt": 0.,
            "fwt": 0.,
            "plasticity_gain": 0.,
        }

    learned = {
        "math": 0, "social": 1,
        "code": 2, "grpo": 2,
    }.get(stage, 2)

    forgetting, bwt, fwt = [], [], []

    for j, task in enumerate(tasks[:learned + 1]):
        past = [row[task] for row in old if task in row]
        if past:
            forgetting.append(
                max(0., max(past) - scores[task])
            )

        diagonal = next(
            (r["accuracy"][task] for r in history
             if r["stage"] == task),
            None,
        )
        if diagonal is not None and j < learned:
            bwt.append(scores[task] - diagonal)

    base = history[0]["accuracy"]

    for j, task in enumerate(tasks[1:], 1):
        prior = next(
            (r["accuracy"][task] for r in history
             if r["stage"] == tasks[j - 1]),
            None,
        )
        if prior is not None:
            fwt.append(prior - base[task])

    current = stage if stage in scores else "code"
    gain = scores[current] - old[-1][current]

    return {
        "average_accuracy": avg,
        "average_forgetting": float(np.mean(forgetting or [0])),
        "bwt": float(np.mean(bwt or [0])),
        "fwt": float(np.mean(fwt or [0])),
        "plasticity_gain": gain,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--stage", required=True)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from custome.rl.sandbox import verify
    from custome.rl.reward import code_only

    model = LLM(
        model=args.model,
        tensor_parallel_size=int(os.environ["RL_ROLLOUT_TP"]),
        gpu_memory_utilization=float(
            os.environ["RL_ROLLOUT_MEM"]
        ),
        max_model_len=(
            int(os.environ["RL_PROMPT_LEN"])
            + int(os.environ["RL_RESPONSE_LEN"])
        ),
    )
    tok = AutoTokenizer.from_pretrained(args.model)
    sampling = SamplingParams(
        temperature=0.,
        max_tokens=int(os.environ["RL_RESPONSE_LEN"]),
    )

    accuracy, detailed = {}, {}

    for task in ("math", "social", "code"):
        path = (
            Path(os.environ["RL_DATA"])
            / f"{task}_eval_raw.parquet"
        )

        ds = load_dataset(
            "parquet",
            data_files=str(path),
            split="train",
        )

        limit = int(os.environ["RL_EVAL_LIMIT"])
        if limit:
            ds = ds.select(range(min(limit, len(ds))))

        prompts = [
            tok.apply_chat_template(
                [{"role": "user", "content": r["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for r in ds
        ]

        outputs = model.generate(prompts, sampling)
        preds = [x.outputs[0].text for x in outputs]

        scores, labels, guesses = [], [], []

        for row, pred in zip(ds, preds):
            if task == "math":
                ok = boxed(pred) == boxed(row["answer"])

            elif task == "social":
                gold = row["answer"]
                guess = label(pred)
                labels.append(gold)
                guesses.append(guess)
                ok = gold == guess

            else:
                result = verify(
                    code_only(pred),
                    apps_io=row["apps_io"],
                )
                ok = result["passed"]

            scores.append(int(ok))

        accuracy[task] = (
            float(np.mean(scores)) if scores else 0.
        )
        detail = {
            "correct": sum(scores),
            "count": len(scores),
        }

        if task == "social":
            precision, recall, f1, _ = (
                precision_recall_fscore_support(
                    labels, guesses,
                    labels=list("ABC"),
                    average="macro",
                    zero_division=0,
                )
            )
            detail.update({
                "precision": float(precision),
                "recall": float(recall),
                "macro_f1": float(f1),
            })

        if task == "code":
            detail["pass@1"] = accuracy[task]
        if task == "math":
            detail["exact_match"] = accuracy[task]

        detailed[task] = detail
        print(task, detail, flush=True)

    del model

    histfile = Path(os.environ["RL_OUT"]) / "history.jsonl"
    hist = [
        json.loads(line)
        for line in histfile.read_text().splitlines()
    ] if histfile.exists() else []

    cl = continual(hist, {"accuracy": accuracy}, args.stage)

    record = {
        "stage": args.stage,
        "model": args.model,
        "accuracy": accuracy,
        "per_task": detailed,
        "continual": cl,
        "eval_limit": int(os.environ["RL_EVAL_LIMIT"]),
    }

    with histfile.open("a") as f:
        f.write(json.dumps(record) + "\n")

    from custome.rl.visualize import history as plot_history
    plot_history(hist + [record])

    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
