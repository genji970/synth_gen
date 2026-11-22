"""One benchmark registry for preparation, evaluation, CL metrics and comparisons."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

from custome.rl.core import Progress, env, files
from custome.rl.data_load import DataLoad, boxed_answer



class ContinualMetrics:
    """Track-separated CL matrix. Common pre-RL stages are included in each track."""

    @staticmethod
    def compute(history, current, stage):
        scores = current["accuracy"]
        if not history:
            return dict(average_accuracy=float(np.mean(list(scores.values()))),
                        average_forgetting=0., bwt=0., fwt=0., plasticity_gain=0.)
        # The training task order is configurable; never assume 3 fixed stages.
        sequence = env("RL_SFT_TASKS").split()
        if stage.startswith("grpo"):
            learned = len(sequence)
        elif stage in sequence:
            learned = sequence.index(stage) + 1
        else:
            learned = 0
        forgetting, bwt, fwt = [], [], []
        for j, task in enumerate(sequence[:learned]):
            if task not in scores:
                continue  # Task was excluded from RL_EVAL_TASKS.
            previous = [r["accuracy"][task] for r in history if task in r["accuracy"]]
            if previous:
                forgetting.append(max(0., max(previous) - scores[task]))
            start = next((r["accuracy"].get(task) for r in history if r["stage"] == task), None)
            if start is not None and j < learned - 1:
                bwt.append(scores[task] - start)
        base = next((r["accuracy"] for r in history if r["stage"] == "base"), history[0]["accuracy"])
        for index, task in enumerate(sequence[1:], 1):
            if task not in scores or task not in base:
                continue
            prior = next((r["accuracy"].get(task) for r in history
                          if r["stage"] == sequence[index - 1]), None)
            if prior is not None:
                fwt.append(prior - base[task])
        task = stage if stage in scores else env("RL_GRPO_TASK")
        if task not in scores:
            task = next(iter(scores))
        prev_value = history[-1]["accuracy"].get(task, scores[task])
        return dict(average_accuracy=float(np.mean(list(scores.values()))),
                    average_forgetting=float(np.mean(forgetting or [0])),
                    bwt=float(np.mean(bwt or [0])), fwt=float(np.mean(fwt or [0])),
                    plasticity_gain=scores[task] - prev_value)



class MetricPlots:
    @staticmethod
    def scatter(series, names, folder, title):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(9, 4))
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for i, name in enumerate(names):
            x = [j for j, row in enumerate(series) if name in row]
            y = [series[j][name] for j in x]
            ax.scatter(x, y, c=colors[i % len(colors)], marker="o", label=name)
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(folder / f"{title}.png", dpi=125)
        plt.close(fig)

    def history(self, rows):
        for field in ("accuracy", "continual"):
            self.scatter([r[field] for r in rows],
                         sorted({k for r in rows for k in r[field]}),
                         files.output / "visualizations", field)

    def compare(self, rows):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        tasks = env("RL_EVAL_TASKS").split()
        for i, row in enumerate(rows):
            ax.scatter(range(len(tasks)), [row[f"{k}_accuracy"] for k in tasks],
                       marker="o", label=row["track"])
        ax.set_xticks(range(len(tasks)), tasks)
        ax.legend(); fig.tight_layout()
        fig.savefig(files.output / "tracks_comparison.png", dpi=125)
        plt.close(fig)


class Evaluator:
    boxed = staticmethod(boxed_answer)

    @staticmethod
    def social_answer(text):
        hits = re.findall(r"(?<![A-Za-z])[ABC](?![A-Za-z])", text.upper())
        return hits[-1] if hits else "?"

    def evaluate(self, model_path, stage):
        from datasets import load_dataset
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from custome.rl.score import CodeVerifier, code_only
        Progress.log("eval", "loading evaluator vLLM model", stage=stage, model=model_path)
        with Progress.watch("eval", "still loading vLLM model", stage=stage):
            model = LLM(model=model_path, tensor_parallel_size=env("RL_EVAL_TP", int),
                        gpu_memory_utilization=env("RL_ROLLOUT_MEM", float),
                        max_model_len=env("RL_PROMPT_LEN", int) + env("RL_RESPONSE_LEN", int))
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        sampler = SamplingParams(temperature=0., max_tokens=env("RL_RESPONSE_LEN", int))
        verifier = CodeVerifier()
        Progress.log("eval", "model ready; starting tasks", stage=stage, tasks=env("RL_EVAL_TASKS"))
        scores, details = {}, {}
        for task in env("RL_EVAL_TASKS").split():
            if task not in DataLoad.names:
                raise ValueError(f"Unknown evaluation task in RL_EVAL_TASKS: {task}")
            records = DataLoad.read(task, "eval")
            limit = env("RL_EVAL_LIMIT", int)
            if limit:
                records = records[:limit]
            Progress.log("eval", "starting task inference", stage=stage, task=task,
                         examples=len(records))
            prompts = [tokenizer.apply_chat_template(
                [{"role": "user", "content": row.prompt}],
                tokenize=False, add_generation_prompt=True,
            ) for row in records]
            Progress.log("eval", "running vLLM generation", task=task,
                         examples=len(prompts))
            with Progress.watch("eval", "still generating evaluation answers", stage=stage, task=task, examples=len(prompts)):
                output = model.generate(prompts, sampler)
            Progress.log("eval", "vLLM generation finished; checking answers", task=task,
                         examples=len(prompts))
            preds = [r.outputs[0].text for r in output]
            correct, gold, predicted = [], [], []
            for index, (row, text) in enumerate(zip(records, preds), 1):
                if index == 1 or index % max(1, env("RL_PROGRESS_EVAL_EVERY", int)) == 0:
                    Progress.log("eval", "verifying predictions", stage=stage, task=task,
                                 checked=index, total=len(records))
                if task == "math":
                    success = self.boxed(text) == row.answer
                elif task == "social":
                    answer = self.social_answer(text)
                    gold.append(row.answer)
                    predicted.append(answer)
                    success = answer == row.answer
                else:
                    success = verifier.verify(code_only(text), apps_io=row.apps_io)["passed"]
                correct.append(int(success))
            scores[task] = float(np.mean(correct)) if correct else 0.
            detail = {"correct": sum(correct), "count": len(correct)}
            if task == "social":
                from sklearn.metrics import precision_recall_fscore_support
                precision, recall, f1, _ = precision_recall_fscore_support(
                    gold, predicted, labels=list("ABC"), average="macro", zero_division=0)
                detail.update(precision=float(precision), recall=float(recall), macro_f1=float(f1))
            if task == "code": detail["pass@1"] = scores[task]
            if task == "math": detail["exact_match"] = scores[task]
            details[task] = detail
            print("[eval]", stage, task, detail, flush=True)
        del model
        history_file = files.output / "history.jsonl"
        all_rows = [json.loads(line) for line in history_file.read_text().splitlines()] if history_file.exists() else []
        if stage.startswith("grpo_"):
            track_stage = stage.split("_step_")[0]
            relevant = [r for r in all_rows if r["stage"] in ("base", "math", "social", "code")
                        or r["stage"] == track_stage or r["stage"].startswith(track_stage + "_step_")]
        else:
            relevant = [r for r in all_rows if not r["stage"].startswith("grpo_")]
        row = {"stage": stage, "model": model_path, "accuracy": scores, "per_task": details,
               "continual": ContinualMetrics.compute(relevant, {"accuracy": scores}, stage),
               "eval_limit": env("RL_EVAL_LIMIT", int)}
        files.append(history_file, [row])
        MetricPlots().history(all_rows + [row])
        return row


class Comparator:
    def compare(self):
        hist = files.output / "history.jsonl"
        records = [json.loads(x) for x in hist.read_text().splitlines()]
        selected = [r for r in records if r["stage"].startswith("grpo_") and "_step_" not in r["stage"]]
        if not selected:
            raise ValueError("No completed GRPO tracks available")
        rows = []
        for record in selected:
            row = {"track": record["stage"].removeprefix("grpo_")}
            row.update({f"{task}_accuracy": record["accuracy"][task]
                        for task in env("RL_EVAL_TASKS").split()})
            row.update(record["continual"])
            rows.append(row)
        with (files.output / "tracks_comparison.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        MetricPlots().compare(rows)
        print(json.dumps(rows, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "eval", "compare"])
    parser.add_argument("--model")
    parser.add_argument("--stage")
    args = parser.parse_args()
    if args.action == "prepare": DataLoad().prepare()
    elif args.action == "compare": Comparator().compare()
    else: Evaluator().evaluate(args.model, args.stage)


if __name__ == "__main__":
    main()
