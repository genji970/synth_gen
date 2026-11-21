
import os
import re
from datetime import datetime, timezone
from uuid import uuid4

from custome.rl import hub, sandbox
from custome.rl.features import distances
from custome.rl.visualize import plot


def code_only(text):
    blocks = re.findall(
        r"```(?:python)?\s*\n(.*?)```",
        text,
        flags=re.S | re.I,
    )
    return blocks[0].strip() if blocks else text.strip()


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    **kwargs,
):
    info = extra_info or {}
    code = code_only(solution_str)

    result = sandbox.verify(
        code,
        test_code=info.get("test_code"),
        apps_io=info.get("apps_io"),
    )

    raw_refs = info.get("reference_batch")
    refs = list(raw_refs) if raw_refs is not None else []

    rewards = {"d1": 0., "d2": 0., "d3": 0.}
    coordinates = None

    if len(refs) >= 2:
        try:
            dr, ref_pts, new_pts = distances(tuple(refs), code)
            rewards.update(dr)
            coordinates = (ref_pts, new_pts)
        except Exception as exc:
            print("[feature reward unavailable]", repr(exc))

    rewards["exec"] = float(result["passed"])
    rewards["err"] = 1.0 / (
        1.0 + len(result["error_types"])
    )

    weights = {
        key: float(os.environ["RL_WEIGHT_" + key.upper()])
        for key in rewards
    }

    if (
        any(v < 0 for v in weights.values())
        or sum(weights.values()) <= 0
    ):
        raise ValueError("Invalid reward weights")

    total = sum(
        rewards[k] * weights[k] for k in rewards
    ) / sum(weights.values())

    sample_id = uuid4().hex[:12]

    record = {
        "id": sample_id,
        "time": datetime.now(timezone.utc).isoformat(),
        "source": "grpo",
        "seed": info.get("seed"),
        "backbone": os.environ.get(
            "RL_TRAIN_MODEL_PATH",
            os.environ["RL_MODEL"],
        ),
        "feature_backbone": os.environ["RL_FEATURE_MODEL"],
        "benchmark": info.get("benchmark"),
        "problem": info.get("problem"),
        "raw_output": solution_str,
        "code": code,
        "test_code": result.get("test_code"),
        "apps_io": info.get("apps_io"),
        "passed": result["passed"],
        "status": result["status"],
        "error_types": result["error_types"],
        "error_message": result["error_message"],
        "pytest_summary": result["summary"],
        "pytest_exit_code": result.get("pytest_exit_code"),
        "coverage": result.get("coverage"),
        "reward_components": rewards,
        "reward_weights": weights,
        "reward_total": total,
        "batch_id": info.get("batch_id"),
    }

    if info.get("split") == "train":
        hub.append([record])

        interval = int(os.environ["RL_VIS_EVERY"])
        if (
            coordinates
            and interval > 0
            and info.get("batch_id", 0) % interval == 0
        ):
            plot(
                info["batch_id"],
                sample_id,
                *coordinates,
                rewards,
            )

    return {
        "score": total,
        "acc": rewards["exec"],
        **{
            f"reward_{key}": value
            for key, value in rewards.items()
        },
    }
