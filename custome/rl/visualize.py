
import os
from pathlib import Path


def plot(batch_id, sample_id, ref, new, rewards):
    if os.environ.get("RL_VISUALIZE") != "1":
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    dest = Path(os.environ["RL_OUT"]) / "visualizations"
    dest.mkdir(parents=True, exist_ok=True)

    for n in (1, 2, 3):
        base = np.asarray(ref[n], float).reshape(-1, n)
        point = np.asarray(new[n], float).reshape(1, n)

        fig = plt.figure(figsize=(5, 4))

        if n == 3:
            ax = fig.add_subplot(111, projection="3d")

            ax.scatter(
                base[:, 0], base[:, 1], base[:, 2],
                c="#356cb4", marker="o", label="Reference",
            )
            ax.scatter(
                point[:, 0], point[:, 1], point[:, 2],
                c="#e88a38", marker="o", label="New",
            )
        else:
            ax = fig.add_subplot(111)

            ax.scatter(
                base[:, 0],
                base[:, 1] if n == 2 else np.zeros(len(base)),
                c="#356cb4", marker="o", label="Reference",
            )
            ax.scatter(
                point[:, 0],
                point[:, 1] if n == 2 else [0],
                c="#e88a38", marker="o", label="New",
            )
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2" if n == 2 else "")

        ax.set_title(
            f"Batch {batch_id} · {n}D · R={rewards[f'd{n}']:.3f}"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            dest / f"{batch_id}_{sample_id}_{n}d.png",
            dpi=130,
        )
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3))

    for i, (key, value) in enumerate(rewards.items()):
        color = (
            "#356cb4" if key.startswith("d")
            else "#e88a38"
        )
        ax.scatter(i, value, c=color, marker="o")

    ax.set_xticks(
        range(len(rewards)),
        list(rewards),
        rotation=30,
    )
    ax.set_ylim(-0.03, 1.03)
    fig.tight_layout()
    fig.savefig(
        dest / f"{batch_id}_{sample_id}_rewards.png",
        dpi=130,
    )
    plt.close(fig)


def history(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dest = Path(os.environ["RL_OUT"]) / "visualizations"
    dest.mkdir(parents=True, exist_ok=True)

    colors = [
        "#356cb4", "#e88a38",
        "#3a9e7a", "#984caf", "#c24f5b",
    ]

    for field, name in (
        ("accuracy", "task_accuracy"),
        ("continual", "continual_metrics"),
    ):
        fig, ax = plt.subplots(figsize=(8, 4))

        labels = sorted({
            key
            for row in rows
            for key in row[field]
        })

        for i, key in enumerate(labels):
            xs = [
                j for j, row in enumerate(rows)
                if key in row[field]
            ]
            ys = [rows[j][field][key] for j in xs]

            ax.scatter(
                xs, ys,
                color=colors[i % len(colors)],
                marker="o",
                label=key,
            )

        ax.set_xticks(
            range(len(rows)),
            [x["stage"] for x in rows],
            rotation=30,
        )
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(dest / f"{name}.png", dpi=130)
        plt.close(fig)
