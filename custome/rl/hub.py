
import fcntl
import json
import os

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def append(records):
    root = Path(os.environ["RL_OUT"])
    root.mkdir(parents=True, exist_ok=True)

    dst = root / "generated_code.jsonl"
    state_file = root / "hub_state.json"

    with (root / ".upload.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)

        with dst.open("a", encoding="utf-8") as f:
            for record in records:
                f.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        default=str,
                    ) + "\n"
                )
            f.flush()
            os.fsync(f.fileno())

        try:
            upload(dst, state_file)
        except Exception as exc:
            # Failed uploads remain saved locally.
            print(
                f"[HF upload pending] {type(exc).__name__}: {exc}",
                flush=True,
            )
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def upload(dst, state_file, force=False):
    from huggingface_hub import HfApi

    threshold = int(os.environ["RL_UPLOAD_EVERY"])
    state = (
        json.loads(state_file.read_text())
        if state_file.exists() else {}
    )

    count = sum(
        1 for _ in dst.open(encoding="utf-8")
    )
    start = state.get("uploaded", 0)

    if count - start < threshold and not force:
        return
    if count == start:
        return

    api = HfApi(
        token=os.environ.get("RL_HF_TOKEN") or None
    )
    owner = (
        os.environ.get("RL_HF_OWNER")
        or api.whoami()["name"]
    )

    if not state.get("repo"):
        suffix = (
            datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            + "-"
            + uuid4().hex[:8]
        )
        repo = (
            f"{owner}/"
            f"{os.environ['RL_HF_REPO_PREFIX']}-{suffix}"
        )

        api.create_repo(
            repo_id=repo,
            repo_type="dataset",
            private=True,
            exist_ok=False,
        )

        state["repo"] = repo
        state_file.write_text(json.dumps(state))

    repo = state["repo"]

    info = api.repo_info(
        repo_id=repo,
        repo_type="dataset",
    )

    if not info.private:
        raise RuntimeError(
            f"Refusing upload: repository is NOT private: {repo}"
        )

    lines = dst.read_text(
        encoding="utf-8"
    ).splitlines(keepends=True)

    chunk_no = state.get("chunks", 0)
    chunk = dst.parent / f"upload_chunk_{chunk_no:06d}.jsonl"

    chunk.write_text(
        "".join(lines[start:]),
        encoding="utf-8",
    )

    api.upload_file(
        path_or_fileobj=str(chunk),
        path_in_repo=f"data/part-{chunk_no:06d}.jsonl",
        repo_id=repo,
        repo_type="dataset",
        commit_message=f"Add samples {start}:{count}",
    )

    state.update(
        uploaded=count,
        chunks=chunk_no + 1,
    )

    tmp = state_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(state))
    tmp.replace(state_file)
    chunk.unlink(missing_ok=True)

    print(
        f"[HF private] {repo}: uploaded {count} records",
        flush=True,
    )


if __name__ == "__main__":
    root = Path(os.environ["RL_OUT"])
    dst = root / "generated_code.jsonl"

    if dst.exists():
        with (root / ".upload.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            upload(
                dst,
                root / "hub_state.json",
                force=True,
            )
