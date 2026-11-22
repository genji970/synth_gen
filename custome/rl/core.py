"""Shared durable files, private Hugging Face storage, and checkpoint monitoring.
No per-track implementations of these mechanics.
"""
from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import threading
from contextlib import contextmanager
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def env(name: str, cast=str, default=None):
    value = os.environ.get(name, default)
    if value is None:
        raise KeyError(f"Missing configuration: {name}")
    return cast(value)


class Progress:
    """One shared, unbuffered progress/heartbeat logger for all pipeline stages.

    All thresholds and verbosity are configured in custome/config.sh. A caller
    passes an accurate known total if available; this logger never guesses ETA.
    """

    _last = {}
    _started = time.monotonic()

    @classmethod
    def log(cls, scope, event, **fields):
        if os.environ.get("RL_PROGRESS_ENABLED", "1") == "0":
            return
        stamp = datetime.now().astimezone().strftime("%H:%M:%S")
        suffix = " ".join(f"{key}={str(value).replace(chr(10), ' ')[:220]}"
                          for key, value in fields.items() if value is not None)
        print(f"[{stamp}][{scope}][pid={os.getpid()}] {event}"
              + (f" | {suffix}" if suffix else ""), flush=True)

    @classmethod
    def interval(cls):
        return max(1.0, env("RL_PROGRESS_INTERVAL_SECS", float, "15"))

    @classmethod
    def pulse(cls, scope, key, event, *, force=False, **fields):
        now = time.monotonic()
        marker = (scope, key)
        if force or now - cls._last.get(marker, -float("inf")) >= cls.interval():
            cls._last[marker] = now
            cls.log(scope, event, **fields)

    @classmethod
    @contextmanager
    def watch(cls, scope, event, **fields):
        """Emit periodic status even if an API/model-load call blocks silently."""
        if os.environ.get("RL_PROGRESS_ENABLED", "1") == "0":
            yield
            return
        stop = threading.Event()
        begin = time.monotonic()
        def heartbeat():
            while not stop.wait(cls.interval()):
                cls.log(scope, event, elapsed_s=round(time.monotonic()-begin,1), **fields)
        thread = threading.Thread(target=heartbeat, name=f"heartbeat-{scope}", daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=1)

    @classmethod
    def wait_process(cls, child, scope, *, label=None, started=None):
        """Keep terminal live during long trainer/merger subprocesses."""
        begin = time.monotonic() if started is None else started
        while child.poll() is None:
            try:
                child.wait(timeout=cls.interval())
            except __import__("subprocess").TimeoutExpired:
                cls.log(scope, "still running", task=label, elapsed_s=round(time.monotonic() - begin, 1))
        if child.returncode:
            cls.log(scope, "FAILED", task=label, exit_code=child.returncode,
                    elapsed_s=round(time.monotonic() - begin, 1))
            raise __import__("subprocess").CalledProcessError(child.returncode, child.args)
        cls.log(scope, "completed", task=label, elapsed_s=round(time.monotonic() - begin, 1))


def grpo_steps(track: str) -> int:
    """Single step-resolution rule for VERL trainer, producer and stream loader."""
    name = str(track).upper()
    steps = env(f"RL_GRPO_STEPS_{name}", int)
    if steps <= 0:
        raise ValueError(f"RL_GRPO_STEPS_{name} must be positive")
    return steps


def require_model_lineage(path: str) -> str:
    """Admit only the config.sh base or its locally derived checkpoints.

    A fine-tuned stage may live under RL_CKPT; arbitrary second HF model IDs
    or external directories are deliberately rejected.
    """
    name = str(path)
    base = env("RL_BASE_MODEL")
    if name == base:
        return name
    check_root = Path(env("RL_CKPT")).expanduser().resolve()
    candidate = Path(name).expanduser().resolve()
    if candidate == check_root or check_root in candidate.parents:
        return name
    raise ValueError(
        f"Model override is not from RL_BASE_MODEL={base!r} or its local "
        f"RL_CKPT={str(check_root)!r} lineage: {name!r}"
    )


class Files:
    @property
    def output(self):
        return Path(env("RL_OUT"))

    @property
    def checkpoint(self):
        return Path(env("RL_CKPT"))

    @property
    def data(self):
        return Path(env("RL_DATA"))

    def track(self, name=None):
        return self.output / "tracks" / (name or env("RL_TRACK", default="shared"))

    def anchor(self, step, name=None):
        return self.track(name) / "anchors" / f"step_{step:06d}.json"

    def ready(self, step, name=None):
        return self.track(name) / "ready" / f"step_{step:06d}.json"

    @staticmethod
    def atomic(path, value):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex[:6]}.tmp")
        temp.write_text(json.dumps(value, ensure_ascii=False, default=str), encoding="utf-8")
        temp.replace(path)

    @staticmethod
    def read(path, default=None):
        path = Path(path)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default

    @staticmethod
    def append(path, records):
        """Atomically append whole JSONL lines and return durable byte references.

        Offsets and SHA-256 checksums let training consume exactly the samples
        persisted to accepted_training.jsonl, not in-memory copies.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        refs = []
        with path.with_suffix(path.suffix + ".lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            with path.open("ab") as handle:
                for row in records:
                    line = (json.dumps(row, ensure_ascii=False, default=str) + "\n").encode("utf-8")
                    offset = handle.tell()
                    handle.write(line)
                    refs.append({"offset": offset, "sha256": hashlib.sha256(line).hexdigest()})
                handle.flush()
                os.fsync(handle.fileno())
        return refs

    @staticmethod
    def read_refs(path, refs):
        """Read and integrity-check exact saved JSONL records under a shared lock."""
        path = Path(path)
        results = []
        with path.with_suffix(path.suffix + ".lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_SH)
            with path.open("rb") as handle:
                for ref in refs:
                    offset = int(ref["offset"])
                    if offset < 0:
                        raise ValueError(f"Negative JSONL offset: {offset}")
                    handle.seek(offset)
                    line = handle.readline()
                    if not line.endswith(b"\n") or hashlib.sha256(line).hexdigest() != ref["sha256"]:
                        raise ValueError(f"Persisted training sample missing/corrupted: {path} offset={offset}")
                    results.append(json.loads(line))
        return results

    def code_records(self, name=None):
        return self.track(name) / "generated_code.jsonl"


files = Files()


class PrivateHub:
    """Single private-HF contract for both model and dataset repositories."""

    repo_type = None
    prefix_setting = None

    def __init__(self, name, track="shared"):
        self.name, self.track = name, track
        self.state_path = files.output / "hf" / f"{self.repo_type}_{track}_{name}.json"

    @property
    def client(self):
        from huggingface_hub import HfApi
        token = os.environ.get("RL_HF_TOKEN") or os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF write token is required: private upload cannot be skipped")
        return HfApi(token=token)

    @property
    def token(self):
        return os.environ.get("RL_HF_TOKEN") or os.environ.get("HF_TOKEN")

    def repository(self):
        state = files.read(self.state_path, {})
        if not state.get("repo"):
            Progress.log("hf", "creating PRIVATE Hugging Face repository", kind=self.repo_type,
                         track=self.track, name=self.name)
            owner = env("RL_HF_OWNER") or self.client.whoami()["name"]
            uid = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
            repo = f"{owner}/{env(self.prefix_setting)}-{self.track}-{self.name}-{uid}"
            self.client.create_repo(repo_id=repo, repo_type=self.repo_type, private=True, exist_ok=False)
            state["repo"] = repo
            files.atomic(self.state_path, state)
        repo = state["repo"]
        Progress.pulse("hf", f"repo-{self.repo_type}-{self.track}-{self.name}",
                       "checking repository privacy", kind=self.repo_type, repo=repo)
        if not self.client.repo_info(repo_id=repo, repo_type=self.repo_type).private:
            raise RuntimeError(f"Refusing to upload/download PUBLIC HF repo: {repo}")
        return repo


class PrivateDataset(PrivateHub):
    """Durable local event log plus accepted-only training log, then PRIVATE HF.

    Both streams have independent upload cursors. Every generated attempt and
    every GRPO rollout is retained, even when test/PCA verification fails.
    Only accepted synthesis examples enter accepted_training.jsonl.
    """

    repo_type, prefix_setting = "dataset", "RL_HF_DATA_PREFIX"

    def __init__(self, name, track="shared"):
        super().__init__(name, track)
        # A single repo is reused across runs/tracks; cursors remain run/track-specific.
        # RL_HF_DATA_REPO can pin an existing private dataset repository explicitly.
        # Use a stable code-directory fallback instead of a per-run output directory.
        # This survives new RL_RUN_ID values even if rl_config.sh was not sourced.
        self.state_path = Path(env("RL_ROOT", default=str(Path(__file__).resolve().parent))) / "hf_data_repository.json"

    def repository(self):
        configured = env("RL_HF_DATA_REPO", default="").strip()
        lock_path = self.state_path.with_suffix(".json.lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = files.read(self.state_path, {})
            repo = configured or state.get("repo")
            if not repo:
                # Migrate a repo created by the prior release during THIS run.
                legacy = files.output / "hf" / f"dataset_{self.track}_{self.name}.json"
                repo = files.read(legacy, {}).get("repo")
            if not repo:
                owner = env("RL_HF_OWNER") or self.client.whoami()["name"]
                uid = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
                repo = f"{owner}/{env(self.prefix_setting)}-{uid}"
                Progress.log("hf-data", "creating reusable PRIVATE dataset repository", repo=repo)
                self.client.create_repo(repo_id=repo, repo_type="dataset", private=True, exist_ok=False)
            # Never save/use a public repo, including a user-configured repo.
            if not self.client.repo_info(repo_id=repo, repo_type="dataset").private:
                raise RuntimeError(f"Refusing PUBLIC Hugging Face dataset repository: {repo}")
            if state.get("repo") != repo:
                files.atomic(self.state_path, {"repo": repo})
            else:
                Progress.pulse("hf-data", "reuse-private-dataset", "reusing PRIVATE dataset repository",
                               repo=repo, state_file=self.state_path)
            return repo

    @property
    def path(self):
        return files.code_records(self.track)

    @property
    def accepted_path(self):
        return files.track(self.track) / "accepted_training.jsonl"

    def streams(self):
        return {"all_attempts": self.path, "accepted_training": self.accepted_path}

    @staticmethod
    def _validate_training_record(row, step=None):
        """One acceptance invariant for both JSONL writes and GRPO reads."""
        from custome.rl.data_load import Sample
        sample = Sample.from_dict(row)
        sample.assert_prompt_integrity()
        if ((step is not None and sample.step != step) or
                sample.source != "synth_gen" or sample.accepted is not True or
                sample.passed is not True or sample.synth_gen_passed is not True or
                not sample.generated_code or not sample.test_code or
                not sample.metadata.get("pca_matched")):
            raise ValueError(
                f"Unverified JSONL training sample at step={step}: "
                f"source_id={sample.source_id!r}, attempt={sample.attempt}"
            )

    def append(self, records):
        from custome.rl.data_load import DataLoad
        mapper = DataLoad()
        normalized = [(mapper.rollout(row) if row.get("source") == "grpo"
                       else mapper.synth(row)).to_dict() for row in records]
        accepted = [row for row in normalized if row.get("accepted") is True and
                    row.get("source") == "synth_gen"]
        # Audit history must survive even if an incorrectly marked accepted row
        # fails the strict training-only validation below.
        files.append(self.path, normalized)
        for row in accepted:
            self._validate_training_record(row)
        # Fsync is complete before the producer exposes these byte references.
        refs = files.append(self.accepted_path, accepted) if accepted else []
        if refs:
            Progress.log("data-store", "accepted samples durably saved to JSONL",
                         count=len(refs), path=self.accepted_path)
        return refs

    def load_accepted(self, refs, step):
        """GRPO MUST read accepted examples from the local JSONL, not a queue copy."""
        rows = files.read_refs(self.accepted_path, refs)
        for row in rows:
            self._validate_training_record(row, step)
        return rows

    def _upload_stream(self, repo, name, path, flush):
        if not path.exists():
            return 0
        cursor = path.with_suffix(".upload_state.json")
        state = files.read(cursor, {"uploaded": 0, "byte_offset": 0, "chunks": 0})
        if state.get("repo") and state["repo"] != repo:
            raise RuntimeError(f"Upload cursor belongs to {state['repo']}, refusing switch to {repo}")
        if int(state.get("uploaded", 0)) and not state.get("repo"):
            raise RuntimeError(f"Legacy upload cursor has no destination repo: {cursor}; inspect manually")
        size = env("RL_UPLOAD_EVERY", int)
        if size < 1:
            raise ValueError("RL_UPLOAD_EVERY must be >=1")
        start_offset = int(state.get("byte_offset", 0))
        if "byte_offset" not in state and int(state.get("uploaded", 0)):
            # Read legacy line-count cursor only once when upgrading an older run.
            with path.open("rb") as reader:
                for _ in range(int(state["uploaded"])):
                    if not reader.readline():
                        raise RuntimeError(f"Invalid old upload cursor: {path}")
                start_offset = reader.tell()
        with path.with_suffix(path.suffix + ".lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            with path.open("rb") as reader:
                reader.seek(start_offset)
                # Stream only the next chunk, never reload a huge entire JSONL file.
                lines = [reader.readline() for _ in range(size)]
                lines = [line for line in lines if line]
                end_offset = reader.tell()
        if not lines or (len(lines) < size and not flush):
            return 0
        chunk = int(state["chunks"])
        temp = path.parent / f".hf_{name}_{chunk:07d}_{os.getpid()}.jsonl"
        temp.write_bytes(b"".join(lines))
        try:
            Progress.log("hf-data", "uploading JSONL chunk to PRIVATE HF dataset", stream=name,
                         records=len(lines), part=chunk, repo=repo)
            run_id = env("RL_RUN_ID", default="manual")
            if any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for c in run_id):
                raise ValueError("RL_RUN_ID contains invalid HF path characters")
            self.client.upload_file(
                path_or_fileobj=str(temp),
                path_in_repo=f"runs/{run_id}/tracks/{self.track}/data/{name}/part-{chunk:07d}.jsonl",
                repo_id=repo, repo_type="dataset",
                commit_message=f"{run_id}/{self.track}/{name}: records {state['uploaded']}:{int(state['uploaded']) + len(lines)}",
            )
            files.atomic(cursor, {
                "repo": repo,
                "uploaded": int(state["uploaded"]) + len(lines),
                "byte_offset": end_offset, "chunks": chunk + 1,
            })
        finally:
            temp.unlink(missing_ok=True)
        Progress.log("hf-data", "JSONL chunk uploaded", stream=name,
                     total_uploaded=int(state["uploaded"]) + len(lines), part=chunk)
        return len(lines)

    def upload(self, flush=False):
        repo = self.repository()  # fail closed if the repo is not private
        total = 0
        for name, path in self.streams().items():
            # Separate lock prevents multiple uploaders from advancing a stream's
            # cursor concurrently, without blocking the generator's append lock.
            upload_lock_path = path.with_suffix(".hf_upload.lock")
            upload_lock_path.parent.mkdir(parents=True, exist_ok=True)
            with upload_lock_path.open("a+") as upload_lock:
                fcntl.flock(upload_lock, fcntl.LOCK_EX)
                if flush:
                    # Drain *every* remaining chunk, not just one, before finishing.
                    while True:
                        moved = self._upload_stream(repo, name, path, flush=True)
                        total += moved
                        if not moved:
                            break
                else:
                    total += self._upload_stream(repo, name, path, flush=False)
        return total

    def _uploaded_all(self):
        for path in self.streams().values():
            if not path.exists():
                continue
            state = files.read(path.with_suffix(".upload_state.json"), {})
            if int(state.get("byte_offset", -1)) != path.stat().st_size:
                return False
        return True

    async def watch(self, finished):
        self.repository()
        ready = files.track(self.track) / ".uploader_ready"
        ready.parent.mkdir(parents=True, exist_ok=True)
        ready.write_text("private HF dataset verified", encoding="utf-8")
        attempts = 0
        Progress.log("hf-data", "background private upload watcher ready", track=self.track,
                     poll_s=env("RL_UPLOAD_POLL_SECS", float))
        while True:
            Progress.pulse("hf-data", f"watch-{self.track}", "waiting for more generated samples",
                           track=self.track, finished=finished.exists())
            try:
                await asyncio.to_thread(self.upload, finished.exists())
                attempts = 0
            except Exception as exc:
                attempts += 1
                print(f"[HF upload retry {attempts}] {exc}", flush=True)
                if attempts >= env("RL_HF_MAX_RETRIES", int):
                    raise
            if finished.exists() and self._uploaded_all():
                Progress.log("hf-data", "all local records uploaded", track=self.track)
                return
            await asyncio.sleep(env("RL_UPLOAD_POLL_SECS", float))


class PrivateModel(PrivateHub):
    repo_type, prefix_setting = "model", "RL_HF_MODEL_PREFIX"

    def upload_folder(self, folder, subdir=None):
        repo = self.repository()
        Progress.log("hf-model", "uploading checkpoint/model folder", source=folder, subdir=subdir, repo=repo)
        self.client.upload_folder(folder_path=str(folder), repo_id=repo, repo_type="model",
                                  path_in_repo=subdir, commit_message=f"Save {Path(folder).name}")
        Progress.log("hf-model", "model folder uploaded", source=folder, repo=repo)
        return repo

    def download(self, target):
        from huggingface_hub import snapshot_download
        Progress.log("hf-model", "downloading private model snapshot", target=target)
        return snapshot_download(repo_id=self.repository(), repo_type="model", local_dir=str(target), token=self.token)

    @staticmethod
    def completed(folder):
        folder = Path(folder)
        tracker = folder / "latest_checkpointed_iteration.txt"
        if not tracker.exists():
            return []
        limit = int(tracker.read_text().strip())
        return sorted((p for p in folder.glob("global_step_*") if p.is_dir()
                       and int(p.name.rsplit("_", 1)[-1]) <= limit),
                      key=lambda p: int(p.name.rsplit("_", 1)[-1]))

    def watch(self, checkpoint_folder):
        folder = Path(checkpoint_folder)
        marker = folder / ".hf_watcher_ready"
        progress = folder / ".uploaded_checkpoints.json"
        self.repository()
        marker.write_text("private HF model repository verified")
        done = set(files.read(progress, []))
        Progress.log("hf-model", "checkpoint upload watcher ready", folder=folder,
                     already_uploaded=len(done))
        while True:
            Progress.pulse("hf-model", f"watch-{folder}", "monitoring checkpoints",
                           folder=folder, completed=len(done))
            for ckpt in self.completed(folder):
                if ckpt.name not in done:
                    self.upload_folder(ckpt, f"checkpoints/{ckpt.name}")
                    done.add(ckpt.name)
                    files.atomic(progress, sorted(done))
            if (folder / ".training_finished").exists():
                Progress.log("hf-model", "checkpoint upload watcher finished", folder=folder,
                             uploaded=len(done))
                return
            time.sleep(5)
