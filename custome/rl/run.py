"""Single entrypoint: shared SFT then RL-only track branching, monitors, and eval.
Run from the project's activated environment after sourcing rl_config.sh.
"""
from __future__ import annotations

import argparse
import hashlib
import asyncio
import atexit
import os
import secrets
import shutil
import signal
import subprocess
import sys
import time
import threading
from collections import deque
from pathlib import Path

from custome.rl.core import PrivateDataset, PrivateModel, Progress, env, files, grpo_steps, require_model_lineage


def verl_command(*args):
    repo = env("RL_VERL_REPO")
    return ["uv", "run", "--project", repo, "--frozen", "--all-packages",
            "--extra", "fsdp", "--extra", "vllm", *map(str, args)]


class ProcessGroup:
    """Only one implementation of subprocess lifetime / log / failure handling."""

    def __init__(self):
        self.children = []

    def start(self, label, command, **env_overrides):
        log = files.output / f"{label}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        Progress.log("pipeline-worker", "starting background worker", label=label, log=str(log))
        child_env = {**os.environ, **env_overrides, "PYTHONUNBUFFERED": "1"}
        child = subprocess.Popen(command, env=child_env,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, encoding="utf-8", errors="replace", bufsize=1)
        # Read pipes continuously so heavy verifier output cannot deadlock.
        # Preserve original complete log while echoing marked lines to terminal.
        def relay():
            with log.open("a", encoding="utf-8") as handle:
                try:
                    for line in child.stdout:
                        handle.write(line)
                        handle.flush()
                        if os.environ.get("RL_PROGRESS_CHILD_OUTPUT", "1") != "0":
                            print(f"[{label}] {line.rstrip()}", flush=True)
                finally:
                    child.stdout.close()
        thread = threading.Thread(target=relay, name=f"log-{label}", daemon=True)
        thread.start()
        self.children.append((label, child, log))
        return child

    @staticmethod
    def wait_ready(child, marker, timeout=120):
        begin = time.monotonic()
        end = begin + timeout
        while time.monotonic() < end:
            if Path(marker).exists():
                Progress.log("pipeline-worker", "worker ready", pid=child.pid, marker=marker,
                             elapsed_s=round(time.monotonic()-begin,1))
                return
            if child.poll() is not None:
                raise RuntimeError(f"Background worker stopped early: {child.pid} exit={child.returncode}")
            Progress.pulse("pipeline-worker", f"ready-{child.pid}", "waiting for ready marker",
                           marker=marker, elapsed_s=round(time.monotonic()-begin,1))
            time.sleep(.5)
        raise TimeoutError(f"Background worker not ready: {marker}")

    @staticmethod
    def wait(child):
        try:
            Progress.wait_process(child, "pipeline-worker", label=f"worker pid={child.pid}")
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"Worker process {child.pid} exited {error.returncode}") from error

    def close(self):
        for _, child, _ in reversed(self.children):
            if child.poll() is None:
                child.terminate()
        for _, child, _ in self.children:
            if child.poll() is None:
                try:
                    child.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    child.kill()
        self.children.clear()


class Trainer:
    """SFT and GRPO use VERL. No optimizer loop is reimplemented."""

    def __init__(self):
        self.root = Path(__file__).resolve().parent

    def train(self, mode, task, model, track="shared"):
        model = require_model_lineage(model)
        n_gpus = env("RL_GPUS_PER_NODE", int)
        assert n_gpus >= 1
        if mode == "sft":
            if env("RL_SFT_BATCH", int) % n_gpus:
                raise ValueError("SFT global batch must be divisible by GPUs/node")
            command = verl_command("torchrun", f"--nnodes={env('RL_NNODES')}",
                                   f"--nproc_per_node={n_gpus}", f"--node_rank={env('RL_NODE_RANK')}",
                                   f"--master_addr={env('RL_MASTER_ADDR')}",
                                   f"--master_port={env('RL_MASTER_PORT')}",
                                   "-m", "custome.rl.run", "sft_entry",
                                   "engine=fsdp", f"engine.strategy={env('RL_FSDP_STRATEGY')}",
                                   f"engine.fsdp_size={env('RL_FSDP_SIZE')}",
                                   f"engine.param_offload={env('RL_PARAM_OFFLOAD')}",
                                   f"engine.optimizer_offload={env('RL_OPTIM_OFFLOAD')}",
                                   f"model.path={model}",
                                   f"model.enable_gradient_checkpointing={env('RL_GRAD_CKPT')}",
                                   f"data.train_files={files.data / (task + '_train.parquet')}",
                                   f"data.val_files={files.data / (task + '_eval.parquet')}",
                                   "data.messages_key=messages", "data.num_workers=0",
                                   f"data.train_batch_size={env('RL_SFT_BATCH')}",
                                   f"data.micro_batch_size_per_gpu={env('RL_SFT_MICRO')}",
                                   f"data.max_length={env('RL_SFT_MAX_LENGTH')}",
                                   "data.truncation=right", f"optim.lr={env('RL_SFT_LR')}",
                                   f"trainer.total_epochs={env('RL_SFT_EPOCHS')}",
                                   f"trainer.total_training_steps={env(f'RL_SFT_STEPS_{task.upper()}', int)}",
                                   f"trainer.seed={env('RL_RANDOM_SEED')}",
                                   f"trainer.save_freq={env('RL_MODEL_SAVE_EVERY')}",
                                   f"trainer.test_freq={env('RL_SFT_TEST_EVERY')}",
                                   "trainer.resume_mode=disable", 'trainer.logger=["console"]',
                                   f"trainer.default_local_dir={files.checkpoint / ('sft_' + task)}")
        elif mode == "grpo":
            if task != env("RL_GRPO_TASK") or task != "code":
                raise ValueError("This pipeline supports GRPO code task only, configured by RL_GRPO_TASK=code")
            if env("RL_GRPO_BATCH", int) * env("RL_ROLLOUT_N", int) % env("RL_GRPO_MINI", int):
                raise ValueError("Rollout samples must divide PPO mini batch")
            command = verl_command("python", "-m", "verl.trainer.main_ppo",
                                   "algorithm.adv_estimator=grpo", "algorithm.use_kl_in_reward=false",
                                   f"data.train_files={files.data / 'code_train.parquet'}",
                                   f"data.val_files={files.data / 'code_eval.parquet'}",
                                   "data.custom_dataloader={_target_:custome.rl.tracks.StreamingCodeLoader}",
                                   f"data.gen_batch_size={env('RL_GRPO_BATCH')}",
                                   f"data.train_batch_size={env('RL_GRPO_BATCH')}",
                                   f"data.max_prompt_length={env('RL_PROMPT_LEN')}",
                                   f"data.max_response_length={env('RL_RESPONSE_LEN')}",
                                   "data.truncation=right", f"actor_rollout_ref.model.path={model}",
                                   f"actor_rollout_ref.model.enable_gradient_checkpointing={env('RL_GRAD_CKPT')}",
                                   "actor_rollout_ref.model.use_remove_padding=true",
                                   f"actor_rollout_ref.actor.strategy={env('RL_FSDP_STRATEGY')}",
                                   f"actor_rollout_ref.ref.strategy={env('RL_FSDP_STRATEGY')}",
                                   f"actor_rollout_ref.actor.fsdp_config.param_offload={env('RL_PARAM_OFFLOAD')}",
                                   f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={env('RL_OPTIM_OFFLOAD')}",
                                   f"actor_rollout_ref.actor.optim.lr={env('RL_GRPO_LR')}",
                                   f"actor_rollout_ref.actor.ppo_mini_batch_size={env('RL_GRPO_MINI')}",
                                   f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={env('RL_GRPO_MICRO')}",
                                   "actor_rollout_ref.actor.use_kl_loss=true",
                                   f"actor_rollout_ref.actor.kl_loss_coef={env('RL_KL')}",
                                   "actor_rollout_ref.rollout.name=vllm",
                                   f"actor_rollout_ref.rollout.n={env('RL_ROLLOUT_N')}",
                                   f"actor_rollout_ref.rollout.tensor_model_parallel_size={env('RL_ROLLOUT_TP')}",
                                   f"actor_rollout_ref.rollout.gpu_memory_utilization={env('RL_ROLLOUT_MEM')}",
                                   "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
                                   "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
                                   f"reward.custom_reward_function.path={self.root / 'score.py'}",
                                   "reward.custom_reward_function.name=compute_score",
                                   "reward.reward_manager.name=naive", "reward.num_workers=1",
                                   f"trainer.n_gpus_per_node={env('RL_GPUS_PER_NODE')}",
                                   f"trainer.nnodes={env('RL_NNODES')}",
                                   f"trainer.total_training_steps={grpo_steps(track)}",
                                   "trainer.total_epochs=1", f"trainer.seed={env('RL_RANDOM_SEED')}",
                                   f"data.seed={env('RL_RANDOM_SEED')}",
                                   f"trainer.test_freq={env('RL_EVAL_EVERY')}",
                                   f"trainer.save_freq={env('RL_MODEL_SAVE_EVERY')}",
                                   'trainer.logger=["console"]', "trainer.v1.trainer_mode=sync",
                                   "trainer.resume_mode=disable",
                                   f"trainer.default_local_dir={files.checkpoint / ('grpo_' + track)}")
        else:
            raise ValueError("Expected sft or grpo")
        env_extra = {"RL_TRACK": track, "RL_SFT_TASK": task, "RL_TRAIN_MODEL_PATH": model}
        steps = env(f"RL_SFT_STEPS_{task.upper()}", int) if mode == "sft" else grpo_steps(track)
        Progress.log("trainer", "starting training", mode=mode, task=task, track=track,
                     steps=steps, model=model, gpus=n_gpus)
        child = subprocess.Popen(command, cwd=env("RL_VERL_REPO"),
                                 env={**os.environ, **env_extra, "PYTHONUNBUFFERED": "1"})
        Progress.wait_process(child, "trainer", label=f"{mode}/{task}/{track} steps={steps}")

    @staticmethod
    def merge(checkpoint_dir, target, cpu_only=False):
        environment = os.environ.copy()
        if cpu_only:
            environment["CUDA_VISIBLE_DEVICES"] = ""
        Progress.log("merge", "merging FSDP checkpoint", source=checkpoint_dir, target=target)
        child = subprocess.Popen(verl_command("python", "-m", "verl.model_merger", "merge", "--backend", "fsdp",
                                    "--local_dir", str(checkpoint_dir), "--target_dir", str(target)),
                                 env={**environment, "PYTHONUNBUFFERED": "1"})
        Progress.wait_process(child, "merge", label=str(target))

    @staticmethod
    def eval(model, stage, per_step=False):
        settings = os.environ.copy()
        if per_step:
            settings["RL_EVAL_LIMIT"] = env("RL_PER_STEP_EVAL_LIMIT")
            settings["CUDA_VISIBLE_DEVICES"] = env("RL_EVAL_CUDA_VISIBLE_DEVICES")
        Progress.log("eval", "starting evaluation", stage=stage, model=model)
        child = subprocess.Popen(verl_command("python", "-m", "custome.rl.tasks", "eval",
                                    "--model", str(model), "--stage", stage),
                                 env={**settings, "PYTHONUNBUFFERED": "1"})
        Progress.wait_process(child, "eval", label=stage)


class TraceSFT:
    """Record sampled *actually indexed* training examples during each SFT.

    Reuses the existing Code-SFT last-batch trace for synth seeds; no extra
    sampler or second dataset loader. Every rank writes its own selected set.
    """

    @staticmethod
    def install():
        task = env("RL_SFT_TASK", default="")
        if not task:
            return
        from verl.trainer import sft_trainer
        from custome.rl.data_load import DataLoad, Sample
        original = sft_trainer.SFTTrainer._build_dataset

        class TracedDataset:
            def __init__(self, dataset):
                self.dataset = dataset
                size = env("RL_SFT_BATCH", int) // env("RL_GPUS_PER_NODE", int)
                self.recent = deque(maxlen=size) if task == "code" else None
                self.selected, self.fallback, self.seen = {}, {}, set()
                self.fraction = env("RL_PCA_REFERENCE_FRACTION", float)
                if not 0 < self.fraction <= 1:
                    raise ValueError("RL_PCA_REFERENCE_FRACTION must be in (0, 1]")
                rank = env("RANK", int, "0")
                self.path = DataLoad.pca_trace_path(task, rank)
                self.last_path = files.output / "sft_final" / f"rank_{rank}.json"
                self.path.parent.mkdir(parents=True, exist_ok=True)
                atexit.register(self.save)

            def __len__(self):
                return len(self.dataset)

            def __getitem__(self, index):
                example = self.dataset[index]
                record = Sample.from_dict(self.dataset.dataframe.iloc[index].to_dict())
                if self.recent is not None:
                    self.recent.append(record.to_dict())
                identifier = record.source_id
                if identifier not in self.seen:
                    self.seen.add(identifier)
                    compact = {"source_id": identifier, "pca_text": record.pca_text(),
                               "task": task}
                    floor = max(4, env("RL_PCA_MIN_REFERENCE_SAMPLES", int))
                    if len(self.fallback) < floor:
                        self.fallback[identifier] = compact
                    # Stable deterministic Bernoulli sampling of distinct seen IDs.
                    fingerprint = hashlib.sha256(
                        f"{env('RL_RANDOM_SEED')}:{task}:{identifier}".encode()
                    ).digest()
                    if int.from_bytes(fingerprint[:8], "big") / 2**64 < self.fraction:
                        self.selected[identifier] = compact
                return example

            def save(self):
                files.atomic(self.path, {
                    "task": task, "fraction": self.fraction,
                    "observed_ids": sorted(self.seen),
                    "selected": list(self.selected.values()),
                    "fallback": list(self.fallback.values()),
                })
                if self.recent is not None:
                    files.atomic(self.last_path, list(self.recent))

        def build(trainer):
            original(trainer)
            trainer.train_dataset = TracedDataset(trainer.train_dataset)

        sft_trainer.SFTTrainer._build_dataset = build


def sft_entry():
    import hydra
    from verl.trainer import sft_trainer
    TraceSFT.install()

    @hydra.main(config_path=env("RL_VERL_REPO") + "/verl/trainer/config",
                config_name="sft_trainer_engine", version_base=None)
    def main(cfg):
        sft_trainer.run_sft(cfg)

    main()


class Pipeline:
    def __init__(self):
        self.workers = ProcessGroup()
        self.trainer = Trainer()
        self.python = [sys.executable, "-m", "custome.rl.run"]

    def background(self, label, action, *args, **overrides):
        return self.workers.start(label, self.python + [action, *map(str, args)], **overrides)

    def run_with_checkpoints(self, mode, task, model, stage, track="shared"):
        folder = files.checkpoint / stage
        folder.mkdir(parents=True, exist_ok=True)
        for old in (".training_finished", ".hf_watcher_ready"):
            (folder / old).unlink(missing_ok=True)
        Progress.log("pipeline", "starting monitored training stage", stage=stage, mode=mode, track=track)
        watcher = self.background(f"hf_ckpt_{stage}", "monitor", stage, folder, track)
        self.workers.wait_ready(watcher, folder / ".hf_watcher_ready")
        eval_watcher = None
        if env("RL_FULL_EVAL_EVERY", int) > 0:
            eval_watcher = self.background(f"eval_watch_{stage}", "eval_watch", stage, folder)
        try:
            self.trainer.train(mode, task, model, track)
        finally:
            (folder / ".training_finished").touch()
        self.workers.wait(watcher)
        if eval_watcher:
            self.workers.wait(eval_watcher)
        checkpoints = PrivateModel.completed(folder)
        if not checkpoints:
            raise RuntimeError(f"Training stage {stage} produced no completed checkpoints")
        Progress.log("pipeline", "training stage finished", stage=stage,
                     checkpoint=checkpoints[-1], checkpoint_count=len(checkpoints))
        return checkpoints[-1]

    def private_merge(self, checkpoint, name, track="shared", actor=False):
        merged = files.checkpoint / f"hf_after_{name}"
        Progress.log("pipeline", "merging then PRIVATE uploading checkpoint", stage=name, track=track)
        self.trainer.merge(checkpoint / "actor" if actor else checkpoint, merged)
        model = PrivateModel(name, track)
        Progress.log("pipeline", "uploading merged model to private HF repo", stage=name)
        model.upload_folder(merged)
        loaded = files.checkpoint / f"private_loaded_{name}"
        Progress.log("pipeline", "downloading model for next stage", stage=name)
        result = model.download(loaded)
        Progress.log("pipeline", "model ready for next stage", stage=name, path=result)
        return result

    def service(self, track, model):
        service = self.background(f"reward_{track}", "service", RL_TRACK=track,
                                  RL_TRAIN_MODEL_PATH=str(model),
                                  **({"CUDA_VISIBLE_DEVICES": env("RL_SERVICE_CUDA_VISIBLE_DEVICES")}
                                     if env("RL_SERVICE_CUDA_VISIBLE_DEVICES") else {}))
        started = time.monotonic()
        for _ in range(240):
            Progress.pulse("pipeline", f"service-{track}", "waiting for reward service model/PCA initialization",
                           track=track, elapsed_s=round(time.monotonic()-started,1))
            if service.poll() is not None:
                raise RuntimeError(f"Reward service terminated; see reward_{track}.log")
            try:
                from urllib.request import urlopen
                with urlopen(env("RL_SERVICE_URL") + "/health", timeout=.5) as r:
                    if r.status == 200:
                        Progress.log("pipeline", "reward service ready", track=track,
                                     elapsed_s=round(time.monotonic()-started,1))
                        return service
            except Exception:
                time.sleep(.5)
        raise TimeoutError("Reward service never became healthy")

    def run(self):
        from custome.rl.tracks import CodeTrack, bootstrap_sft_anchor
        from custome.rl.data_load import DataLoad
        from custome.rl.tasks import Comparator
        if env("RL_NNODES", int) != 1:
            raise RuntimeError("Automated pipeline needs a single node; multi-node trainer requires external coordination")
        if not (os.environ.get("HF_TOKEN") or os.environ.get("RL_HF_TOKEN")):
            raise RuntimeError("HF_TOKEN is mandatory; private uploads cannot be disabled")
        if not os.environ.get("SYNTHGEN_REMOTE_TOKEN"):
            raise RuntimeError("SYNTHGEN_REMOTE_TOKEN is required for remote Podman code tests")
        os.environ["RL_SERVICE_TOKEN"] = env("RL_SERVICE_TOKEN") or secrets.token_hex(32)
        files.output.mkdir(parents=True, exist_ok=True)
        Progress.log("pipeline", "FULL EXPERIMENT START", model=env("RL_BASE_MODEL"),
                     sft_tasks=env("RL_SFT_TASKS"), tracks=env("RL_TRACKS"),
                     output=files.output)
        try:
            Progress.log("pipeline", "phase 1: HF preflight")
            self.workers.wait(self.background("preflight", "preflight"))
            Progress.log("pipeline", "phase 2: download and normalize train/eval datasets")
            DataLoad().prepare()
            Progress.log("pipeline", "auditing all saved SFT/eval prompts for answer/code leakage")
            DataLoad.audit_cached()
            Progress.log("pipeline", "phase 3: evaluate base model")
            self.trainer.eval(env("RL_BASE_MODEL"), "base")
            model = env("RL_BASE_MODEL")
            tasks = env("RL_SFT_TASKS").split()
            if (not tasks or tasks[-1] != "code" or len(tasks) != len(set(tasks))
                    or not set(tasks).issubset({"math", "social", "code"})):
                raise ValueError("RL_SFT_TASKS must be unique tasks from math/social/code, ending with code")
            if env("RL_GRPO_TASK") != "code":
                raise ValueError("RL_GRPO_TASK must be code for code synthesis and Code-SFT anchor")
            for task_index, task in enumerate(tasks, 1):
                Progress.log("pipeline", "SFT task starting", task=task,
                             index=task_index, total=len(tasks), steps=env(f"RL_SFT_STEPS_{task.upper()}",int))
                ckpt = self.run_with_checkpoints("sft", task, model, f"sft_{task}")
                Progress.log("pipeline", "consolidating PCA sample trace", task=task)
                DataLoad.consolidate_pca_trace(task)
                model = self.private_merge(ckpt, task)
                self.trainer.eval(model, task)
                Progress.log("pipeline", "SFT task complete", task=task,
                             index=task_index, total=len(tasks))
            # Branch ONLY here. Every GRPO track starts with the identical Code SFT checkpoint.
            common_model = model
            # One frozen Code-SFT encoder for ALL task reference spaces and
            # candidate scoring. Fitting during each SFT with its changing
            # weights would make reference/candidate coordinates incomparable.
            Progress.log("pipeline", "phase 4: build prior-task PCA reference banks")
            self.workers.wait(self.background(
                "pca_build", "pca_build", env("RL_FEATURE_MODEL")
            ))
            Progress.log("pipeline", "prior-task PCA reference banks ready")
            track_names = env("RL_TRACKS").split()
            for track_index, name in enumerate(track_names, 1):
                Progress.log("pipeline", "GRPO track starting", track=name, index=track_index,
                             total=len(track_names), steps=grpo_steps(name))
                track = CodeTrack.make(name)
                grpo_steps(name)  # Validate that each configured track has its own step count.
                os.environ["RL_TRACK"] = name
                os.environ["RL_TRAIN_MODEL_PATH"] = str(common_model)
                track_dir = files.track(name)
                track_dir.mkdir(parents=True, exist_ok=True)
                for filename in (".data_finished", ".uploader_ready", ".policy_watcher_ready", ".producer_finished"):
                    (track_dir / filename).unlink(missing_ok=True)
                service = self.service(name, common_model)
                uploader = self.background(f"hf_data_{name}", "upload_watch", name)
                self.workers.wait_ready(uploader, track_dir / ".uploader_ready")
                producer = None
                policy_watcher = None
                try:
                    # Both tracks use the SFT-final batch for GRPO step 0.
                    bootstrap_sft_anchor(name)
                    if track.stream:
                        # The only model used by synth_gen is an evolving
                        # snapshot of this same GRPO actor, not a separate LLM.
                        ckpt_folder = files.checkpoint / f"grpo_{name}"
                        ckpt_folder.mkdir(parents=True, exist_ok=True)
                        (ckpt_folder / ".training_finished").unlink(missing_ok=True)
                        policy_watcher = self.background(
                            f"policy_{name}", "policy_watch", name, str(common_model)
                        )
                        self.workers.wait_ready(
                            policy_watcher, track_dir / ".policy_watcher_ready"
                        )
                        prod_env = {"RL_TRACK": name}
                        producer = self.background(f"synth_{name}", "producer", **prod_env)
                    ckpt = self.run_with_checkpoints("grpo", "code", common_model, f"grpo_{name}", name)
                    if producer:
                        self.workers.wait(producer)
                    if policy_watcher:
                        self.workers.wait(policy_watcher)
                    (track_dir / ".data_finished").touch()
                    self.workers.wait(uploader)
                    merged = self.private_merge(ckpt, f"grpo_{name}", track=name, actor=True)
                    self.trainer.eval(merged, f"grpo_{name}")
                    Progress.log("pipeline", "GRPO track finished and evaluated",
                                 track=name, index=track_index, total=len(track_names))
                finally:
                    if producer and producer.poll() is None:
                        producer.terminate()
                    if policy_watcher and policy_watcher.poll() is None:
                        policy_watcher.terminate()
                    if service.poll() is None:
                        service.terminate()
                        service.wait(timeout=20)
            Progress.log("pipeline", "phase 5: comparing baseline vs ours")
            Comparator().compare()
            Progress.log("pipeline", "FULL EXPERIMENT FINISHED", output=files.output)
        finally:
            self.workers.close()


def eval_watcher(stage, folder):
    folder = Path(folder)
    interval = env("RL_FULL_EVAL_EVERY", int)
    done = set()
    runner = Trainer()
    while True:
        for ckpt in PrivateModel.completed(folder):
            step = int(ckpt.name.rsplit("_", 1)[-1])
            if step in done or step % interval:
                continue
            merged = files.checkpoint / "eval_snapshots" / stage / f"step_{step}"
            runner.merge(ckpt / "actor" if stage.startswith("grpo_") else ckpt, merged)
            runner.eval(merged, f"{stage}_step_{step}", per_step=True)
            done.add(step)
        if (folder / ".training_finished").exists():
            return
        time.sleep(10)


def policy_watch(track, initial_model):
    """Publish merged snapshots of the *trained GRPO actor* for synth_gen.

    Snapshot step -1 = Code-SFT policy before the first GRPO update.
    VERL global_step_1 = first update, which we label policy_step=0.
    Publishing is atomic and generation reads only complete model folders.
    """
    folder = files.checkpoint / f"grpo_{track}"
    out = files.track(track)
    out.mkdir(parents=True, exist_ok=True)
    pointer = out / "policy_latest.json"
    ready = out / ".policy_watcher_ready"
    error_file = out / "policy_watcher_error.txt"
    error_file.unlink(missing_ok=True)
    interval = env("RL_POLICY_SNAPSHOT_EVERY", int)
    if interval < 1:
        raise ValueError("RL_POLICY_SNAPSHOT_EVERY must be positive")
    files.atomic(pointer, {"path": str(initial_model), "policy_step": -1})
    ready.write_text("actor policy snapshot service ready", encoding="utf-8")
    Progress.log("policy-watch", "watcher ready", track=track, snapshot=initial_model)
    done = set()
    runner = Trainer()
    try:
        while True:
            for ckpt in PrivateModel.completed(folder):
                number = int(ckpt.name.rsplit("_", 1)[-1])
                if number in done or number % interval:
                    continue
                target = files.checkpoint / "policy_snapshots" / track / f"step_{number:06d}"
                if not (target / "config.json").exists():
                    runner.merge(ckpt / "actor", target, cpu_only=True)
                if not (target / "config.json").exists():
                    raise RuntimeError(f"Model merger did not produce a usable HF model: {target}")
                # Latest pointer never references a partially merged model.
                files.atomic(pointer, {"path": str(target), "policy_step": number - 1})
                Progress.log("policy-watch", "new actor snapshot published", step=number-1, path=target)
                done.add(number)
                # Never remove a snapshot while a pool worker could be loading it.
                # Multiprocess jobs may be queued with an older pointer; cleanup
                # is safe only AFTER both actor training AND producer exit.
            if (folder / ".training_finished").exists() and (out / ".producer_finished").exists():
                kept = env("RL_POLICY_KEEP_SNAPSHOTS", int)
                if kept > 0:
                    snapshots = sorted((files.checkpoint / "policy_snapshots" / track).glob("step_*"))
                    for old in snapshots[:-kept]:
                        shutil.rmtree(old)
                return
            time.sleep(env("RL_POLICY_POLL_SECS", float))
    except BaseException as error:
        error_file.write_text(f"{type(error).__name__}: {error}", encoding="utf-8")
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["pipeline", "train", "sft_entry", "service", "producer",
                                            "monitor", "eval_watch", "upload_watch", "policy_watch", "pca_build", "preflight"])
    parser.add_argument("args", nargs="*")
    a, extra = parser.parse_known_args()
    if a.action == "pipeline": Pipeline().run()
    elif a.action == "train":
        mode, task, model, *rest = a.args
        Trainer().train(mode, task, model, rest[0] if rest else "shared")
    elif a.action == "sft_entry":
        # Hydra must see only overrides; remove this CLI subcommand from argv.
        sys.argv = [sys.argv[0], *a.args, *extra]
        sft_entry()
    elif a.action == "service":
        from custome.rl.score import serve
        serve()
    elif a.action == "producer":
        from custome.rl.tracks import produce
        produce()
    elif a.action == "monitor": PrivateModel(a.args[0], a.args[2]).watch(a.args[1])
    elif a.action == "upload_watch":
        name = a.args[0]
        asyncio.run(PrivateDataset("generated", name).watch(files.track(name) / ".data_finished"))
    elif a.action == "policy_watch": policy_watch(*a.args)
    elif a.action == "pca_build":
        if a.args[0] != env("RL_FEATURE_MODEL"):
            raise ValueError("PCA build must use the SAME Code-SFT model as reward scoring")
        from custome.rl.score import FeatureSpace, TaskPCABank
        bank = TaskPCABank.build(FeatureSpace())
        print(f"[prior PCA] built {len(bank.tasks)} task banks: {bank.tasks}", flush=True)
    elif a.action == "eval_watch": eval_watcher(*a.args)
    elif a.action == "preflight":
        from custome.rl.core import PrivateHub
        from huggingface_hub import HfApi
        HfApi(token=os.environ.get("HF_TOKEN") or os.environ.get("RL_HF_TOKEN")).whoami()
        from custome.rl.tracks import CodeTrack
        for name in env("RL_TRACKS").split():
            CodeTrack.make(name)
        repo = PrivateDataset("generated", "ours").repository()
        print(f"[OK] HF authenticated; private dataset repo: {repo}; RL tracks valid", flush=True)


if __name__ == "__main__":
    main()
