"""RL-only extension point. Baseline and synthesized data share one loader API.
Producer uses asyncio and a bounded verification queue; consumer is VERL's sync loader.
"""
from __future__ import annotations

import asyncio
import time
from uuid import uuid4

from custome.rl.core import PrivateDataset, Progress, env, files, grpo_steps
from custome.rl.data_load import DataLoad, Sample
from custome.rl.score import call_service


class CodeTrack:
    """Ordinary APPS baseline; subclasses override ONLY data provision/filtering."""

    registry = {}
    stream = False
    filter_distance = False

    def __init_subclass__(cls, name=None, **kwargs):
        super().__init_subclass__(**kwargs)
        if name:
            CodeTrack.registry[name] = cls

    @classmethod
    def make(cls, name):
        if name not in cls.registry:
            raise ValueError(f"Unregistered RL track: {name}. Available: {list(cls.registry)}")
        return cls.registry[name](name)

    def __init__(self, name):
        self.name = name
        self.sources = {}

    def get_original(self, split):
        if split not in self.sources:
            rows = [r.to_dict() for r in DataLoad.read("code", "eval" if split == "val" else "train")]
            if split != "val":
                rows = [r for r in rows if Sample.from_dict(r).code]
            if not rows:
                raise ValueError(f"Empty normalized code dataset: {split}")
            self.sources[split] = rows
        return self.sources[split]

    def batch(self, step, size, split):
        records = self.get_original(split)
        if split == "val":
            return records[step * size:(step + 1) * size]
        return [records[(step * size + i) % len(records)] for i in range(size)]

    def consumed(self, step, rows):
        """Preserve previous input batch for synth_gen seed reuse ONLY.

        PCA scoring now uses the persistent 10%-sampled earlier SFT task banks.
        """
        batch = [Sample.from_dict(row) for row in rows]
        files.atomic(files.anchor(step + 1, self.name), {
            "step": step + 1, "reference_step": step,
            # Do NOT forward gold/previous prediction code to synth_gen.
            # Only prior batch QUESTION texts are used as generation seeds.
            "problems": [row.question for row in batch],
            "source": "previous_grpo_input_batch",
        })

    def rejection_reason(self, record, duplicate=False):
        """One decision policy shared by the producer and all synthesized tracks."""
        # The reward service deliberately returns only verification fields
        # (record=False); it has no prompt/question to inspect. The producer
        # checks the complete candidate BEFORE scoring, and JSONL validates
        # it again before training.
        if record.get("question") and record.get("kind"):
            try:
                Sample.from_dict(record).assert_prompt_integrity()
            except ValueError:
                return "prompt_leak"
        if duplicate:
            return "duplicate"
        if record.get("passed") is not True and env("RL_REQUIRED_PASS") == "1":
            return "test_failed"
        if (record.get("source") == "synth_gen" and
                record.get("synth_gen_passed") is not True and
                env("RL_REQUIRE_SYNTH_VERIFIER") == "1"):
            return "synth_gen_verifier"
        if self.filter_distance:
            metadata = record.get("metadata") or {}
            if not metadata.get("pca_by_task"):
                return "pca_unavailable"
            if not metadata.get("pca_matched"):
                return "distance"
        return ""

    def accepts(self, record):
        return not self.rejection_reason(record)


class Baseline(CodeTrack, name="baseline"):
    pass


class SynthTrack(CodeTrack, name="ours"):
    stream = True
    filter_distance = True

    def batch(self, step, size, split):
        if split == "val":
            return super().batch(step, size, split)
        ready = files.ready(step, self.name)
        error = files.track(self.name) / f"producer_error_{step:06d}.txt"
        start = time.monotonic()
        while not ready.exists():
            Progress.pulse("grpo-loader", f"ready-{self.name}-{step}",
                           "waiting for synthesized training batch", track=self.name,
                           step=step, expected=size,
                           elapsed_s=round(time.monotonic()-start,1))
            if error.exists():
                raise RuntimeError(error.read_text(encoding="utf-8"))
            time.sleep(0.5)
        manifest = files.read(ready)
        if not isinstance(manifest, dict) or manifest.get("format") != "accepted_jsonl_refs_v1":
            raise ValueError(f"Step {step}: readiness manifest must reference accepted_training.jsonl")
        if manifest.get("step") != step:
            raise ValueError(f"Step mismatch in readiness manifest: {ready}")
        refs = manifest["records"]
        if len(refs) != size:
            raise RuntimeError(f"Step {step}: expected {size}, got {len(refs)}")
        rows = PrivateDataset("generated", self.name).load_accepted(refs, step)
        Progress.log("grpo-loader", "reading persisted JSONL training batch", track=self.name,
                     step=step, count=len(rows), source=str(files.track(self.name) / "accepted_training.jsonl"),
                     elapsed_s=round(time.monotonic()-start,1))
        return rows


class StreamingCodeLoader:
    """VERL V1 data.custom_dataloader._target_. Same class for every track."""

    def __init__(self, *, _data_config, _split, _batch_size, _tokenizer, _processor, _collate_fn):
        self.split, self.size, self.collate = _split, _batch_size, _collate_fn
        self.step = 0
        self.track = CodeTrack.make(env("RL_TRACK"))

    @staticmethod
    def to_verl(rows, step, split="train"):
        return [Sample.from_dict(row).to_verl(step, split) for row in rows]

    def __iter__(self):
        if self.split == "val":
            limit = env("RL_GRPO_VAL_MAX", int)
            rows = self.track.get_original("val")[:limit]
            for i in range(0, len(rows), self.size):
                yield self.collate(self.to_verl(rows[i:i + self.size], i // self.size, split="val"))
            return
        while self.step < grpo_steps(self.track.name):
            step = self.step
            # Previous input anchors are needed only by the synthetic seed producer.
            # PCA score/acceptance is independent of these GRPO batches.
            rows = self.track.batch(step, self.size, self.split)
            self.step += 1
            self.track.consumed(step, rows)
            Progress.log("grpo-loader", "feeding model training batch", track=self.track.name,
                         step=step+1, total_steps=grpo_steps(self.track.name), examples=len(rows))
            yield self.collate(self.to_verl(rows, step))

    def state_dict(self):
        return {"step": self.step, "track": self.track.name}

    def load_state_dict(self, state):
        if state["track"] != self.track.name:
            raise ValueError("Cannot restore a checkpoint from another track")
        self.step = int(state["step"])


# Code generation and model ownership live exclusively in custome.main.


class AsyncProducer:
    """Bounded async queue; concurrent verification overlaps next synth LLM call."""

    def __init__(self, track):
        if not track.stream:
            raise ValueError("Producer is valid for synthesized tracks only")
        self.track, self.store = track, PrivateDataset("generated", track.name)
        self.mapper = DataLoad()
        self.synth = None
        self._batch_refs = {}

    async def anchor(self, step):
        path = files.anchor(step, self.track.name)
        start = time.monotonic()
        while not path.exists():
            Progress.pulse("producer", f"anchor-{step}", "waiting for training batch anchor",
                           step=step, elapsed_s=round(time.monotonic()-start,1))
            await asyncio.sleep(0.25)
        return files.read(path)

    async def make_batch(self, step, anchor):
        """Asynchronous slots: a rejected candidate retries its EXACT seed.

        Each output slot owns one seed. The current actor is accessed through
        a generation lock (model reload/generate are not concurrent); Podman
        verifications for other slots can run in parallel via the asyncio queue.
        """
        target = env("RL_GRPO_BATCH", int)
        limit = env("RL_MAX_ATTEMPTS_PER_STEP", int)
        gen_limit = env("RL_SYNTH_BATCH_SIZE", int)
        verify_limit = env("RL_VERIFY_CONCURRENCY", int)
        if gen_limit < 1 or verify_limit < 1:
            raise ValueError("RL_SYNTH_BATCH_SIZE and RL_VERIFY_CONCURRENCY must both be >=1")
        queue = asyncio.Queue(maxsize=env("RL_SYNTH_QUEUE_SIZE", int))
        gate = asyncio.Lock()
        generation_slots = asyncio.Semaphore(gen_limit)
        verify_gate = asyncio.Lock()
        accepted, seen, errors = [], set(), []
        attempts = 0
        started = time.monotonic()
        Progress.log("producer", "starting candidate generation for batch", step=step,
                     batch_size=target, max_attempts=limit, concurrent_generation=gen_limit,
                     concurrent_verification=verify_limit)
        loop = asyncio.get_running_loop()

        async def save(row):
            return await asyncio.to_thread(self.store.append, [row])

        async def make_slot(slot):
            nonlocal attempts
            seed = anchor["problems"][slot % len(anchor["problems"])]
            while True:
                async with gate:
                    if limit and attempts >= limit:
                        return False
                    attempts += 1
                    attempt = attempts
                Progress.pulse("producer", f"generate-{step}", "generating candidate",
                               step=step, attempt=attempt, accepted=len(accepted), target=target,
                               seed_slot=slot, elapsed_s=round(time.monotonic()-started,1))
                # Runs alongside VERL updates and other seeds' verification.
                # A separate model copy is owned by each spawn worker.
                async with generation_slots:
                    snapshot = {}
                    try:
                        snapshot = await self.refresh_policy(step)
                        if hasattr(self.synth, "submit"):
                            # A persistent spawn worker owns/reuses its model copy.
                            job = self.synth.submit(snapshot, anchor, step, attempt, seed)
                            candidate = await asyncio.wrap_future(job)
                        else:
                            # Lightweight injectable fake for CPU-only tests.
                            candidate = await asyncio.to_thread(
                                self.synth.generate, anchor, step, attempt, seed
                            )
                    except Exception as exc:
                        Progress.log("producer", "generation request FAILED; retry same seed", step=step,
                                     attempt=attempt, seed_slot=slot, error=f"{type(exc).__name__}: {exc}")
                        await save(self.mapper.synth({
                            "id": uuid4().hex[:16], "source": "synth_gen", "step": step,
                            "attempt": attempt, "seed": seed, "seed_slot": slot,
                            "generator_model": snapshot.get("path", ""),
                            "policy_snapshot_step": snapshot.get("policy_step", -1),
                            "accepted": False, "status": "generation_error",
                            "failure_stage": "synth_gen_generation",
                            "failure_reason": str(exc),
                            "error_message": str(exc), "error_types": [type(exc).__name__],
                            "raw_source": {"seed": seed, "reference_step": anchor.get("reference_step", "")},
                        }).to_dict())
                        continue
                # Bind the generation result to the exact requested training
                # step/attempt/seed before normalizing or persisting its tests.
                mismatched = {name: {"expected": expected, "returned": candidate.get(name)}
                              for name, expected in (("step", step), ("attempt", attempt), ("seed", seed))
                              if candidate.get(name) is not None and candidate[name] != expected}
                if mismatched:
                    # Keep the entire invalid response for audit, but NEVER let
                    # it enter the training batch or masquerade as a valid step.
                    candidate["metadata"] = {**dict(candidate.get("metadata") or {}),
                                             "request_mismatches": mismatched}
                    candidate.update({"status": "data_mismatch", "accepted": False,
                                      "rejection_reason": "data_mismatch", "failure_stage": "data_mapping",
                                      "error_message": str(mismatched), "failure_reason": str(mismatched)})
                    Progress.log("producer", "rejected mismatched generation result", step=step,
                                 attempt=attempt, mismatches=mismatched)
                candidate.update({"step": step, "attempt": attempt, "seed": seed, "seed_slot": slot})
                candidate = self.mapper.synth(candidate).to_dict()
                if mismatched:
                    await save(candidate)
                    continue  # Retry the same seed after retaining full provenance.

                try:
                    Sample.from_dict(candidate).assert_prompt_integrity()
                except ValueError as error:
                    candidate.update({"passed": False, "accepted": False, "status": "prompt_leak",
                                      "rejection_reason": "prompt_leak", "failure_stage": "problem_generation",
                                      "failure_reason": str(error), "error_message": str(error)})
                    Progress.log("producer", "rejected answer-leaking problem; retrying same seed",
                                 step=step, attempt=attempt, reason=str(error))
                    await save(candidate)
                    continue

                if candidate["status"] == "generation_error" or not candidate["generated_code"] or not candidate["test_code"]:
                    # No way to earn a valid code reward without both code
                    # and runnable tests; keep the entire failed attempt locally.
                    reason = ("generation_error" if candidate["status"] == "generation_error" else
                              "empty_code" if not candidate["generated_code"] else "no_tests")
                    candidate.update({
                        "passed": False, "accepted": False,
                        "status": reason, "rejection_reason": reason,
                        "failure_stage": "synth_gen_generation" if reason == "generation_error" else "code_verification",
                        "failure_reason": candidate.get("synth_gen_error") or reason,
                        "error_message": candidate.get("error_message") or candidate.get("synth_gen_error") or reason,
                        "error_types": candidate.get("error_types") or ["NoTests" if reason == "no_tests" else "NoCode"],
                    })
                    Progress.log("producer", "candidate rejected before pytest; retry same seed",
                                 step=step, attempt=attempt, seed_slot=slot, reason=reason)
                    await save(self.mapper.synth(candidate).to_dict())
                    continue  # Retry this exact same seed.
                future = loop.create_future()
                await queue.put((candidate, future))
                if await future:
                    return True
                # Failure (test/PCA/duplicate) repeats this very same seed.

        async def verify():
            while True:
                item = await queue.get()
                if item is None:
                    queue.task_done()
                    return
                candidate, future = item
                accepted_now = False
                try:
                    result = await asyncio.to_thread(call_service, {
                        "source": "synth_gen", "record": False, "code": candidate["generated_code"],
                        "raw_output": candidate["raw_output"],
                        "extra_info": {
                            "question": candidate["question"], "seed": candidate["seed"],
                            "test_code": candidate.get("test_code"),
                            "batch_id": step,
                            # Transfer only scoring inputs: the producer retains
                            # the full raw generation and verifier histories.
                            "source_record": {
                                "source": "synth_gen",
                                "synth_gen_passed": candidate.get("synth_gen_passed"),
                                "synth_gen_failed_verifier": candidate.get("synth_gen_failed_verifier", ""),
                                "synth_gen_error": candidate.get("synth_gen_error", ""),
                            },
                        },
                    })
                    record = result["record"]
                    async with verify_gate:
                        duplicate = candidate["generated_code"] in seen
                        seen.add(candidate["generated_code"])
                        # The score service returns only verification fields;
                        # restore synth_gen provenance before the acceptance gate.
                        reason = self.track.rejection_reason({
                            **record, "source": "synth_gen",
                            "synth_gen_passed": candidate.get("synth_gen_passed"),
                        }, duplicate=duplicate)
                        accepted_now = not reason
                    detail = ""
                    if reason == "test_failed":
                        detail = record.get("error_message") or record.get("status") or "pytest did not pass"
                    elif reason == "synth_gen_verifier":
                        detail = (f"{candidate.get('synth_gen_failed_verifier', '')}: "
                                  f"{candidate.get('synth_gen_error', '')}").strip()
                    elif reason in ("distance", "pca_unavailable"):
                        ratios = record.get("distance_ratios") or {}
                        detail = ", ".join(
                            f"d{n}={ratios.get(f'd{n}', 'missing')} threshold={env(f'RL_THRESHOLD_D{n}') }"
                            for n in (1, 2, 3)
                        )
                    elif reason == "duplicate":
                        detail = "Identical code generated in another attempt of this batch"
                    # Keep full synth generation provenance alongside the
                    # new per-task PCA metadata (without silently overwriting).
                    source_meta = dict(candidate.get("metadata") or {})
                    candidate.update({key: record[key] for key in Sample.verification_fields if key in record})
                    candidate["metadata"] = {
                        **source_meta, **dict(record.get("metadata") or {}),
                    }
                    candidate.update({
                        "accepted": accepted_now, "rejection_reason": reason,
                        "reference_step": "prior_sft_task_pca",
                        "success_reason": (
                            "original synth_gen verification, independent Podman pytest, "
                            "prior SFT task PCA (1D/2D/3D) thresholds and deduplication passed"
                            if accepted_now else ""
                        ),
                        "failure_stage": (
                            "pca_filter" if reason in ("distance", "pca_unavailable") else
                            "deduplication" if reason == "duplicate" else
                            "synth_gen_verifier" if reason == "synth_gen_verifier" else
                            "code_verification" if reason == "test_failed" else ""
                        ),
                        "failure_reason": detail,
                    })
                    candidate = self.mapper.synth(candidate).to_dict()
                    durable_refs = await save(candidate)
                    if accepted_now:
                        # The GRPO loader reads these verified offsets from disk.
                        # No accepted code is passed to the trainer from memory.
                        if durable_refs is not None:
                            if len(durable_refs) != 1:
                                raise RuntimeError("Accepted sample was not persisted exactly once")
                            self._batch_refs.setdefault(step, []).extend(durable_refs)
                        accepted.append(candidate)
                        Progress.log("producer", "candidate ACCEPTED", step=step, attempt=candidate.get("attempt"),
                                     accepted=len(accepted), target=target,
                                     matched_tasks=(candidate.get("metadata") or {}).get("pca_matched_tasks", []))
                    else:
                        Progress.pulse("producer", f"reject-{step}", "candidate REJECTED; retrying seed",
                                       step=step, attempt=candidate.get("attempt"),
                                       reason=reason, accepted=len(accepted), target=target)
                except Exception as exc:
                    Progress.log("producer", "verification/score ERROR", step=step,
                                 error=f"{type(exc).__name__}: {exc}")
                    errors.append(str(exc))
                    candidate.update({
                        "accepted": False, "status": "generation_error",
                        "failure_stage": "verification", "failure_reason": str(exc),
                        "error_message": str(exc), "error_types": [type(exc).__name__],
                    })
                    await save(self.mapper.synth(candidate).to_dict())
                finally:
                    if not future.done():
                        future.set_result(accepted_now)
                    queue.task_done()

        workers = [
            asyncio.create_task(verify())
            for _ in range(verify_limit)
        ]
        slot_tasks = [asyncio.create_task(make_slot(slot)) for slot in range(target)]
        try:
            await asyncio.gather(*slot_tasks)
            await queue.join()
        finally:
            for task in slot_tasks:
                task.cancel()
            await asyncio.gather(*slot_tasks, return_exceptions=True)
            for _ in workers:
                await queue.put(None)
            await asyncio.gather(*workers)
        Progress.log("producer", "batch generation finished", step=step,
                     accepted=len(accepted), target=target, attempts=attempts,
                     elapsed_s=round(time.monotonic()-started,1))
        if len(accepted) != target:
            raise RuntimeError(
                f"Step {step}: accepted {len(accepted)}/{target}, "
                f"attempted {attempts}/{limit or 'unlimited'}; {errors[-3:]}"
            )
        return accepted

    async def refresh_policy(self, step):
        """Bound policy lag while consuming the latest *same actor* checkpoint.

        The producer can use a recently completed actor snapshot and continue
        generating while the actor optimizes the current batch. An exactly
        instantaneous copy of in-flight FSDP parameters is not assumed.
        """
        max_lag = env("RL_SYNTH_MAX_POLICY_LAG_STEPS", int)
        if max_lag < 1:
            raise ValueError("RL_SYNTH_MAX_POLICY_LAG_STEPS must be >=1")
        pointer = files.track(self.track.name) / "policy_latest.json"
        failure = files.track(self.track.name) / "policy_watcher_error.txt"
        end = time.monotonic() + env("RL_POLICY_WAIT_TIMEOUT", float)
        while True:
            Progress.pulse("producer", f"policy-{step}", "waiting for fresh actor snapshot",
                           step=step, allowed_lag=max_lag,
                           elapsed_s=round(time.monotonic()-(end-env("RL_POLICY_WAIT_TIMEOUT",float)),1))
            if failure.exists():
                raise RuntimeError(failure.read_text(encoding="utf-8"))
            snapshot = files.read(pointer, {})
            if snapshot and int(snapshot["policy_step"]) >= step - max_lag:
                return snapshot
            if time.monotonic() > end:
                raise TimeoutError(f"No sufficiently recent GRPO actor snapshot for step {step}")
            await asyncio.sleep(.5)

    async def run(self):
        from custome.main import CodeGenerationPool
        self.synth = CodeGenerationPool()
        try:
            Progress.log("producer", "async producer online", track=self.track.name,
                         total_steps=grpo_steps(self.track.name))
            for step in range(grpo_steps(self.track.name)):
                if files.ready(step, self.track.name).exists():
                    continue
                Progress.log("producer", "waiting for previous training batch seed anchor", step=step,
                             total_steps=grpo_steps(self.track.name))
                anchor = await self.anchor(step)
                Progress.log("producer", "seed anchor ready", step=step,
                             seed_problems=len(anchor["problems"]))
                if not anchor.get("problems"):
                    raise ValueError("Missing seed questions in previous training batch")
                batch = await self.make_batch(step, anchor)
                refs = self._batch_refs.pop(step, [])
                if len(refs) != len(batch):
                    raise RuntimeError(f"Step {step}: {len(batch)} accepted but only {len(refs)} persisted references")
                # Fail before publishing ready if JSONL is missing/corrupted.
                self.store.load_accepted(refs, step)
                files.atomic(files.ready(step, self.track.name), {
                    "format": "accepted_jsonl_refs_v1", "step": step, "records": refs,
                })
                Progress.log("producer", "published saved-JSONL training batch", step=step,
                             accepted=len(refs), jsonl=str(self.store.accepted_path))
        finally:
            Progress.log("producer", "closing async synthesis workers", track=self.track.name)
            # All worker model copies must finish before snapshots can be removed.
            await asyncio.to_thread(self.synth.close)
            (files.track(self.track.name) / ".producer_finished").touch()


def bootstrap_sft_anchor(name="ours"):
    folder = files.output / "sft_final"
    n = env("RL_GPUS_PER_NODE", int) * env("RL_NNODES", int)
    rows = []
    for rank in range(n):
        path = folder / f"rank_{rank}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing final code-SFT rank batch: {path}")
        rows += files.read(path, [])
    samples = [Sample.from_dict(row) for row in rows]
    prompts = [row.question for row in samples if row.code]
    if not prompts:
        raise ValueError("Final Code SFT batch has no usable problem seeds")
    files.atomic(files.anchor(0, name), {
        "step": 0, "reference_step": "code_sft_last_step", "problems": prompts,
        "source": "last_code_sft_batch",
    })


def produce():
    track = CodeTrack.make(env("RL_TRACK"))
    try:
        asyncio.run(AsyncProducer(track).run())
    except BaseException as error:
        for step in range(grpo_steps(track.name)):
            if not files.ready(step, track.name).exists():
                path = files.track(track.name) / f"producer_error_{step:06d}.txt"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(str(error), encoding="utf-8")
        raise


if __name__ == "__main__":
    produce()
