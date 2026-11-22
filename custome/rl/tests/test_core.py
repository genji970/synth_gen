"""CPU-only contract tests; never invoke remote Podman, GPUs or HF network."""
import asyncio
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from custome.rl.core import PrivateDataset, files
from custome.rl.data_load import DataLoad, Sample
from custome.rl.score import BatchPCA, TaskPCABank
from custome.rl.tracks import CodeTrack, AsyncProducer, StreamingCodeLoader


@pytest.fixture
def settings(monkeypatch, tmp_path):
    values = {
        "RL_OUT": str(tmp_path), "RL_CKPT": str(tmp_path / "checkpoints"),
        "RL_ROOT": str(tmp_path / "project-rl"), "RL_HF_DATA_REPO": "", "RL_RUN_ID": "manual",
        "RL_TRACK": "ours", "RL_GRPO_BATCH": "4", "RL_GRPO_STEPS_OURS": "3", "RL_GRPO_STEPS_BASELINE": "3",
        "RL_SYNTH_BATCH_SIZE": "2", "RL_POLICY_KEEP_SNAPSHOTS": "2", "RL_POLICY_POLL_SECS": "0.01",
        "RL_MAX_ATTEMPTS_PER_STEP": "10", "RL_VERIFY_CONCURRENCY": "2", "RL_SYNTH_QUEUE_SIZE": "2",
        "RL_SYNTH_MAX_POLICY_LAG_STEPS": "2", "RL_POLICY_WAIT_TIMEOUT": "3",
        "RL_PCA_REFERENCE_FRACTION": "0.1", "RL_PCA_MIN_REFERENCE_SAMPLES": "4",
        "RL_PCA_MATCH_POLICY": "any", "RL_PCA_INCLUDE_CURRENT_TASK": "0",
        "RL_GRPO_TASK": "code", "RL_SFT_TASKS": "math social code",
        "RL_FEATURE_MODEL": str(tmp_path / "encoder"), "RL_NNODES": "1",
        "RL_GPUS_PER_NODE": "2", "RL_RANDOM_SEED": "42",
        "RL_REQUIRED_PASS": "1", "RL_REQUIRE_SYNTH_VERIFIER": "1", "RL_UPLOAD_EVERY": "2", "RL_HF_TOKEN": "fake-test-token",
        "RL_HF_OWNER": "mock", "RL_HF_DATA_PREFIX": "test-private",
        **{f"RL_THRESHOLD_D{n}": "1.5" for n in (1, 2, 3)},
    }
    for key, val in values.items():
        monkeypatch.setenv(key, val)
    files.atomic(files.track() / "policy_latest.json", {"path": "/fake/sft", "policy_step": -1})
    return tmp_path


def test_track_inheritance():
    assert not CodeTrack.make("baseline").stream
    assert CodeTrack.make("ours").stream
    assert CodeTrack.make("ours").filter_distance
    assert set(CodeTrack.registry) == {"baseline", "ours"}
    with pytest.raises(ValueError):
        CodeTrack.make("unknown")


def test_pca_projects_without_refitting():
    vectors = {str(i): np.array([float(i), i * i, i**3, (-1)**i]) for i in range(4)}
    pca = BatchPCA(tuple(vectors), vectors.__getitem__)
    for key in vectors:
        scores, ratios, coords = pca.score(key, vectors.__getitem__)
        assert all(scores[f"d{n}"] == pytest.approx(1) for n in (1, 2, 3))
        assert all(ratios[f"d{n}"] == pytest.approx(0) for n in (1, 2, 3))
        assert len(coords[3][0]) == 4


def _publish_saved_training(step, rows):
    """Emulate the producer: commit validated JSONL before publishing refs."""
    normalized = []
    for r in rows:
        row = Sample.from_dict(r).to_dict()
        row.update(step=step, passed=True, accepted=True, synth_gen_passed=True,
                   test_code=row.get("test_code") or "def test_ok(): assert True")
        row["metadata"] = {**row.get("metadata", {}),
                           "pca_matched": True, "pca_by_task": {"math": {"matched": True}}}
        normalized.append(row)
    refs = PrivateDataset("generated", "ours").append(normalized)
    files.atomic(files.ready(step), {"format": "accepted_jsonl_refs_v1", "step": step, "records": refs})
    return refs


def test_stream_publishes_consumed_batch(settings):
    import pandas as pd
    monkeypatch_rows = [DataLoad().synth(dict(
        id=i, problem=f"prompt{i}", code=f"answer{i}",
        test_code="def test_x(): pass", passed=True,
        accepted=True, source="synth_gen",
    )).to_dict() for i in range(4)]
    _publish_saved_training(0, monkeypatch_rows)
    files.atomic(files.anchor(0), {"codes": ["before0", "before1", "before2", "before3"],
                                  "problems": ["before"], "reference_step": "code_sft_last_step"})
    loader = StreamingCodeLoader(_data_config=None, _split="train", _batch_size=4,
                                 _tokenizer=None, _processor=None, _collate_fn=lambda rows: rows)
    rows = next(iter(loader))
    assert len(rows) == 4
    assert files.read(files.anchor(1))["problems"] == [r["question"] for r in monkeypatch_rows]
    assert "codes" not in files.read(files.anchor(1))
    assert all(r["extra_info"]["batch_id"] == 0 for r in rows)
    assert "reference_batch" not in rows[0]["extra_info"]
    assert rows[0]["extra_info"]["question"] == "prompt0"


def test_next_step_publishes_seed_anchor_but_not_pca_references(settings):
    batch0 = [DataLoad().synth(dict(id=i, problem=f"p{i}", code=f"step0-{i}")).to_dict()
              for i in range(4)]
    batch1 = [DataLoad().synth(dict(id=i, problem=f"q{i}", code=f"step1-{i}")).to_dict()
              for i in range(4)]
    files.atomic(files.anchor(0), {"codes": [f"sft-{i}" for i in range(4)], "problems": ["base"]})
    _publish_saved_training(0, batch0)
    _publish_saved_training(1, batch1)
    loader = StreamingCodeLoader(_data_config=None, _split="train", _batch_size=4,
                                 _tokenizer=None, _processor=None, _collate_fn=lambda rows: rows)
    stream = iter(loader)
    first, second = next(stream), next(stream)
    assert "reference_batch" not in first[0]["extra_info"]
    assert "reference_batch" not in second[0]["extra_info"]
    assert files.read(files.anchor(1))["problems"] == [f"p{i}" for i in range(4)]
    assert files.read(files.anchor(2))["problems"] == [f"q{i}" for i in range(4)]
    assert "codes" not in files.read(files.anchor(1))
    assert "codes" not in files.read(files.anchor(2))


def test_async_producer_filters_by_class_policy(monkeypatch, settings):
    from custome.rl import tracks
    output = []
    seeds = []
    monkeypatch.setattr(PrivateDataset, "append", lambda self, rows: output.extend(rows))

    class FakeSynth:
        def generate(self, anchor, step, attempt, seed):
            seeds.append(seed)
            return {"id": str(attempt), "problem": "p", "code": f"code{attempt}",
                    "raw_output": f"code{attempt}", "seed": seed, "test_code": "test",
                    "synth_gen_passed": True}

    def fake_service(payload):
        attempt = int(payload["code"].removeprefix("code"))
        near = attempt >= 3
        return {"record": {"passed": True,
                            "distance_ratios": {f"d{n}": 0.1 if near else 4.0 for n in (1, 2, 3)},
                            "reward_components": {f"d{n}": 0.9 for n in (1, 2, 3)},
                            "reward_total": 0.9,
                            "metadata": {"pca_by_task": {"math": {"matched": near}}, "pca_matched": near}}}

    monkeypatch.setattr(tracks, "call_service", fake_service)
    worker = AsyncProducer(CodeTrack.make("ours"))
    worker.synth = FakeSynth()
    rows = asyncio.run(worker.make_batch(0, {
        "codes": ["a", "b", "c", "d"], "problems": ["seedA", "seedB"], "reference_step": 0,
    }))
    assert len(rows) == 4
    assert all(int(r["generated_code"].removeprefix("code")) >= 3 for r in rows)
    assert any(r.get("rejection_reason") == "distance" for r in output)
    assert seeds.count("seedA") >= 2  # Rejected attempts use their original slot seed.
    assert all(r["seed"] == ["seedA", "seedB"][r["seed_slot"] % 2] for r in rows)
    assert {r["seed"] for r in rows} == {"seedA", "seedB"}


def test_no_independent_synth_model_env(monkeypatch, settings):
    from custome.main import VerifiedCodeGenerator
    import inspect
    text = inspect.getsource(VerifiedCodeGenerator)
    assert "RL_SYNTH_MODEL" not in text
    assert "current_grpo_actor_snapshot" in text


def test_actor_snapshot_watcher_publishes_updated_model(monkeypatch, settings):
    from custome.rl import run
    folder = files.checkpoint / "grpo_ours"
    ckpt = folder / "global_step_1"
    (ckpt / "actor").mkdir(parents=True)
    (folder / ".training_finished").touch()
    (files.track("ours") / ".producer_finished").touch()

    monkeypatch.setenv("RL_POLICY_SNAPSHOT_EVERY", "1")
    monkeypatch.setattr(run.PrivateModel, "completed", staticmethod(lambda _: [ckpt]))

    def fake_merge(path, target, cpu_only=False):
        assert path == ckpt / "actor"
        assert cpu_only
        Path(target).mkdir(parents=True, exist_ok=True)
        (Path(target) / "config.json").write_text("{}")

    monkeypatch.setattr(run.Trainer, "merge", staticmethod(fake_merge))
    run.policy_watch("ours", "/fake/code_sft_checkpoint")
    pointer = files.read(files.track("ours") / "policy_latest.json")
    assert pointer["policy_step"] == 0
    assert (Path(pointer["path"]) / "config.json").exists()


def test_hf_dataset_private_check_and_incremental_upload(monkeypatch, settings):
    class FakeRepo:
        private = True

    class FakeAPI:
        def __init__(self, *args, **kwargs): pass
        def whoami(self): return {"name": "mock"}
        def create_repo(self, **kwargs): assert kwargs["private"] is True
        def repo_info(self, **kwargs): return FakeRepo()
        def upload_file(self, **kwargs):
            assert kwargs["repo_type"] == "dataset"
            assert len(Path(kwargs["path_or_fileobj"]).read_text().splitlines()) == 2

    pkg = types.ModuleType("huggingface_hub")
    pkg.HfApi = FakeAPI
    monkeypatch.setitem(sys.modules, "huggingface_hub", pkg)
    dataset = PrivateDataset("generated", "ours")
    dataset.append([
        DataLoad().synth({"id": "a", "problem": "A?", "code": "pass", "passed": False}).to_dict(),
        DataLoad().synth({"id": "b", "problem": "B?", "code": "pass", "passed": True}).to_dict(),
    ])
    assert dataset.upload() == 2
    assert dataset.upload() == 0
    assert files.read(dataset.path.with_suffix(".upload_state.json"))["uploaded"] == 2


def test_auto_create_one_private_dataset_and_reuse_across_runs(monkeypatch, settings):
    """Blank RL_HF_DATA_REPO creates exactly once, persists, and reuses privately."""
    created, uploaded = [], []
    class FakeAPI:
        def __init__(self, *args, **kwargs): pass
        def whoami(self): return {"name": "testowner"}
        def create_repo(self, **kw):
            assert kw["private"] is True and kw["repo_type"] == "dataset"
            assert kw["exist_ok"] is False
            created.append(kw["repo_id"])
        def repo_info(self, **kw):
            assert kw["repo_id"] in created
            return types.SimpleNamespace(private=True)
        def upload_file(self, **kw):
            uploaded.append((kw["repo_id"], kw["path_in_repo"]))

    hub = types.ModuleType("huggingface_hub")
    hub.HfApi = FakeAPI
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setenv("RL_ROOT", str(settings / "project-rl"))
    monkeypatch.setenv("RL_HF_DATA_REPO", "")
    monkeypatch.setenv("RL_HF_OWNER", "")
    monkeypatch.setenv("RL_RUN_ID", "run_01")

    store = PrivateDataset("generated", "ours")
    store.append([{"source": "synth_gen", "problem": f"q{i}", "code": "pass"}
                  for i in range(2)])
    assert store.upload() == 2
    assert len(created) == 1
    assert created[0].startswith("testowner/test-private-")
    saved = files.read(settings / "project-rl" / "hf_data_repository.json")
    assert saved == {"repo": created[0]}

    monkeypatch.setenv("RL_OUT", str(settings / "second-run"))
    monkeypatch.setenv("RL_RUN_ID", "run_02")
    next_store = PrivateDataset("generated", "ours")
    next_store.append([{"source": "synth_gen", "problem": f"r{i}", "code": "pass"}
                       for i in range(2)])
    assert next_store.upload() == 2
    assert len(created) == 1
    assert all(repo == created[0] for repo, _ in uploaded)
    assert any("runs/run_01/" in path for _, path in uploaded)
    assert any("runs/run_02/" in path for _, path in uploaded)


def test_public_repository_blocks_upload(monkeypatch, settings):
    class FakeRepo:
        private = False

    class FakeAPI:
        def __init__(self, *args, **kwargs): pass
        def whoami(self): return {"name": "mock"}
        def create_repo(self, **kwargs): assert kwargs["private"] is True
        def repo_info(self, **kwargs): return FakeRepo()

    pkg = types.ModuleType("huggingface_hub")
    pkg.HfApi = FakeAPI
    monkeypatch.setitem(sys.modules, "huggingface_hub", pkg)
    with pytest.raises(RuntimeError, match="PUBLIC"):
        PrivateDataset("generated", "ours").repository()


def test_mapped_mcqa_has_choices_and_single_canonical_prompt():
    record = DataLoad().social({
        "context": "A friend is upset.", "question": "What is kind?",
        "answerA": "Listen", "answerB": "Ignore", "answerC": "Laugh", "label": "1",
    })
    assert record.kind == "mcqa" and record.answer == "A"
    assert record.choices == {"A": "Listen", "B": "Ignore", "C": "Laugh"}
    assert record.to_sft()["messages"][0]["content"] == record.prompt
    assert record.to_sft()["messages"][1]["content"] == "A"
    assert record.raw_source["context"] == "A friend is upset."


def test_mapped_math_preserves_reference_and_final_answer():
    record = DataLoad().math({"problem": "Compute 2+3.", "solution": "2+3=\\boxed{5}"})
    assert record.kind == "short_answer" and record.answer == "5"
    assert record.reference == "5"  # Gold label; NO derivation
    assert record.sft_solution == "2+3=\\boxed{5}"
    assert record.training_answer == record.sft_solution  # Assistant SFT target only
    assert "2+3=" not in record.prompt


def test_mapped_apps_preserves_all_references_and_raw_tests():
    original = {"question": "Add one.", "solutions": '["def f(x): return x+1", "def f(x): return 1+x"]',
                "input_output": '{"fn_name":"f","inputs":[[1]],"outputs":[2]}', "difficulty": "interview"}
    record = DataLoad().code(original)
    assert len(record.references) == 2
    assert record.passed is None  # Reference solutions are not automatically verified
    assert record.raw_source["difficulty"] == "interview"
    assert "fn_name" in record.apps_io
    assert Sample.from_dict(record.to_storage()).to_dict() == record.to_dict()


def test_synth_failures_store_detailed_reason_and_never_train(settings):
    record = DataLoad().synth({
        "id": "try-1", "problem": "Return a sum.", "code": "return undefined_name",
        "seed": "same-seed", "attempt": 7, "policy_snapshot_step": 8,
        "generator_model": "/checkpoints/actor-8", "raw_output": "```python\nreturn undefined_name\n```",
        "test_code": "def test_sum(): assert f(1)==2", "passed": False,
        "status": "failed", "synth_gen_passed": False,
        "synth_gen_failed_verifier": "MypyVerifier",
        "synth_gen_error": 'Name "undefined_name" is not defined',
        "error_types": ["NameError"], "error_message": "NameError: undefined_name",
        "pytest_summary": {"failed": 1}, "rejection_reason": "test_failed",
        "accepted": False,
    })
    assert record.question == "Return a sum."
    assert record.generated_code == "return undefined_name"
    assert record.failure_reason == "test_failed"
    assert record.synth_gen_failed_verifier == "MypyVerifier"
    assert record.pytest_summary["failed"] == 1
    assert record.accepted is False and not CodeTrack.make("ours").accepts(record.to_dict())
    assert Sample.from_dict(record.to_storage()).to_dict() == record.to_dict()


def test_generated_training_prompt_reused_for_grpo():
    record = DataLoad().synth({
        "id": "ok-1", "problem": "Return twice x.",
        "code": "def twice(x): return 2*x", "passed": True,
        "accepted": True,
    })
    grpo = record.to_verl(3)
    assert grpo["raw_prompt"][0]["content"] == record.prompt
    assert grpo["extra_info"]["source_record"]["question"] == record.question
    assert "reference_batch" not in grpo["extra_info"]
    assert record.to_sft()["messages"][0]["content"] == record.prompt


def test_private_jsonl_records_have_full_schema(monkeypatch, settings):
    store = PrivateDataset("generated", "ours")
    store.append([{
        "source": "synth_gen", "id": "fail", "problem": "Sorting exercise",
        "code": "def sort(x): return x", "passed": False,
        "status": "failed", "error_types": ["AssertionError"],
        "error_message": "AssertionError: wrong order", "test_code": "def test_order(): pass",
        "seed": "sorting seed", "accepted": False, "rejection_reason": "test_failed",
    }])
    saved = json.loads(store.path.read_text().splitlines()[0])
    assert saved["schema_version"] == 1
    assert saved["task"] == "code" and saved["kind"] == "code"
    assert saved["question"] == "Sorting exercise"
    assert saved["generated_code"] == "def sort(x): return x"
    assert saved["error_message"] == "AssertionError: wrong order"
    assert saved["accepted"] is False


def test_mp_process_count_and_spawn_config(monkeypatch):
    from custome import main as generator_api
    observed = {}

    class Executor:
        def __init__(self, max_workers, mp_context):
            observed["processes"] = max_workers
            observed["start_method"] = mp_context.get_start_method()
        def submit(self, fn, *args):
            observed["args"] = args
            from concurrent.futures import Future
            future = Future()
            future.set_result({"seed": args[-1], "code": "pass"})
            return future
        def shutdown(self, **kwargs):
            observed["shutdown"] = kwargs

    monkeypatch.setattr(generator_api, "ProcessPoolExecutor", Executor)
    monkeypatch.setattr(generator_api, "_visible_gpu_count", lambda: 2)
    monkeypatch.setenv("RL_SYNTH_GPU_MEMORY_FRACTION", "0.90")
    monkeypatch.setenv("RL_SYNTH_BATCH_SIZE", "4")
    monkeypatch.setenv("RL_SYNTH_PROCESSES", "3")
    monkeypatch.setenv("RL_SYNTH_START_METHOD", "spawn")
    with generator_api.CodeGenerationPool() as pool:
        assert pool.submit({"path": "/model", "policy_step": 1}, {}, 0, 4, "seedA").result()["seed"] == "seedA"
    assert observed["processes"] == 3
    assert observed["start_method"] == "spawn"
    assert observed["args"][0]["path"] == "/model"
    assert observed["shutdown"]["cancel_futures"] is True


def test_mp_invalid_config(monkeypatch):
    from custome.main import CodeGenerationPool
    monkeypatch.setenv("RL_SYNTH_PROCESSES", "0")
    with pytest.raises(ValueError, match=">= 1"):
        CodeGenerationPool()
    monkeypatch.setenv("RL_SYNTH_PROCESSES", "1")
    monkeypatch.setenv("RL_SYNTH_START_METHOD", "fork")
    with pytest.raises(ValueError, match="spawn"):
        CodeGenerationPool()


def test_gpu_round_robin_and_per_process_budget():
    from custome.main import _worker_cuda_plan
    for n_gpus, expected in (
        (1, ["cuda:0"] * 4),
        (2, ["cuda:0", "cuda:1", "cuda:0", "cuda:1"]),
        (4, [f"cuda:{i}" for i in range(4)]),
    ):
        plans = [_worker_cuda_plan(worker_ordinal=i, gpu_count=n_gpus,
                                  processes=4, budget=0.90) for i in range(4)]
        assert [plan[0] for plan in plans] == expected
        for device, fraction, slots in plans:
            assert fraction * slots == pytest.approx(0.90)
            assert slots == (4 if n_gpus == 1 else 2 if n_gpus == 2 else 1)
    # Non-divisible worker count: GPU0 gets 2 workers; GPUs1,2 get 1 each.
    plans = [_worker_cuda_plan(worker_ordinal=i, gpu_count=3,
                              processes=4, budget=0.90) for i in range(4)]
    assert [p[0] for p in plans] == ["cuda:0", "cuda:1", "cuda:2", "cuda:0"]
    assert [p[1] for p in plans] == pytest.approx([0.45, 0.90, 0.90, 0.45])


def test_gpu_generation_fails_without_cuda(monkeypatch):
    from custome import main as generator_api
    monkeypatch.setenv("RL_SYNTH_PROCESSES", "4")
    monkeypatch.setenv("RL_SYNTH_BATCH_SIZE", "4")
    monkeypatch.setenv("RL_SYNTH_GPU_MEMORY_FRACTION", "0.90")
    monkeypatch.setenv("RL_SYNTH_START_METHOD", "spawn")
    monkeypatch.setattr(generator_api, "_visible_gpu_count", lambda: 0)
    with pytest.raises(RuntimeError, match="No CUDA GPU visible"):
        generator_api.CodeGenerationPool()
    with pytest.raises(RuntimeError, match="No CUDA GPU visible"):
        generator_api._worker_cuda_plan(worker_ordinal=0)


def test_gpu_memory_budget_is_applied_before_model_load(monkeypatch):
    from custome import main as generator_api
    import torch
    seen = {}
    monkeypatch.setattr(generator_api, "_worker_cuda_plan",
                        lambda: ("cuda:0", 0.225, 4))
    monkeypatch.setattr(torch.cuda, "set_device", lambda i: seen.setdefault("device", i))
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction",
                        lambda fraction, device: seen.update(fraction=fraction, limit_device=device))
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda i: types.SimpleNamespace(total_memory=24 * 1024**3))
    monkeypatch.setattr(generator_api.VerifiedCodeGenerator, "_make_generator",
                        lambda self: types.SimpleNamespace(verifiers=[1, 2]))
    worker = generator_api.VerifiedCodeGenerator()
    assert worker.device == "cuda:0"
    assert seen == {"device": 0, "fraction": 0.225, "limit_device": 0}


def test_generation_uses_tokenizer_attention_mask():
    # Contract check for the inner LangChain adapter without invoking a real GPU.
    import ast
    from custome import main as generator_api
    from inspect import getsource
    tree = ast.parse(getsource(generator_api))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    generate_calls = [node for node in calls
                      if isinstance(node.func, ast.Attribute) and node.func.attr == "generate"]
    assert any(any(k.arg == "attention_mask" for k in node.keywords)
               for node in generate_calls)
    tokenizer_calls = [node for node in calls
                       if isinstance(node.func, ast.Attribute)
                       and node.func.attr == "apply_chat_template"]
    assert any(any(k.arg == "return_dict" and isinstance(k.value, ast.Constant)
                       and k.value.value is True for k in node.keywords)
               for node in tokenizer_calls)


def test_mp_producer_submits_same_seed_after_rejection(monkeypatch, settings):
    from custome.rl import tracks
    from concurrent.futures import Future
    monkeypatch.setenv("RL_SYNTH_BATCH_SIZE", "2")
    monkeypatch.setenv("RL_GRPO_BATCH", "4")
    seen = []
    saved = []
    monkeypatch.setattr(PrivateDataset, "append", lambda self, rows: saved.extend(rows))

    class FakePool:
        def submit(self, snapshot, anchor, step, attempt, seed):
            seen.append((attempt, seed, snapshot["policy_step"]))
            fut = Future()
            fut.set_result({"id": str(attempt), "problem": "new exercise", "code": f"code{attempt}",
                            "seed": seed, "synth_gen_passed": True, "test_code": "def test_x(): assert True", "raw_source": {"origin": seed}})
            return fut

    def scoring(payload):
        attempt = int(payload["code"][4:])
        close = attempt > 4
        return {"record": {"passed": True,
                "distance_ratios": {f"d{n}": .2 if close else 9.0 for n in (1, 2, 3)},
                "reward_components": {}, "reward_total": 0.4,
                "metadata": {"pca_by_task": {"social": {"matched": close}}, "pca_matched": close}}}

    monkeypatch.setattr(tracks, "call_service", scoring)
    prod = AsyncProducer(CodeTrack.make("ours"))
    prod.synth = FakePool()
    rows = asyncio.run(prod.make_batch(0, {"codes": ["a", "b", "c", "d"],
                      "problems": ["one", "two"], "reference_step": "code_sft_last_step"}))
    assert len(rows) == 4
    assert any(x["rejection_reason"] == "distance" for x in saved)
    assert len(seen) >= 8
    assert {x[2] for x in seen} == {-1}
    assert all(r["seed"] == ["one", "two"][r["seed_slot"] % 2] for r in rows)
    assert all("origin" in r["raw_source"] for r in rows)


def test_generator_preserves_original_chat_and_verifier_failure(monkeypatch):
    from custome.main import VerifiedCodeGenerator
    from types import SimpleNamespace as NS
    obj = VerifiedCodeGenerator.__new__(VerifiedCodeGenerator)
    obj.path = '/models/current_actor'
    obj.policy_step = 7
    obj.load_actor = lambda snapshot: None
    obj._web_context = lambda seed: ''
    record = NS(
        problem='Implement the function.', code='def f(x): return x', test_code='def test_f(): assert f(1)==2',
        is_correct=False, failed_verifier_name='PythonTestBasedVerifier',
        failed_verifier_error_message='AssertionError: expected 2',
        response=NS(content='raw code response'),
        prompt=NS(messages=[NS(content='prompt')]),
        chat=NS(messages=[NS(type='human', content='original full prompt'),
                          NS(type='ai', content='candidate code')]),
        verifications=None,
    )
    obj.generator = NS(generate_training_data=lambda seed: [record])
    result = obj.generate({'path': obj.path, 'policy_step': 7}, {}, 4, 11, 'same-seed')
    assert result['policy_snapshot_step'] == 7
    assert result['synth_gen_failed_verifier'] == 'PythonTestBasedVerifier'
    assert result['synth_gen_error'] == 'AssertionError: expected 2'
    assert result['raw_source']['generation_chat'][0]['content'] == 'original full prompt'
    assert result['seed'] == 'same-seed'
    normalized = DataLoad().synth(result)
    assert normalized.synth_gen_passed is False
    assert normalized.test_code == 'def test_f(): assert f(1)==2'
    assert normalized.raw_source['original_seed'] == 'same-seed'


def test_synth_generation_history_and_full_test_trace_are_canonical():
    output = DataLoad().synth({
        "id": "attempt-12", "seed": "same seed", "problem": "Add one.",
        "code": "def f(x): return x+1", "test_code": "def test_f(): assert f(2)==3",
        "synth_gen_passed": True, "synth_verification_success": True,
        "generation_history": [
            {"revision": 0, "code": "def f(x): return x", "is_correct": False},
            {"revision": 1, "code": "def f(x): return x+1", "is_correct": True},
        ],
        "verification_history": [
            {"event": "pytest_execution", "pytest_exit_code": 1,
             "executed_source": "old source and tests", "pytest_report": {"summary": {"failed": 1}}},
            {"event": "pytest_execution", "pytest_exit_code": 0,
             "executed_source": "new source and tests", "pytest_report": {"summary": {"passed": 1}}},
        ],
        "passed": True, "accepted": True, "pytest_report": {"tests": [{"outcome": "passed"}]},
        "pytest_stdout": "one passed", "pytest_stderr": "", "pytest_exit_code": 0,
        "raw_source": {"original_seed": "same seed", "generation_chat": [
            {"role": "human", "content": "generate a function"},
        ]},
    })
    saved = Sample.from_dict(output.to_storage())
    assert saved.generation_history[0]["code"] == "def f(x): return x"
    assert saved.verification_history[0]["pytest_report"]["summary"]["failed"] == 1
    assert saved.verification_history[1]["pytest_exit_code"] == 0
    assert saved.pytest_report["tests"][0]["outcome"] == "passed"
    assert saved.pytest_stdout == "one passed"
    assert saved.raw_source["generation_chat"][0]["content"] == "generate a function"
    assert saved.to_verl(0, ["a", "b", "c", "d"])["extra_info"]["source_record"]["source_id"]
    assert "verification_history" not in saved.reward_reference()  # avoid per-rollout duplication


def test_private_hf_uploads_every_attempt_and_accepted_separately(monkeypatch, settings):
    uploads = []
    created = []

    class FakeAPI:
        def __init__(self, *args, **kwargs): pass
        def whoami(self): return {"name": "mock"}
        def create_repo(self, **kwargs):
            assert kwargs["private"] is True
            created.append(kwargs["repo_id"])
        def repo_info(self, **kwargs): return types.SimpleNamespace(private=True)
        def upload_file(self, **kwargs):
            uploads.append((kwargs["path_in_repo"],
                            Path(kwargs["path_or_fileobj"]).read_text(encoding="utf-8")))
            assert kwargs["repo_id"] == created[0]
            assert kwargs["repo_type"] == "dataset"

    pkg = types.ModuleType("huggingface_hub")
    pkg.HfApi = FakeAPI
    monkeypatch.setitem(sys.modules, "huggingface_hub", pkg)
    store = PrivateDataset("generated", "ours")
    store.append([
        {"source": "synth_gen", "problem": "p1", "code": "a", "accepted": False,
         "passed": False, "error_message": "failed", "verification_history": [{"error": "assertion"}]},
        {"source": "synth_gen", "problem": "p2", "code": "b", "accepted": True,
         "passed": True, "synth_gen_passed": True,
         "metadata": {"pca_matched": True, "pca_by_task": {"math": {"matched": True}}},
         "test_code": "def test_b(): pass", "pytest_report": {"summary": {"passed": 1}}},
        {"source": "synth_gen", "problem": "p3", "code": "c", "accepted": True,
         "passed": True, "synth_gen_passed": True,
         "metadata": {"pca_matched": True, "pca_by_task": {"math": {"matched": True}}},
         "test_code": "def test_c(): pass"},
    ])
    assert len(store.path.read_text().splitlines()) == 3
    assert len(store.accepted_path.read_text().splitlines()) == 2
    assert store.upload() == 4  # two full chunks: all attempts + accepted training
    assert store.upload() == 0  # one all-attempts record waits below threshold
    assert store.upload(flush=True) == 1
    assert store.upload(flush=True) == 0
    assert store._uploaded_all()
    destinations = [name for name, _ in uploads]
    prefix = "runs/manual/tracks/ours/data/"
    assert prefix + "all_attempts/part-0000000.jsonl" in destinations
    assert prefix + "all_attempts/part-0000001.jsonl" in destinations
    assert prefix + "accepted_training/part-0000000.jsonl" in destinations
    all_lines = [json.loads(line) for name, contents in uploads
                 if name.startswith(prefix + "all_attempts/") for line in contents.splitlines()]
    accepted_lines = [json.loads(line) for name, contents in uploads
                      if name.startswith(prefix + "accepted_training/") for line in contents.splitlines()]
    assert len(all_lines) == 3 and len(accepted_lines) == 2
    assert {row["question"] for row in accepted_lines} == {"p2", "p3"}
    assert all(row["accepted"] is True for row in accepted_lines)
    assert all_lines[0]["verification_history"][0]["error"] == "assertion"
    assert all_lines[1]["pytest_report"]["summary"]["passed"] == 1


def test_podman_verification_keeps_full_pytest_report_and_output(monkeypatch):
    from custome.rl.score import CodeVerifier
    monkeypatch.setenv("RL_ALLOWED_PIP", "")
    module = types.ModuleType("custome.podman_generating")
    report = {
        "summary": {"passed": 0, "failed": 1},
        "tests": [{"outcome": "failed", "call": {"longrepr": "AssertionError: expected 42"}}],
    }
    module.run_pytest = lambda *a, **k: (report, "captured stdout", "captured stderr", 1)
    monkeypatch.setitem(sys.modules, "custome.podman_generating", module)
    result = CodeVerifier().verify("def f(): return 0", "def test_f(): assert f() == 42")
    assert result["passed"] is False
    assert result["pytest_exit_code"] == 1
    assert result["pytest_report"]["tests"][0]["call"]["longrepr"] == "AssertionError: expected 42"
    assert result["pytest_stdout"] == "captured stdout"
    assert result["pytest_stderr"] == "captured stderr"
    assert result["test_code"] == "def test_f(): assert f() == 42"
    no_tests = CodeVerifier().verify("print(1)")
    assert no_tests["status"] == "no_tests" and not no_tests["passed"]


def test_main_selects_verified_revision_but_keeps_all_prior_failures():
    from custome.main import VerifiedCodeGenerator
    from types import SimpleNamespace as NS
    obj = VerifiedCodeGenerator.__new__(VerifiedCodeGenerator)
    obj.path = "/actors/policy_8"
    obj.policy_step = 8
    obj.load_actor = lambda snapshot: None
    obj._web_context = lambda seed: ""

    def make_record(code, correct):
        return NS(
            problem="Return 7", code=code, test_code="def test_f(): assert f() == 7",
            is_correct=correct, verification_success=correct,
            failed_verifier_name="" if correct else "PythonTestBasedVerifier",
            failed_verifier_error_message="" if correct else "AssertionError",
            response=NS(content=code), prompt=NS(messages=[NS(content="Return 7")]),
            chat=NS(messages=[NS(type="human", content="Return 7")]), verifications=None,
        )

    failed = make_record("def f(): return 0", False)
    passed = make_record("def f(): return 7", True)
    obj.generator = NS(generate_training_data=lambda seed: [failed, passed])
    result = obj.generate({"path": obj.path, "policy_step": 8}, {}, 3, 5, "old-prompt")
    assert result["code"] == "def f(): return 7"
    assert result["synth_gen_passed"] is True
    assert len(result["generation_history"]) == 2
    assert result["generation_history"][0]["failed_error"] == "AssertionError"
    assert result["generation_history"][1]["is_correct"] is True
    assert DataLoad().synth(result).generation_history[0]["code"] == "def f(): return 0"


def test_generator_exception_returns_partial_audit_not_lost():
    from custome.main import VerifiedCodeGenerator
    from types import SimpleNamespace as NS
    obj = VerifiedCodeGenerator.__new__(VerifiedCodeGenerator)
    obj.path = "/actors/policy_3"
    obj.policy_step = 3
    obj.load_actor = lambda snapshot: None

    def crash(seed):
        obj.verification_history.extend([
            {"verifier": "PythonTestBasedVerifier", "passed": False,
             "code": "def f(): return 0", "test_code": "def test_f(): assert f() == 1",
             "error": "AssertionError"},
        ])
        raise RuntimeError("model test generation failed")

    obj.generator = NS(generate_training_data=crash)
    result = obj.generate({"path": obj.path, "policy_step": 3}, {}, 2, 9, "seed")
    assert result["status"] == "generation_error"
    assert result["synth_gen_passed"] is False
    assert result["verification_history"][0]["code"] == "def f(): return 0"
    assert "model test generation failed" in result["raw_source"]["exception_traceback"]
    assert not result["accepted"]
    assert DataLoad().synth(result).verification_history[0]["error"] == "AssertionError"


def test_missing_generated_tests_logged_and_never_sent_for_reward(monkeypatch, settings):
    from custome.rl import tracks
    monkeypatch.setenv("RL_MAX_ATTEMPTS_PER_STEP", "2")
    saved = []
    monkeypatch.setattr(PrivateDataset, "append", lambda self, rows: saved.extend(rows))
    monkeypatch.setattr(tracks, "call_service", lambda *a, **k: pytest.fail("No tests must not reach score"))

    class Fake:
        def generate(self, anchor, step, attempt, seed):
            return {"id": str(attempt), "problem": "Return 2", "seed": seed,
                    "code": "def f(): return 2", "test_code": "", "synth_gen_passed": False,
                    "synth_gen_failed_verifier": "PythonTestBasedVerifier",
                    "synth_gen_error": "Test generation unavailable"}

    prod = AsyncProducer(CodeTrack.make("ours"))
    prod.synth = Fake()
    with pytest.raises(RuntimeError, match="accepted 0/4"):
        asyncio.run(prod.make_batch(0, {
            "codes": ["a", "b", "c", "d"],
            "problems": ["seed a", "seed b"], "reference_step": "code_sft_last_step",
        }))
    assert len(saved) == 2
    assert {row["rejection_reason"] for row in saved} == {"no_tests"}
    assert all(row["accepted"] is False for row in saved)
    assert all(row["synth_gen_error"] == "Test generation unavailable" for row in saved)


def test_smoke_model_default_is_one_canonical_base(monkeypatch, settings):
    from custome.sample_test import CodeGenerationSmokeTest
    monkeypatch.setenv("RL_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    monkeypatch.setenv("RL_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    test = CodeGenerationSmokeTest(seed="seed", samples=1)
    assert test.model_snapshot() == {"path": "Qwen/Qwen2.5-1.5B-Instruct", "policy_step": -1}
    monkeypatch.setenv("RL_MODEL", "some/other-model")
    with pytest.raises(ValueError, match="RL_MODEL must equal RL_BASE_MODEL"):
        CodeGenerationSmokeTest(seed="seed", samples=1)


def test_smoke_latest_uses_completed_actor_snapshot(monkeypatch, settings):
    from custome.sample_test import CodeGenerationSmokeTest
    monkeypatch.setenv("RL_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    monkeypatch.setenv("RL_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    target = settings / "merged_actor"
    target.mkdir()
    (target / "config.json").write_text("{}")
    files.atomic(files.track("ours") / "policy_latest.json", {"path": str(target), "policy_step": 7})
    assert CodeGenerationSmokeTest(seed="seed", samples=1, use_latest=True).model_snapshot() == {
        "path": str(target), "policy_step": 7,
    }


def test_smoke_saves_exact_generated_code_test_and_report(monkeypatch, settings, capsys):
    from custome.sample_test import CodeGenerationSmokeTest
    from custome.rl.score import CodeVerifier
    monkeypatch.setenv("RL_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    monkeypatch.setenv("RL_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    smoke = CodeGenerationSmokeTest(seed="seed", samples=1)
    code = "def inc(n): return n + 1"
    tests = "def test_inc(): assert inc(1) == 2"
    result = CodeVerifier.result(passed=True, status="passed", summary={"passed": 1, "failed": 0},
                                 pytest_report={"summary": {"passed": 1}, "tests": [{"outcome": "passed"}]},
                                 pytest_exit_code=0, test_code=tests, pytest_stdout="1 passed")

    class FakeVerifier:
        def verify(self, actual_code, actual_tests):
            assert actual_code == code and actual_tests == tests
            return result

    record = smoke.verify_and_record({
        "id": "first", "seed": "seed", "problem": "Increment a number.", "code": code,
        "test_code": tests, "source": "synth_gen", "synth_gen_passed": True,
        "verification_history": [{"verifier": "MypyVerifier", "passed": True}],
    }, FakeVerifier(), PrivateDataset("generated", "ours"))
    smoke.print_result(record, 1)
    assert code in capsys.readouterr().out
    assert record["passed"] is True and record["accepted"] is False
    assert record["failure_stage"] == "pca_not_run"
    assert record["metadata"]["smoke_test"]["synth_vs_independent_pytest_agree"] is None
    assert record["verification_history"][-1]["code"] == code
    assert record["verification_history"][-1]["test_code"] == tests
    assert record["pytest_report"]["summary"]["passed"] == 1
    rows = [json.loads(x) for x in (files.track() / "generated_code.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["generated_code"] == code
    assert rows[0]["verification_history"][-1]["pytest_stdout"] == "1 passed"
    assert not (files.track() / "accepted_training.jsonl").exists()


def test_smoke_no_tests_is_recorded_not_pretended_success(monkeypatch, settings):
    from custome.sample_test import CodeGenerationSmokeTest
    from custome.rl.score import CodeVerifier
    monkeypatch.setenv("RL_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    monkeypatch.setenv("RL_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

    class FakeVerifier:
        def verify(self, code, tests):
            assert tests == ""
            return CodeVerifier.result(status="no_tests", error_types=["NoTests"],
                                       error_message="No executable generated tests", test_code="")

    result = CodeGenerationSmokeTest(seed="one", samples=1).verify_and_record({
        "id": "failed", "problem": "Return a number", "code": "def f(): return 1",
        "source": "synth_gen", "synth_gen_passed": False,
        "synth_gen_failed_verifier": "PythonTestBasedVerifier", "synth_gen_error": "missing test",
    }, FakeVerifier(), PrivateDataset("generated", "ours"))
    assert result["passed"] is False
    assert result["status"] == "no_tests"
    assert result["metadata"]["smoke_test"]["synth_vs_independent_pytest_agree"] is None
    assert result["synth_gen_failed_verifier"] == "PythonTestBasedVerifier"
    assert result["accepted"] is False
    assert result["verification_history"][-1]["passed"] is False
    assert not (files.track() / "accepted_training.jsonl").exists()


def test_per_track_steps_are_canonical(monkeypatch):
    from custome.rl.core import grpo_steps
    monkeypatch.setenv("RL_GRPO_STEPS_BASELINE", "4")
    monkeypatch.setenv("RL_GRPO_STEPS_OURS", "7")
    assert grpo_steps("baseline") == 4
    assert grpo_steps("ours") == 7
    monkeypatch.setenv("RL_GRPO_STEPS_OURS", "0")
    with pytest.raises(ValueError, match="must be positive"):
        grpo_steps("ours")


def test_sft_stage_steps_are_config_driven(monkeypatch):
    import inspect
    from custome.rl.run import Trainer, Pipeline
    source = inspect.getsource(Trainer.train)
    pipeline = inspect.getsource(Pipeline.run)
    assert "RL_SFT_STEPS_" in source
    assert "trainer.total_training_steps=" in source
    assert "RL_SFT_TASKS" in pipeline
    assert "RL_GRPO_TASK" in pipeline
    assert "RL_STEPS" not in source


def test_rl_config_does_not_define_training_defaults():
    from pathlib import Path
    script = Path(__file__).resolve().parents[1] / "rl_config.sh"
    content = script.read_text()
    assert "RL_STEPS=" not in content
    assert "RL_SFT_BATCH=" not in content
    assert "RL_GRPO_BATCH=" not in content
    assert 'RL_BASE_MODEL="Qwen/' not in content
    assert 'RL_MODEL="${RL_BASE_MODEL}"' in content


def test_one_model_setting_accepts_any_configured_base(monkeypatch, settings):
    from custome.rl.core import require_model_lineage
    from custome.sample_test import CodeGenerationSmokeTest
    for base in ("Qwen/Qwen2.5-1.5B-Instruct", "some-company/a-new-causal-chat-model"):
        monkeypatch.setenv("RL_BASE_MODEL", base)
        monkeypatch.setenv("RL_MODEL", base)
        assert require_model_lineage(base) == base
        assert CodeGenerationSmokeTest(seed="seed", samples=1).model_snapshot()["path"] == base
        with pytest.raises(ValueError, match="Model override"):
            require_model_lineage("different-company/second-model")
    checkpoint = Path(str(settings / "checkpoints")) / "private_loaded_code"
    assert require_model_lineage(str(checkpoint)) == str(checkpoint)


def test_config_has_only_one_model_default_and_disables_legacy_llm():
    config = Path(__file__).resolve().parents[2] / "config.sh"
    content = config.read_text()
    assert content.count('export RL_BASE_MODEL=') == 1
    assert 'unset LLM' in content
    assert not any(x in content for x in ('export RL_SYNTH_MODEL=', 'export RL_FEATURE_BASE_MODEL='))
    assert 'Qwen/Qwen2.5-1.5B-Instruct' in content


def test_per_task_traces_deduplicate_and_preserve_10pct_ratio(settings):
    from custome.rl.data_load import DataLoad
    monkey = DataLoad()
    for rank, ids in enumerate((("1", "2", "3"), ("3", "4", "5"))):
        files.atomic(monkey.pca_trace_path("math", rank), {
            "task": "math", "observed_ids": list(ids),
            "selected": [{"task": "math", "source_id": i, "pca_text": f"MATH {i}"} for i in ids],
            "fallback": [],
        })
    info = DataLoad.consolidate_pca_trace("math")
    assert info["observed_unique"] == 5
    assert info["selected_count"] == 5
    assert info["fraction"] == pytest.approx(0.1)
    assert len({x["source_id"] for x in info["records"]}) == 5


def test_pca_bank_scores_each_prior_task_with_single_embedded_candidate(monkeypatch, settings):
    monkeypatch.setenv("RL_PCA_MATCH_POLICY", "any")
    # Math has candidate-close coordinates; Social is intentionally distant.
    math = BatchPCA.from_vectors(np.array([
        [0,0,0,0], [1,0,0,0], [0,1,0,0], [0,0,1,0]], dtype=float))
    social = BatchPCA.from_vectors(np.array([
        [50,50,50,50], [51,50,50,50], [50,51,50,50], [50,50,51,50]], dtype=float))
    bank = TaskPCABank({"math": math, "social": social, "code": math},
                       ["math", "social", "code"], "encoder")
    times = []
    def embed(s):
        times.append(s)
        return np.array([0,0,0,0], dtype=float)
    rewards, ratios, _, metadata = bank.score("new code", embed)
    assert times == ["new code"]
    assert metadata["pca_matched"] is True
    assert metadata["pca_match_task"] == "math"
    assert set(metadata["pca_by_task"]) == {"math", "social"}
    assert metadata["pca_by_task"]["math"]["matched"] is True
    assert metadata["pca_by_task"]["social"]["matched"] is False
    assert all(ratios[f"d{n}"] <= 1.5 for n in (1,2,3))
    assert all(rewards[f"d{n}"] > 0 for n in (1,2,3))
    monkeypatch.setenv("RL_PCA_MATCH_POLICY", "all")
    assert bank.score("new code", embed)[3]["pca_matched"] is False


def test_pca_bank_roundtrip_and_prior_task_selection(monkeypatch, settings):
    info = {"tasks": ["math", "social", "code"], "counts": {"math": 4, "social": 4, "code": 4},
            "fraction": 0.1, "encoder_model": str(settings / "encoder")}
    files.atomic(TaskPCABank.manifest_path(), info)
    task_bank = BatchPCA.from_vectors(np.array([
        [0,0,0,0], [1,0,0,0], [0,1,0,0], [0,0,1,0]], dtype=float))
    arrays = {}
    for task in info["tasks"]:
        for key, val in (("mean", task_bank.mean), ("axis", task_bank.axis),
                         ("refs", task_bank.refs), ("scales", np.array(list(task_bank.scales.values())))):
            arrays[f"{task}__{key}"] = val
    TaskPCABank.path().parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(TaskPCABank.path(), **arrays)
    bank = TaskPCABank.load()
    assert bank.prior_tasks() == ["math", "social"]
    monkeypatch.setenv("RL_PCA_INCLUDE_CURRENT_TASK", "1")
    assert bank.prior_tasks() == ["math", "social", "code"]
    monkeypatch.setenv("RL_FEATURE_MODEL", "different-model")
    with pytest.raises(ValueError, match="encoder model"):
        TaskPCABank.load()


def test_pca_match_policy_integrates_with_track_acceptance(settings):
    track = CodeTrack.make("ours")
    record = {"passed": True, "synth_gen_passed": True,
              "distance_ratios": {"d1": 0.01, "d2": 0.01, "d3": 0.01},
              "metadata": {"pca_by_task": {"math": {"matched": False}}, "pca_matched": False}}
    assert track.rejection_reason(record) == "distance"
    record["metadata"]["pca_matched"] = True
    assert track.accepts(record)


def test_sft_trace_samples_actual_indexed_rows_not_unused_dataset(monkeypatch, settings):
    import pandas as pd
    from custome.rl.run import TraceSFT
    monkeypatch.setenv("RL_SFT_TASK", "math")
    monkeypatch.setenv("RL_SFT_BATCH", "8")
    monkeypatch.setenv("RANK", "0")
    # Suppress process-exit registration inside the fake test dataset.
    monkeypatch.setattr("atexit.register", lambda fn: fn)

    class ToyDataset:
        def __init__(self):
            self.dataframe = pd.DataFrame([
                DataLoad().math({"problem": f"Question {i}",
                                 "solution": rf"Answer \\boxed{{{i}}}"}, index=i).to_storage()
                for i in range(50)
            ])
        def __len__(self):
            return 50
        def __getitem__(self, index):
            return {"token_ids": index}

    class ToySFT:
        def _build_dataset(self):
            self.train_dataset = ToyDataset()

    fake_trainer_pkg = types.ModuleType("verl.trainer")
    fake_sft_module = types.ModuleType("verl.trainer.sft_trainer")
    fake_sft_module.SFTTrainer = ToySFT
    fake_trainer_pkg.sft_trainer = fake_sft_module
    fake_verl = types.ModuleType("verl")
    fake_verl.trainer = fake_trainer_pkg
    monkeypatch.setitem(sys.modules, "verl", fake_verl)
    monkeypatch.setitem(sys.modules, "verl.trainer", fake_trainer_pkg)
    monkeypatch.setitem(sys.modules, "verl.trainer.sft_trainer", fake_sft_module)
    TraceSFT.install()
    trainer = ToySFT()
    trainer._build_dataset()
    assert trainer.train_dataset[12] == {"token_ids": 12}
    trainer.train_dataset[13]
    trainer.train_dataset[12]  # repeated access does NOT double sample
    trainer.train_dataset[14]
    trainer.train_dataset[15]
    trainer.train_dataset[16]
    trainer.train_dataset.save()
    trace = files.read(DataLoad.pca_trace_path("math", 0))
    assert set(trace["observed_ids"]) == {str(i) for i in range(12, 17)}
    assert {r["source_id"] for r in trace["selected"]} <= set(trace["observed_ids"])
    assert all("USER:" in r["pca_text"] and "ASSISTANT:" in r["pca_text"]
               for r in trace["selected"])


def test_progress_log_includes_stage_and_unbuffered_fields(monkeypatch, capsys):
    from custome.rl.core import Progress
    monkeypatch.setenv("RL_PROGRESS_ENABLED", "1")
    Progress.log("unit-test", "model-loading", current=2, total=5)
    output = capsys.readouterr().out
    assert "[unit-test]" in output
    assert "model-loading" in output
    assert "current=2" in output and "total=5" in output


def test_progress_heartbeat_during_blocking_call(monkeypatch, capsys):
    import time
    from custome.rl.core import Progress
    monkeypatch.setenv("RL_PROGRESS_ENABLED", "1")
    monkeypatch.setattr(Progress, "interval", classmethod(lambda cls: .02))
    with Progress.watch("unit-block", "loading model", checkpoint="dummy"):
        time.sleep(.085)
    output = capsys.readouterr().out
    assert output.count("loading model") >= 2
    assert "elapsed_s=" in output


def test_process_wait_prints_progress_before_success(monkeypatch, capsys):
    import subprocess
    from custome.rl.core import Progress
    monkeypatch.setenv("RL_PROGRESS_ENABLED", "1")
    monkeypatch.setattr(Progress, "interval", classmethod(lambda cls: .02))
    child = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(.09)"])
    Progress.wait_process(child, "unit-subprocess", label="slow-stage")
    output = capsys.readouterr().out
    assert "still running" in output
    assert "completed" in output


def test_background_worker_relay_writes_file_and_terminal(settings, monkeypatch, capsys):
    import time
    from custome.rl.run import ProcessGroup
    monkeypatch.setenv("RL_PROGRESS_CHILD_OUTPUT", "1")
    workers = ProcessGroup()
    try:
        child = workers.start("unit_relay", [sys.executable, "-u", "-c", "print('worker-ready',flush=True)"])
        workers.wait(child)
        # Relay thread is asynchronous; wait briefly for log consumption.
        log = settings / "unit_relay.log"
        for _ in range(50):
            if log.exists() and "worker-ready" in log.read_text():
                break
            time.sleep(.01)
        assert "worker-ready" in log.read_text()
        assert "worker-ready" in capsys.readouterr().out
    finally:
        workers.close()


def test_installer_preserves_custom_model_steps_and_adds_progress(tmp_path):
    import subprocess
    package_root = Path(__file__).resolve().parents[3]
    target = tmp_path / "repo"
    (target / "custome").mkdir(parents=True)
    old_config = (package_root / "custome" / "config.sh").read_text()
    old_config = old_config.replace('export RL_BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct"',
                                    'export RL_BASE_MODEL="user/own-selected-base"')
    old_config = old_config.replace('export RL_SFT_STEPS_MATH=50', 'export RL_SFT_STEPS_MATH=123')
    old_config = old_config.replace('export RL_GRPO_STEPS_OURS=50', 'export RL_GRPO_STEPS_OURS=27')
    old_config = old_config.replace('export RL_PROGRESS_INTERVAL_SECS=15\n', '')
    (target / "custome" / "config.sh").write_text('export CUSTOM_SECRET="unchanged"\n'+old_config)
    subprocess.run(["bash", str(package_root / "install.sh"), str(target)], check=True,
                   stdout=subprocess.DEVNULL)
    installed = (target / "custome" / "config.sh").read_text()
    assert 'export RL_BASE_MODEL="user/own-selected-base"' in installed
    assert 'export RL_SFT_STEPS_MATH=123' in installed
    assert 'export RL_GRPO_STEPS_OURS=27' in installed
    assert 'export RL_PROGRESS_INTERVAL_SECS=15' in installed
    assert 'export CUSTOM_SECRET="unchanged"' in installed


def test_grpo_reads_exact_persisted_jsonl_and_rejects_corruption(settings):
    rows = [DataLoad().synth({"id": f"sample-{i}", "problem": f"Question {i}",
                              "code": f"def f(): return {i}", "source": "synth_gen"}).to_dict()
            for i in range(4)]
    refs = _publish_saved_training(0, rows)
    saved = PrivateDataset("generated", "ours")
    assert len(saved.path.read_text().splitlines()) == 4
    assert len(saved.accepted_path.read_text().splitlines()) == 4
    loaded = CodeTrack.make("ours").batch(0, 4, "train")
    assert [r["question"] for r in loaded] == [f"Question {i}" for i in range(4)]
    assert all(r["metadata"]["pca_matched"] for r in loaded)
    # The loader does not trust the ready manifest as a code source.
    assert "generated_code" not in files.read(files.ready(0))["records"][0]
    with saved.accepted_path.open("r+b") as fd:
        fd.seek(refs[0]["offset"] + 1)
        fd.write(b"X")
    with pytest.raises(ValueError, match="corrupted"):
        CodeTrack.make("ours").batch(0, 4, "train")


def test_training_loader_rejects_unverified_stored_sample(settings):
    store = PrivateDataset("generated", "ours")
    bad = DataLoad().synth({"id": "bad", "problem": "Bad code", "code": "pass",
                             "accepted": True, "passed": False, "synth_gen_passed": True,
                             "step": 0, "test_code": "def test(): pass",
                             "metadata": {"pca_matched": True}}).to_dict()
    with pytest.raises(ValueError, match="Unverified"):
        store.append([bad])
    # Also defend the consumer against corrupt/external JSONL edits.
    refs = files.append(store.accepted_path, [bad])
    files.atomic(files.ready(0), {"format": "accepted_jsonl_refs_v1", "step": 0, "records": refs})
    with pytest.raises(ValueError, match="Unverified"):
        CodeTrack.make("ours").batch(0, 1, "train")


def test_existing_private_hf_repo_reused_with_run_isolation(monkeypatch, settings):
    created, uploaded = [], []
    class FakeAPI:
        def __init__(self, *args, **kwargs): pass
        def repo_info(self, **kw):
            assert kw["repo_id"] == "sampleowner/existing-private"
            return types.SimpleNamespace(private=True)
        def create_repo(self, **kw): created.append(kw)
        def upload_file(self, **kw): uploaded.append(kw["path_in_repo"])

    pkg = types.ModuleType("huggingface_hub")
    pkg.HfApi = FakeAPI
    monkeypatch.setitem(sys.modules, "huggingface_hub", pkg)
    monkeypatch.setenv("RL_ROOT", str(settings / "common"))
    monkeypatch.setenv("RL_HF_DATA_REPO", "sampleowner/existing-private")
    monkeypatch.setenv("RL_UPLOAD_EVERY", "2")
    monkeypatch.setenv("RL_RUN_ID", "run_one")
    store = PrivateDataset("generated", "ours")
    store.append([{"source": "synth_gen", "problem": f"test{i}", "code": "pass", "accepted": False}
                  for i in range(2)])
    assert store.upload() == 2
    assert created == []
    monkeypatch.setenv("RL_OUT", str(settings / "another-run"))
    monkeypatch.setenv("RL_RUN_ID", "run_two")
    newer = PrivateDataset("generated", "ours")
    newer.append([{"source": "synth_gen", "problem": f"other{i}", "code": "pass", "accepted": False}
                  for i in range(2)])
    assert newer.upload() == 2
    assert files.read(settings / "common" / "hf_data_repository.json")["repo"] == "sampleowner/existing-private"
    assert "runs/run_one/tracks/ours/data/all_attempts/part-0000000.jsonl" in uploaded
    assert "runs/run_two/tracks/ours/data/all_attempts/part-0000000.jsonl" in uploaded


def test_upload_cursor_rejects_repo_switch_and_keeps_local_data(monkeypatch, settings):
    uploaded = []
    class FakeAPI:
        def __init__(self, *args, **kwargs): pass
        def repo_info(self, **kw): return types.SimpleNamespace(private=True)
        def upload_file(self, **kw): uploaded.append(kw["repo_id"])
    pkg = types.ModuleType("huggingface_hub")
    pkg.HfApi = FakeAPI
    monkeypatch.setitem(sys.modules, "huggingface_hub", pkg)
    monkeypatch.setenv("RL_HF_DATA_REPO", "owner/private_a")
    store = PrivateDataset("generated", "ours")
    store.append([{"source": "synth_gen", "problem": f"q{i}", "code": "pass"}
                  for i in range(2)])
    assert store.upload() == 2
    monkeypatch.setenv("RL_HF_DATA_REPO", "owner/private_b")
    with pytest.raises(RuntimeError, match="cursor belongs"):
        store.upload()
    assert len(store.path.read_text().splitlines()) == 2
    assert uploaded == ["owner/private_a"]


def test_async_producer_to_saved_jsonl_to_actual_grpo_loader(monkeypatch, settings):
    from custome.rl import tracks
    monkeypatch.setenv("RL_GRPO_BATCH", "2")

    class FakeSynth:
        def generate(self, anchor, step, attempt, seed):
            return {"id": f"attempt-{attempt}", "problem": f"Implement f{attempt}",
                    "code": f"def f{attempt}(x): return x + {attempt}",
                    "seed": seed, "synth_gen_passed": True,
                    "test_code": f"def test_f{attempt}(): assert f{attempt}(1) == {attempt + 1}"}

    def fake_service(payload):
        return {"record": {"passed": True, "status": "passed", "pytest_exit_code": 0,
                           "pytest_summary": {"passed": 1, "failed": 0},
                           "pytest_report": {"summary": {"passed": 1}},
                           "metadata": {"pca_matched": True,
                                        "pca_by_task": {"math": {"matched": True}}},
                           "distance_ratios": {"d1": 0.4, "d2": 0.3, "d3": 0.2}}}

    monkeypatch.setattr(tracks, "call_service", fake_service)
    producer = AsyncProducer(CodeTrack.make("ours"))
    producer.synth = FakeSynth()
    produced = asyncio.run(producer.make_batch(0, {
        "problems": ["seedA", "seedB"], "codes": ["a", "b", "c", "d"],
        "reference_step": "code_sft_last_step",
    }))
    refs = producer._batch_refs.pop(0)
    assert len(produced) == len(refs) == 2
    files.atomic(files.ready(0), {"format": "accepted_jsonl_refs_v1", "step": 0, "records": refs})
    loader = StreamingCodeLoader(_data_config=None, _split="train", _batch_size=2,
                                 _tokenizer=None, _processor=None, _collate_fn=lambda x: x)
    loaded = next(iter(loader))
    assert [v["raw_prompt"][0]["content"] for v in loaded] == [
        row["prompt"] for row in producer.store.load_accepted(refs, 0)
    ]
    assert all(r["metadata"]["pca_matched"] for r in producer.store.load_accepted(refs, 0))
    assert files.read(files.anchor(1))["problems"] == [r["question"] for r in produced]
    assert len(producer.store.path.read_text().splitlines()) == 2
    assert len(producer.store.accepted_path.read_text().splitlines()) == 2


def test_smoke_diagnosis_separates_collection_from_behavioral_failure(settings):
    from custome.sample_test import SampleDiagnosis
    sample = {
        "question": "Implement a Fibonacci integer predicate without using dependencies.",
        "generated_code": "from langchain.schema import HumanMessage\ndef pred(n): return False",
        "test_code": "import pytest\ndef test_x(): assert pred(5)",
        "synth_gen_passed": False, "synth_gen_error": "No tests were run.",
        "passed": False, "status": "failed", "pytest_exit_code": 2,
        "pytest_summary": {"total": 0, "collected": 0},
        "pytest_report": {"collectors": [{"outcome": "failed", "longrepr": "E   ModuleNotFoundError: No module named 'langchain'"}]},
    }
    result = SampleDiagnosis.summarize(sample, index=3)
    assert result["diagnosis"]["category"] == "collection_error"
    assert "ModuleNotFoundError" in SampleDiagnosis.render(result)
    assert "pred(n)" in SampleDiagnosis.render(result)
    assert result["independent_pytest"]["failed_tests"] == 0
    assert result["accepted_for_training"] is False


def test_smoke_diagnosis_detects_missing_test_and_question_leak(settings):
    from custome.sample_test import SampleDiagnosis
    result = SampleDiagnosis.summarize({
        "question": "Task\ndef leaked_solution(): return 1",
        "generated_code": "def leaked_solution(): return 1",
        "test_code": "", "passed": False, "status": "no_tests",
        "synth_gen_passed": False, "synth_gen_error": "No tests",
    }, index=1)
    assert result["diagnosis"]["category"] == "prompt_leak"
    assert result["diagnosis"]["quality_warnings"]


def test_smoke_diagnosis_renders_one_readable_sample_and_persists(settings, monkeypatch, capsys):
    from custome.sample_test import CodeGenerationSmokeTest
    monkeypatch.setenv("RL_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    monkeypatch.setenv("RL_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    smoke = CodeGenerationSmokeTest(seed="seed", samples=4)
    info = smoke.print_result({
        "question": "Return n + 1.", "generated_code": "def inc(n): return n+1",
        "test_code": "def test_inc(): assert inc(1)==2",
        "passed": True, "status": "passed", "synth_gen_passed": True,
        "pytest_summary": {"passed": 1, "failed": 0, "collected": 1},
        "pytest_exit_code": 0,
    }, 2)
    display = capsys.readouterr().out
    assert "SAMPLE 2/4" in display
    assert "[3] MODEL PREDICTION" in display
    assert "[6] FAILURE REASON" in display
    assert "VERIFICATION HISTORY" not in display
    assert smoke.report_path.exists() and "SAMPLE 2/4" in smoke.report_path.read_text()
    row = json.loads(smoke.summary_path.read_text().splitlines()[-1])
    assert row["sample_number"] == 2 and row["predicted_code"] == "def inc(n): return n+1"
    assert row["diagnosis"]["category"] == "passed"


def test_smoke_warnings_flag_missing_import_and_langchain_dependency(settings):
    from custome.sample_test import SampleDiagnosis
    info = SampleDiagnosis.summarize({
        "question": "Determine whether an integer is Fibonacci.",
        "generated_code": "from langchain.schema import HumanMessage\ndef f(n): return n>0",
        "test_code": "@pytest.mark.parametrize('n', [5])\ndef test_x(n): assert f(n)",
        "status": "failed", "passed": False, "pytest_exit_code": 2,
        "pytest_summary": {"collected": 0},
    }, index=1)
    assert len(info["diagnosis"]["quality_warnings"]) == 2


def test_smoke_run_four_samples_prints_separate_sorted_audit(monkeypatch, settings, capsys):
    from concurrent.futures import Future
    from custome import main as generator_api
    from custome import sample_test
    from custome.rl.score import CodeVerifier
    monkeypatch.setenv("RL_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    monkeypatch.setenv("RL_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    monkeypatch.setenv("RL_SYNTH_BATCH_SIZE", "4")
    class Pool:
        def __init__(self, worker_log_dir=None):
            self.worker_log_dir = worker_log_dir
        def __enter__(self): return self
        def __exit__(self, *exc): pass
        def submit(self, snapshot, anchor, step, attempt, seed):
            f = Future()
            f.set_result({"id": f"id-{attempt}", "problem": f"Question {attempt}: increment an integer.",
                          "seed": seed, "code": f"def inc(n): return n + {attempt}",
                          "test_code": f"def test_inc(): assert inc(1) == {attempt + 1}",
                          "synth_gen_passed": True,
                          "verification_history": [{"event": "pytest_execution", "pytest_report": {"summary": {"passed": 1}}}]})
            return f
    class Verifier:
        def verify(self, code, tests):
            return CodeVerifier.result(passed=True, status="passed", summary={"passed": 1, "collected": 1},
                  pytest_report={"summary": {"passed": 1, "collected": 1}},
                  pytest_exit_code=0, test_code=tests)
    monkeypatch.setattr(generator_api, "CodeGenerationPool", Pool)
    monkeypatch.setattr(sample_test, "CodeVerifier", Verifier)
    tester = sample_test.CodeGenerationSmokeTest(seed="seed", samples=4)
    tester.run()
    seen = capsys.readouterr().out
    for n in range(1, 5):
        assert f"SAMPLE {n}/4" in seen
        assert f"SAMPLE {n}: " in seen
    report = tester.report_path.read_text()
    assert report.index("SAMPLE 1/4") < report.index("SAMPLE 2/4") < report.index("SAMPLE 3/4") < report.index("SAMPLE 4/4")
    assert tester.summary_path.exists()
    assert len(tester.summary_path.read_text().splitlines()) == 4


def test_smoke_gpu_pool_worker_log_initializer_is_optional(monkeypatch):
    from custome import main as generator_api
    values = {}
    class Executor:
        def __init__(self, **kwargs):
            values.update(kwargs)
        def shutdown(self, **kwargs): pass
    monkeypatch.setattr(generator_api, "ProcessPoolExecutor", Executor)
    monkeypatch.setattr(generator_api, "_visible_gpu_count", lambda: 2)
    monkeypatch.setenv("RL_SYNTH_GPU_MEMORY_FRACTION", "0.90")
    monkeypatch.setenv("RL_SYNTH_BATCH_SIZE", "4")
    monkeypatch.setenv("RL_SYNTH_PROCESSES", "4")
    monkeypatch.setenv("RL_SYNTH_START_METHOD", "spawn")
    with generator_api.CodeGenerationPool(worker_log_dir="/tmp/sample-worker-logs"):
        assert values["initargs"] == ("/tmp/sample-worker-logs",)
        assert values["initializer"] is generator_api._redirect_worker_output
    values.clear()
    with generator_api.CodeGenerationPool():
        assert "initializer" not in values  # RL training keeps existing logging


def test_prompt_content_unwraps_chat_objects_and_rejects_non_text():
    """Never let default object repr leak into the LLM's user prompt."""
    from custome.rl.data_load import prompt_content
    from types import SimpleNamespace as NS

    class HumanMessage:
        def __init__(self, content): self.content = content
        def __repr__(self): return f"HumanMessage(content={self.content!r})"

    class ChatPromptValue:
        def __init__(self, messages): self.messages = messages
        def __repr__(self): return f"messages={self.messages!r}"

    message = HumanMessage('Solve the following question: 1 + 1 = ?')
    wrapper = ChatPromptValue([message])
    assert prompt_content(wrapper) == 'Solve the following question: 1 + 1 = ?'
    assert 'HumanMessage' not in prompt_content(wrapper)
    assert prompt_content(ChatPromptValue([HumanMessage('line 1'), HumanMessage('line 2')])) == 'line 1\nline 2'
    assert prompt_content({'messages': [{'role': 'user', 'content': 'Only text'}]}) == 'Only text'
    assert prompt_content(NS(content='A sentence')) == 'A sentence'
    with pytest.raises(TypeError, match='must contain text'):
        prompt_content({'question': 'unrecognized chat object'})
    with pytest.raises(TypeError, match='must contain text'):
        prompt_content(NS(foo='no text'))


def test_real_synth_adapter_feeds_problem_text_not_chat_repr(monkeypatch):
    """Exercise the local subclass without GPU, network, or installed synth_gen."""
    from custome import main as api
    from custome.rl.data_load import prompt_content
    from types import SimpleNamespace as NS
    seen = {}
    monkeypatch.setenv("RL_SYNTH_VERBOSE", "0")

    class Message:
        def __init__(self, content):
            self.content = content
            self.type = 'human'

    class ChatPrompt:
        def __init__(self, text): self.messages = [Message(text)]
        def __str__(self):
            return f"messages=[HumanMessage(content={self.messages[0].content!r})]"

    class FakeVerifier:
        def message_to_string(self, obj):
            return str(obj)  # Simulates upstream's unsafe multi-message repr
        def check(self, record):
            seen['verifier_problem'] = self.message_to_string(record.prompt)
            return True, None

    class FakeBase:
        def __init__(self):
            self.verifiers = [FakeVerifier()]
            self.llm = 'CURRENT_ACTOR_ONLY'
        def call_llm(self, prompt, **kwargs):
            seen.setdefault('all_problem_prompts', []).append(prompt)
            seen['problem_gen_prompt'] = prompt
            if len(seen['all_problem_prompts']) == 1:
                # A previous version leaked this source code as the QUESTION.
                return 'def leaked_solution(n):\n    return n + 1'
            return 'Write a function that computes 1 + 1.'
        def convert_prompt(self, text, generation):
            generation.prompt = ChatPrompt(text)
            return generation
        def set_llm(self, value): self.llm = value
        def _fix_response(self, gen, ver):
            seen['fix_called'] = prompt_content(gen.prompt)
            return gen

    def add_module(name, **attrs):
        mod = types.ModuleType(name)
        for k, v in attrs.items(): setattr(mod, k, v)
        if name in ('synth_gen', 'synth_gen.generation', 'synth_gen.verification',
                    'langchain_core', 'langchain_core.language_models'):
            mod.__path__ = []
        monkeypatch.setitem(sys.modules, name, mod)
        return mod

    add_module('langchain_core')
    add_module('langchain_core.language_models')
    add_module('langchain_core.language_models.chat_models', BaseChatModel=object)
    add_module('langchain_core.messages', AIMessage=Message)
    add_module('langchain_core.outputs', ChatGeneration=object, ChatResult=object)
    add_module('synth_gen')
    add_module('synth_gen.generators', PythonVerifiedSynthDataGenerator=FakeBase)
    add_module('synth_gen.generation')
    add_module('synth_gen.generation.parsing', extract_source_code=lambda text: text)
    def record_solution(*, problem, llm, programming_language):
        seen['solution_prompt_problem'] = problem
        seen['solution_model'] = llm
        return 'def f(): return 2'
    add_module('synth_gen.generation.programming', generate_programming_solution=record_solution)
    add_module('synth_gen.verification')
    add_module('synth_gen.verification.python_verifiers')
    add_module('custome.podman_generating', run_pytest=lambda *a, **kw: (None, '', '', 0),
               run_test_coverage=lambda *a, **kw: (None, '', '', 0))

    adapter = api.VerifiedCodeGenerator.__new__(api.VerifiedCodeGenerator)
    adapter.verification_history = []
    generator = adapter._make_generator()
    obj = NS(problem=None, prompt=None, code=None, response=None)
    obj = generator.generate_prompt('A Fibonacci seed', obj)
    assert isinstance(seen['problem_gen_prompt'], str)
    assert 'A Fibonacci seed' in seen['problem_gen_prompt']
    assert len(seen['all_problem_prompts']) == 2  # bad problem was regenerated
    assert seen['all_problem_prompts'][0] == seen['all_problem_prompts'][1]
    assert 'def leaked_solution' not in seen['all_problem_prompts'][1]
    assert obj.prompt.__str__().startswith('messages=[HumanMessage')
    generator._generate_response(obj)
    assert seen['solution_prompt_problem'] == 'Write a function that computes 1 + 1.'
    assert 'HumanMessage' not in seen['solution_prompt_problem']
    assert 'ChatPromptValue' not in seen['solution_prompt_problem']
    assert obj.code == 'def f(): return 2'
    assert obj.response.content == obj.code
    assert generator.verifiers[0].check(obj) == (True, None)
    assert seen['verifier_problem'] == 'Write a function that computes 1 + 1.'
    assert 'HumanMessage' not in seen['verifier_problem']
    assert seen['solution_model'] is generator.llm
    # Fail closed if ALL attempts contain executable code.
    monkeypatch.setenv('RL_SYNTH_PROBLEM_RETRIES', '2')
    generator.call_llm = lambda prompt, **kwargs: 'def leaked(n): return n + 1'
    with pytest.raises(ValueError, match='Generated problem rejected after 2 attempts'):
        generator.generate_prompt('test seed', NS(problem=None, prompt=None))
    assert adapter.rejected_problem_text == 'def leaked(n): return n + 1'
    with pytest.raises(ValueError, match='Refusing solution generation'):
        generator._generate_response(NS(problem='def bad(n): return 1', prompt=None))


def test_every_task_sft_grpo_eval_receives_only_canonical_prompt(settings):
    """Cross-task prompt regressions: MCQA, short-answer, APPS, synthetic."""
    loader = DataLoad()
    rows = [
        loader.math({'problem': 'What is 1+1?', 'solution': 'It is \\boxed{2}'}),
        loader.social({'context': 'A person is hungry.', 'question': 'What next?',
                       'answerA': 'Eat', 'answerB': 'Sleep', 'answerC': 'Run', 'label': '1'}),
        loader.code({'question': 'Implement add(a, b).',
                     'solutions': json.dumps(['def add(a, b): return a+b']),
                     'input_output': '{}'}),
        loader.synth({'problem': 'Implement square(n).',
                      'code': 'def square(n): return n*n',
                      'test_code': 'def test_square(): assert square(2)==4'}),
    ]
    for sample in rows:
        sft = sample.to_sft()
        grpo = sample.to_verl(step=0)
        assert isinstance(sample.prompt, str)
        assert sft['messages'][0] == {'role': 'user', 'content': sample.prompt}
        assert grpo['raw_prompt'][0] == {'role': 'user', 'content': sample.prompt}
        assert 'HumanMessage(' not in sample.prompt
        assert 'ChatPromptValue(' not in sample.prompt
        assert 'messages=[' not in sample.prompt
        assert sample.question in sample.prompt
        # Evaluation consumes Sample.prompt, same as the two training APIs.
    assert 'Choices:' in rows[1].prompt and 'A. Eat' in rows[1].prompt
    assert 'Write runnable Python code only.' in rows[2].prompt


# Leakage hardening: assistant SFT targets are allowed; user prompts are not.
def test_math_gold_reference_contains_only_final_answer(settings):
    raw = {"problem": "Compute 2+3.", "solution": "First calculate 2+3=5, therefore \\boxed{5}"}
    sample = DataLoad().math(raw)
    assert sample.reference == sample.answer == "5"
    assert sample.sft_solution == sample.training_answer == raw["solution"]
    assert raw["solution"] not in sample.prompt
    assert raw["solution"] not in sample.to_sft()["messages"][0]["content"]
    assert sample.to_sft()["messages"][1]["content"] == raw["solution"]
    assert sample.to_verl(0)["raw_prompt"][0]["content"] == sample.prompt
    # Pre-existing Parquet files should be upgraded without changing model target.
    old = sample.to_dict()
    old["reference"] = old["sft_solution"]
    old["sft_solution"] = ""
    recovered = Sample.from_dict(old)
    assert recovered.reference == "5" and recovered.training_answer == raw["solution"]


def test_math_unboxed_answer_does_not_turn_reasoning_into_gold(settings):
    with pytest.raises(ValueError, match="boxed"):
        DataLoad().math({"problem": "What is 3+4?", "solution": "So the answer is 7."})
    with pytest.raises(ValueError, match="boxed"):
        DataLoad().math({"problem": "What is 3+4?", "solution": "Therefore 7"})
    with pytest.raises(ValueError, match="answer/solution marker"):
        DataLoad().math({"problem": "The answer is \\boxed{7}", "solution": "\\boxed{7}"}).assert_prompt_integrity()


def test_social_answer_only_in_assistant_not_question_label(settings):
    sample = DataLoad().social({"context": "Someone feels ill.", "question": "What should they do?",
                                "answerA": "Rest", "answerB": "Run", "answerC": "Cook", "label": 1})
    assert sample.to_verl(2)["raw_prompt"][0]["content"] == sample.prompt
    assert "label" not in sample.prompt.lower()
    assert sample.reference == "" and sample.answer == "A"
    assert sample.to_sft()["messages"][1]["content"] == "A"
    with pytest.raises(ValueError, match="gold label"):
        DataLoad().social({"context": "Anything", "question": "Which?", "answerA": "A",
                           "answerB": "B", "answerC": "C", "label": 4})
    with pytest.raises(ValueError, match="gold answer label"):
        DataLoad().social({"context": "Correct answer: B", "question": "Which?",
                           "answerA": "A", "answerB": "B", "answerC": "C", "label": 2}).to_sft()


def test_apps_prompt_never_receives_reference_code_or_test_answers(settings):
    code = "def solve(a,b): return a+b\n"
    sample = DataLoad().code({"question": "Implement addition of two integers.",
                              "solutions": json.dumps([code]),
                              "input_output": json.dumps({"inputs": ["1 2"], "outputs": ["3"]})})
    assert code not in sample.prompt
    assert "3" not in sample.prompt
    assert sample.to_sft()["messages"][1]["content"] == code
    grpo = sample.to_verl(0)
    assert grpo["raw_prompt"] == [{"role": "user", "content": sample.prompt}]
    assert grpo["reward_model"]["ground_truth"] == ""
    assert grpo["extra_info"]["apps_io"] == sample.apps_io
    assert grpo["extra_info"]["apps_io"] not in grpo["raw_prompt"][0]["content"]
    long_gold = "def solve(x):\n    # reference solution\n    return sorted(x) * 10\n" * 3
    with pytest.raises(ValueError, match="official reference"):
        DataLoad().code({"question": "Implement sorting.\n" + long_gold,
                         "solutions": json.dumps([long_gold])}).to_sft()


@pytest.mark.parametrize("question", [
    "python\ndef fib(n):\n    return n",
    "def fibonacci(n):\n    return n if n < 2 else fibonacci(n-1)+fibonacci(n-2)",
    "Answer key: 34",
    "\n```python\ndef f(x): return x\n```",
    "messages=[HumanMessage(content='implement fib')]",
    "import os\nassert True",
    "Implement add(x)\n    return x + 1",
])
def test_synth_problem_with_code_or_answer_is_rejected(question, settings):
    from custome.rl.data_load import problem_leak_reason
    assert problem_leak_reason(question)
    sample = DataLoad().synth({"problem": question, "code": "def f(): return 7",
                               "test_code": "def test_f(): assert f()==7", "passed": True,
                               "synth_gen_passed": True, "accepted": True})
    with pytest.raises(ValueError, match="synthetic question leaks|serialized chat object"):
        sample.to_verl(0)
    with pytest.raises(ValueError, match="synthetic question leaks|serialized chat object"):
        sample.to_sft()
    assert CodeTrack.make("ours").rejection_reason(sample.to_dict()) == "prompt_leak"
    store = PrivateDataset("generated", "ours")
    with pytest.raises(ValueError, match="synthetic question leaks|serialized chat object"):
        store.append([sample.to_dict()])
    assert store.path.exists()  # Rejected original audit trail stays durable.
    assert not store.accepted_path.exists()


def test_question_only_synth_seed_anchor_hides_gold_code(settings):
    rows = [DataLoad().code({"question": f"Write arithmetic function {i}.",
                             "solutions": json.dumps([f"def solve(): return {i}"])})
            for i in range(4)]
    track = CodeTrack.make("baseline")
    track.consumed(0, [sample.to_dict() for sample in rows])
    anchor = files.read(files.anchor(1, "baseline"))
    assert anchor["problems"] == [row.question for row in rows]
    assert "codes" not in anchor
    assert all(row.reference not in json.dumps(anchor) for row in rows)


def test_synth_generation_retries_answer_leaks_without_reusing_output(monkeypatch):
    from custome import main as api
    from types import SimpleNamespace as NS
    seen = []
    # Reuse the class instantiated in the existing full adapter test by
    # spying on the source-level guards; no CUDA imports are needed.
    instruction = api.synth_problem_instruction("Fibonacci with a twist")
    assert "TOPIC SEED" in instruction
    assert "reference code" in instruction.lower() or "source code" in instruction.lower()
    assert "HumanMessage(" not in instruction
    assert "def fib(" not in instruction
    from custome.rl.data_load import problem_leak_reason
    assert problem_leak_reason("Compute a Fibonacci number from an integer input.") == ""



def test_synth_verifier_failure_cannot_pass_scoring_acceptance(settings):
    # Reward service returns a filtered verification record without source/name.
    # Producer now restores synthesis provenance before acceptance decisions.
    track = CodeTrack.make("ours")
    result = {"passed": True, "metadata": {"pca_matched": True,
              "pca_by_task": {"math": {"matched": True}}}}
    assert track.rejection_reason({**result, "source": "synth_gen",
                                   "synth_gen_passed": False}) == "synth_gen_verifier"
    assert track.rejection_reason({**result, "source": "synth_gen",
                                   "synth_gen_passed": True}) == ""


def test_smoke_code_answer_leak_is_recorded_without_podman(monkeypatch, settings):
    from custome.sample_test import CodeGenerationSmokeTest, SampleDiagnosis
    monkeypatch.setenv("RL_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    monkeypatch.setenv("RL_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    smoke = CodeGenerationSmokeTest(seed="fib", samples=1)

    class NeverCallPodman:
        def verify(self, *args, **kwargs):
            raise AssertionError("Invalid question must be rejected before Podman")

    row = smoke.verify_and_record({
        "id": "leaky-question", "seed": "fib", "problem": "def fib(n): return n",
        "code": "def fib(n): return n", "source": "synth_gen", "synth_gen_passed": False,
    }, NeverCallPodman(), PrivateDataset("generated", "ours"))
    diagnosis = SampleDiagnosis.summarize(row, index=1)
    assert row["status"] == "prompt_leak" and row["accepted"] is False
    assert row["failure_stage"] == "problem_generation"
    assert diagnosis["diagnosis"]["category"] == "prompt_leak"
    assert not PrivateDataset("generated", "ours").accepted_path.exists()
    assert PrivateDataset("generated", "ours").path.exists()


def test_strict_math_rejects_unbalanced_box_before_using_reasoning(settings):
    from custome.rl.data_load import strict_math_answer
    assert strict_math_answer('Then the answer is \boxed{5') == ''
    assert strict_math_answer(r'The value is \boxed{\frac{1}{2}}') == r'\frac{1}{2}'
    with pytest.raises(ValueError, match='boxed'):
        DataLoad().math({'problem': 'Compute.', 'solution': 'We find \boxed{5'})


def test_cached_dataset_audit_checks_all_tasks_and_blocks_poisoned_prompt(monkeypatch, settings):
    """Audit the exact normalized record layer, without a real HF download."""
    records = {
        "math": DataLoad().math({"problem": "Compute 2+3.", "solution": r"Then \boxed{5}"}),
        "social": DataLoad().social({"context": "Someone is thirsty.", "question": "What now?",
                                     "answerA": "Drink", "answerB": "Sleep", "answerC": "Jump", "label": 1}),
        "code": DataLoad().code({"question": "Write an addition function.",
                                 "solutions": json.dumps(["def add(a,b): return a+b"])}),
    }
    src = settings / "cached"
    src.mkdir()
    monkeypatch.setenv("RL_DATA", str(src))
    for name in records:
        for split in ("train", "eval"):
            (src / f"{name}_{split}_raw.parquet").touch()

    class FakeFrame:
        def __init__(self, rows): self.rows = rows
        def __len__(self): return len(self.rows)
        def to_dict(self, orient):
            assert orient == "records"
            return self.rows

    state = {key: [value.to_storage()] for key, value in records.items()}
    fake_pandas = types.ModuleType("pandas")
    fake_pandas.read_parquet = lambda path: FakeFrame(state[Path(path).name.split("_")[0]])
    monkeypatch.setitem(sys.modules, "pandas", fake_pandas)

    report = DataLoad.audit_cached()
    assert set(report) == {f"{t}/{s}" for t in records for s in ("train", "eval")}
    assert all(v["invalid"] == 0 for v in report.values())
    assert files.read(files.output / "dataset_audit" / "cached_report.json") == report

    poisoned = records["math"].to_storage()
    poisoned["question"] = r"The gold is \boxed{5}"
    state["math"] = [poisoned]
    with pytest.raises(ValueError, match="BLOCKED"):
        DataLoad.audit_cached()


@pytest.mark.parametrize("problem", [
    """### Problem Statement:
Given a, b compute their sum.
**Example:**
Input: a=2, b=3
Result: 2 + 3 = 5""",
    "Generate Fibonacci values using modulo a large prime number (e.g., 10^9 + 7).",
])
def test_synth_problem_rejects_worked_example_and_ambiguous_modulo(problem, settings):
    from custome.rl.data_load import problem_leak_reason
    assert problem_leak_reason(problem)
    sample = DataLoad().synth({"problem": problem, "code": "def f(): return 7", "test_code": "def test_f(): assert f()==7"})
    with pytest.raises(ValueError, match="synthetic question leaks"):
        sample.to_verl(0)


def test_generated_test_quality_rejects_tautology_and_zero_collected():
    from custome.rl.score import CodeVerifier
    code = "def fibonacci_mod(n): return n"
    bad = [
        "assert fibonacci_mod(1) == 1",
        "def test_large():\n    expected_result = fibonacci_mod(100)\n    assert fibonacci_mod(100) == expected_result",
        "def test_foo():\n    assert fibonacci_mod(1) == fibonacci_mod(1)",
        "import pytest\ndef fibonacci_mod(n): return 999\ndef test_x(): assert fibonacci_mod(1) == 999",
        "def test_x():\n    assert pytest.approx(fibonacci_mod(1)) == 1",
    ]
    for test in bad:
        issue = CodeVerifier.generated_test_issue(code, test)
        assert issue, test
    assert CodeVerifier.generated_test_issue(code, "def test_f():\n    assert fibonacci_mod(3) == 3") == ""
    assert CodeVerifier.generated_test_issue(code, "import unittest\nclass TestF(unittest.TestCase):\n    def test_x(self):\n        self.assertEqual(fibonacci_mod(3), 3)") == ""


def test_invalid_generated_pytest_rejected_before_remote_execution(monkeypatch, settings):
    from custome.rl.score import CodeVerifier
    from custome.sample_test import SampleDiagnosis
    # Incomplete code in the original trace is not executed for invalid tests.
    row = CodeVerifier().verify("def f(n): return n", "assert f(1) == 1")
    assert row["status"] == "invalid_test"
    assert row["passed"] is False and row["pytest_report"] == {}
    diagnosed = SampleDiagnosis.summarize({"question": "Compute f(n).", "generated_code": "def f(n): return n", "test_code": "assert f(1) == 1", "status": row["status"], "error_message": row["error_message"], "passed": False}, index=1)
    assert diagnosed["diagnosis"]["category"] == "invalid_test"
    assert diagnosed["diagnosis"]["outcome"] == "정답 여부 미판정"


def test_generated_test_without_asserts_never_qualifies(settings):
    from custome.rl.score import CodeVerifier
    code = "def fibonacci_mod(n): return n"
    for bad in ["def test_x(): pass", "def test_x(): fibonacci_mod(1)", "assert fibonacci_mod(1) == 1\ndef test_x(): assert fibonacci_mod(2) == 2"]:
        assert CodeVerifier.generated_test_issue(code, bad), bad


# Problem + unverified prediction -> test generation; never let candidate
# outputs serve as labels (independent-oracle contract).
def test_generated_test_prompt_explicitly_includes_problem_and_actual_prediction():
    from custome.main import synth_pytest_instruction
    prompt = synth_pytest_instruction(
        "Return the sum of two integers.",
        "def add(a,b): return a-b", "def add(a, b)",
    )
    assert "Return the sum of two integers." in prompt
    assert "def add(a,b): return a-b" in prompt
    assert "MODEL-PREDICTED PYTHON CODE / UNTRUSTED" in prompt
    assert "NEVER use a call to the predicted function" in prompt
    assert "correct reference solution" in prompt


def test_test_generator_retries_self_oracle_then_accepts_independent_oracle(monkeypatch):
    from custome.main import generate_pytest_for_prediction
    calls = []
    def fake_model(prompt):
        calls.append(prompt)
        if len(calls) == 1:
            return """def test_add():
    expected = add(2,3)
    assert add(2,3) == expected"""
        return """def test_add():
    assert add(2,3) == 5
    assert add(-1,1) == 0"""
    code = "def add(a,b): return a+b"
    test = generate_pytest_for_prediction("Return a+b.", code, llm=None,
                                           call_model=fake_model, attempts=3)
    assert len(calls) == 2
    assert "PREVIOUS INVALID TEST FEEDBACK" in calls[1]
    assert "assert add(2,3) == 5" in test


def test_test_generator_fails_closed_if_all_generated_tests_are_invalid():
    from custome.main import generate_pytest_for_prediction
    from custome.rl.score import CodeVerifier
    def bad_model(prompt):
        return """def test_add():
    expected_result = add(3,4)
    assert add(3,4) == expected_result"""
    with pytest.raises(ValueError, match="InvalidGeneratedTest: after 2 attempts"):
        generate_pytest_for_prediction("Add integers", "def add(a,b):return a+b", None,
                                       call_model=bad_model, attempts=2)
    assert CodeVerifier.generated_test_issue("def add(a,b):return a+b", bad_model(""))


def test_generated_test_never_accepts_predicted_output_as_expected_value():
    from custome.rl.score import CodeVerifier
    candidate = "def add(a,b): return a+b"
    tests = [
        "def test_add():\n    expected_result = add(2,3)\n    assert add(2,3) == expected_result",
        "def test_add():\n    gold_label = add(2,3)\n    assert gold_label == 5",
        "def test_add():\n    result = add(2,3)\n    other = add(2,3)\n    assert result == other",
        "def test_add():\n    assert True",
    ]
    for test in tests:
        assert CodeVerifier.generated_test_issue(candidate,test), test
    assert not CodeVerifier.generated_test_issue(
        candidate, "def test_add():\n    actual=add(2,3)\n    assert actual == 5")


def test_problem_code_generation_used_in_both_smoke_and_grpo_paths():
    # Both entrypoints instantiate the same CodeGenerationPool. No duplicate
    # test-authoring model/call is defined in the producer.
    from custome.main import VerifiedCodeGenerator, CodeGenerationPool
    from custome.sample_test import CodeGenerationSmokeTest
    from custome.rl.tracks import AsyncProducer
    import inspect
    source = inspect.getsource(VerifiedCodeGenerator._make_generator)
    assert "generate_pytest_for_prediction(" in source
    assert "record.test_code = generate_pytest_for_prediction" in source
    assert "owner._active_test_record = record" in source
    assert "generated_test_issue" in source
    assert "CodeGenerationPool" in inspect.getsource(CodeGenerationSmokeTest)
    assert "CodeGenerationPool" in inspect.getsource(AsyncProducer)


def test_generated_pytest_retry_audits_every_rejected_attempt():
    from custome.main import generate_pytest_for_prediction
    history = []
    def fake_model(_):
        return "def test_add():\n    expected = add(2,3)\n    assert add(2,3) == expected"
    with pytest.raises(ValueError, match="InvalidGeneratedTest"):
        generate_pytest_for_prediction("Compute sum", "def add(a,b):return a+b", llm=None,
                                       call_model=fake_model, attempts=3, audit=history.append)
    assert len(history) == 3
    assert all(x["event"] == "pytest_generation_attempt" for x in history)
    assert all(x["passed"] is False for x in history)
    assert all("expected" in x["error"] for x in history)
    assert all("MODEL-PREDICTED PYTHON CODE" in x["generation_prompt"] for x in history)
    assert all("def add(a,b):return a+b" in x["code"] for x in history)
