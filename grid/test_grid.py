"""Offline tests of the grid tooling. No model calls.  Run from the project root:  python3 -m pytest grid -q"""
from __future__ import annotations

import gzip
import json
import shutil
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "grid"))

import run  # noqa: E402
from backends import GUARD_PREFIX, GridBackend  # noqa: E402
from semaudit.core import Completion, FatalBackendError, Task  # noqa: E402


# --- roster: the standard state ----------------------------------------------------------------------

def _logged_specs():
    specs = {}
    for name in ("models-v2.toml", "models-v3.toml"):
        for b in tomllib.loads((run.H / "configs" / name).read_text())["backends"]:
            specs[b["name"]] = b
    return specs


def test_roster_has_the_eight_models_in_one_state():
    r = run.roster()
    assert list(r) == run.MODELS and len(r) == 8
    logged = _logged_specs()
    for name, spec in r.items():
        params = spec["params"]
        assert params["max_tokens"] == 1024
        assert "temperature" not in params and "seed" not in params and "top_logprobs" not in params
        if spec["kind"] == "anthropic":
            assert params["thinking"] in ("disabled", "between_tools")
        else:
            assert params["reasoning_effort"] in ("none", "minimal")
        if name != "qwen-3.7-flash":  # reused arms keep their logged settings exactly
            assert spec == logged[name]
    assert r["qwen-3.7-flash"]["model"] == logged["qwen-3.7-flash-t0"]["model"]


# --- guards and raw responses ------------------------------------------------------------------------

class _Inner:
    def __init__(self, comp, params=None):
        self.name, self.comp = "m", comp
        self.params = params or {}
        self.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=self._create)))

    def _create(self, **kw):
        return SimpleNamespace(model_dump=lambda mode="json": {"id": "r1", "body": kw.get("x")})

    def describe(self):
        return {"provider": "openrouter", "model": "a/m", "params": self.params}

    def complete(self, task):
        self.client.chat.completions.create(x=task.user)  # what a real backend does inside complete()
        return Completion(text="True", served_model=self.comp.served_model, usage=dict(self.comp.usage),
                          extra=dict(self.comp.extra))


TASK = Task(task_id="t:0", identity="filter_algebra", condition="p", unit="u", repeat=0, op="filter",
            items=("u",), system="s", user="hello", payload={}, context={})


def test_raw_response_is_stored():
    b = GridBackend(_Inner(Completion(text="True", served_model="a/m", usage={})))
    comp = b.complete(TASK)
    assert comp.error is None and comp.extra["raw_response"] == {"id": "r1", "body": "hello"}


@pytest.mark.parametrize("comp", [
    Completion(text="True", served_model="a/m", usage={"reasoning_tokens": 3}),
    Completion(text="True", served_model="a/other", usage={}),
    Completion(text="True", served_model="a/m", usage={}, extra={"served_provider": "Other Host"}),
])
def test_guard_logs_the_call_then_stops(comp):
    inner = _Inner(comp, params={"provider": {"order": ["openai"]}})
    b = GridBackend(inner)
    first = b.complete(TASK)
    assert first.error and first.error.startswith(GUARD_PREFIX)
    with pytest.raises(FatalBackendError):
        b.complete(TASK)


# --- end to end, offline: every kind of run, resume, crash recovery ---------------------------------

def _count(path: Path) -> int:
    with gzip.open(path, "rt") as fh:
        return sum(1 for _ in fh)


@pytest.fixture
def root(tmp_path):
    yield tmp_path / "runs"
    shutil.rmtree(tmp_path, ignore_errors=True)


@pytest.mark.parametrize("kind,where", [("count", "cuad-B"), ("batch", "abt-B"), ("idem", "arxiv-A"),
                                        ("sym", "abt"), ("place", "wdc"), ("lr", "argq-B")])
def test_offline_run_resume_and_recovery(kind, where, root):
    records = {"wdc": "wdc-products", "abt": "abt-buy"}.get(where.split("-")[0])
    data = Path(__file__).resolve().parents[1] / "data"
    if records and not (data / records / "records.csv").exists():
        pytest.skip(f"needs data/{records}/records.csv, which the release does not redistribute")
    r = {"kind": kind, "where": where, "run_id": f"grid-{where}-{kind}", "usd": 0, "cap": 0}
    model = "gpt-5.6-luna"
    run.run_one(r, model, offline=True, root=root)
    log = root / r["run_id"] / f"{model}.jsonl.gz"
    n = _count(log)
    if kind != "lr":
        _, tasks, _, _, _ = run.tasks_for(kind, where)
        expected = len(tasks) + (len([t for t in tasks if t.stage == 0]) if kind == "idem" else 0)
        assert n == expected
    else:
        assert n == 200 * 5 * (5 + 5) + 20 * 5  # 5 claims + 5 rewordings on 200 rows, and 20 COUNT batches
    design = json.loads((root / r["run_id"] / "design.json").read_text())
    assert design  # written before the first call
    run.run_one(r, model, offline=True, root=root)  # resume: nothing left to send
    assert _count(log) == n
    # crash: keep the first half of the log, as an interrupted run would; resume completes it
    with gzip.open(log, "rt") as fh:
        lines = fh.readlines()
    with gzip.open(log, "wt") as fh:
        fh.writelines(lines[: len(lines) // 2])
    run.run_one(r, model, offline=True, root=root)
    with gzip.open(log, "rt") as fh:
        ids = [json.loads(x)["task_id"] for x in fh]
    assert len(ids) == n and len(set(ids)) == n  # every call once, none twice


def test_raw_response_reaches_the_log_and_resume_keeps_it(root):
    """The wrapper's raw body is written into each logged call, flushed as it happens, and a resume
    adds nothing and leaves the stored bodies untouched."""
    class Fake(_Inner):
        def complete(self, task):
            self.client.chat.completions.create(x=task.user)
            return Completion(text="True", served_model="a/m", usage={"cost": 0.0})
    ds, tasks, design, seed, config = run.tasks_for("count", "cuad-B")
    tasks = tasks[:40]
    b = GridBackend(Fake(Completion(text="True", served_model="a/m", usage={})))
    run.run_tasks("grid-test-raw", ds, tasks, design, seed, config, b, root, cap=None)
    log = root / "grid-test-raw" / "m.jsonl.gz"
    with gzip.open(log, "rt") as fh:
        recs = [json.loads(x) for x in fh]
    assert len(recs) == 40
    assert all(r["extra"]["raw_response"]["body"] == r["user"] for r in recs)
    run.run_tasks("grid-test-raw", ds, tasks, design, seed, config, b, root, cap=None)
    with gzip.open(log, "rt") as fh:
        assert [json.loads(x) for x in fh] == recs


def test_account_limit_stops_the_model_at_once():
    class Limited(_Inner):
        def complete(self, task):
            return Completion(text=None, error="PermissionDeniedError(403): Key limit exceeded ($250.00 of $250.00 used)")
    b = GridBackend(Limited(Completion(text=None)))
    first = b.complete(TASK)
    assert first.error.startswith("PermissionDeniedError")  # the call is logged as it failed
    with pytest.raises(FatalBackendError):
        b.complete(TASK)
