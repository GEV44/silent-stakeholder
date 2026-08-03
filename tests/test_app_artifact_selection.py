"""Which cached run the explorer opens, and whether it can be trusted to say so.

Written by block-e after a demo dress rehearsal (`docs/DEMO_DRESS_REHEARSAL.md`)
found that `artifact_dir()` offers **every** `out/` subdirectory containing a
`top_gaps.json`, with no distinction between the primary run and runs we have
documented as unpresentable — the PPSSPP appendix withheld by `REQ-E-01` renders
identically to the WordPress run, because both are real data and only the *need
titles* are fabricated. Invariant 1 in `demo-ui` cannot catch that.

Neither `artifact_dir()` nor `FIRECODE_OUTPUT_DIR` had any test coverage, and
`FIRECODE_OUTPUT_DIR` — the only mitigation that needs no code change — was
undocumented until this rehearsal.

Follows the AST-extraction pattern in `tests/test_app_safety.py`: `app.py` runs
Streamlit calls at import time, so the pure helpers are compiled out of the source
and exercised directly. Offline, no Streamlit runtime.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent

# `artifact_dir` calls these two, so all three are compiled into one namespace and
# the real implementations are exercised rather than stubs.
_HELPERS = ("load_json", "load_manifest", "is_demo_manifest", "artifact_dir")


class _RecordingSidebar:
    """Stands in for `st.sidebar`, recording what the picker was offered."""

    def __init__(self, choose: int = 0) -> None:
        self.offered: list[Any] | None = None
        self.calls = 0
        self._choose = choose

    def selectbox(self, _label: str, options: Any, **_kwargs: Any) -> Any:
        self.calls += 1
        self.offered = list(options)
        return self.offered[self._choose]


class _FakeStreamlit:
    def __init__(self, choose: int = 0) -> None:
        self.sidebar = _RecordingSidebar(choose)


def _app_namespace(default_output: Path, demo_output: Path, st: Any) -> dict[str, Any]:
    """Compile app.py's artifact-selection helpers with injected paths."""

    tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
    nodes = [
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name in _HELPERS
    ]
    assert len(nodes) == len(_HELPERS), f"app.py no longer defines all of {_HELPERS}"
    namespace: dict[str, Any] = {
        "os": __import__("os"),
        "json": json,
        "hashlib": hashlib,
        "Path": Path,
        "Any": Any,
        "DEFAULT_OUTPUT": default_output,
        "DEMO_OUTPUT": demo_output,
        "st": st,
    }
    exec(  # noqa: S102 - compiling our own source, not user input
        compile(ast.Module(body=nodes, type_ignores=[]), "<app-helpers>", "exec"),
        namespace,
    )
    return namespace


def _write_run(directory: Path, *, mode: str) -> Path:
    """A minimally-plausible cached run: what the picker looks for plus a manifest."""

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "top_gaps.json").write_text("[]", encoding="utf-8")
    (directory / "run_manifest.json").write_text(
        json.dumps({"mode": mode}), encoding="utf-8"
    )
    return directory


# --------------------------------------------------------------------------
# FIRECODE_OUTPUT_DIR — the documented way to pin the demo to one run
# --------------------------------------------------------------------------


def test_output_dir_env_pins_the_run_and_never_shows_a_picker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pinning must bypass the picker entirely, not merely preselect it.

    This is the mitigation the demo recipe depends on: with several runs cached,
    a viewer must not be able to select a withheld one.
    """

    demo = _write_run(tmp_path / "demo", mode="demo_fixture")
    out = _write_run(tmp_path / "out", mode="deterministic_baseline")
    _write_run(tmp_path / "out" / "withheld", mode="deterministic_baseline")
    pinned = _write_run(tmp_path / "out" / "primary", mode="deterministic_baseline")

    st = _FakeStreamlit()
    ns = _app_namespace(out, demo, st)
    monkeypatch.setenv("FIRECODE_OUTPUT_DIR", str(pinned))

    chosen, demo_mode = ns["artifact_dir"]()

    assert chosen == pinned
    assert demo_mode is False
    assert st.sidebar.calls == 0, "a pinned run must not offer a picker at all"


def test_bad_output_dir_env_falls_back_to_demo_rather_than_a_real_looking_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A typo in the pin must fail safe.

    The dangerous failure is silently rendering some *other* real run as if it
    were the pinned one. Falling back to the demo fixture is safe because it
    carries the DEMO banner; falling back to a real run would not.
    """

    demo = _write_run(tmp_path / "demo", mode="demo_fixture")
    out = _write_run(tmp_path / "out", mode="deterministic_baseline")

    st = _FakeStreamlit()
    ns = _app_namespace(out, demo, st)
    monkeypatch.setenv("FIRECODE_OUTPUT_DIR", str(tmp_path / "typo-does-not-exist"))

    chosen, demo_mode = ns["artifact_dir"]()

    assert demo_mode is True, "an unusable pin must never resolve to real-looking data"
    assert chosen == demo


# --------------------------------------------------------------------------
# Provenance must come from the manifest, not from the path
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "expected_demo"),
    [("demo_fixture", True), ("deterministic_baseline", False)],
)
def test_provenance_is_read_from_the_manifest_not_the_directory_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, expected_demo: bool
) -> None:
    """A directory called anything at all is classified by its manifest only.

    Regression guard for the shipped bug where demo state was inferred from a
    file being absent, so fixtures copied into `out/` rendered with no banner.
    """

    demo = _write_run(tmp_path / "demo", mode="demo_fixture")
    out = _write_run(tmp_path / "out", mode="deterministic_baseline")
    # Deliberately reassuring name, contradicted by the manifest.
    candidate = _write_run(tmp_path / "out" / "real-production-run", mode=mode)

    ns = _app_namespace(out, demo, _FakeStreamlit())
    monkeypatch.setenv("FIRECODE_OUTPUT_DIR", str(candidate))

    _, demo_mode = ns["artifact_dir"]()

    assert demo_mode is expected_demo


def test_manifest_that_parses_but_is_not_a_mapping_is_treated_as_demo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`is_demo_manifest`'s defensive branch: valid JSON of the wrong shape.

    A manifest that parses to a list rather than an object carries no readable
    `mode`, so provenance is unknown and the safe reading is "demo".
    """

    demo = _write_run(tmp_path / "demo", mode="demo_fixture")
    out = _write_run(tmp_path / "out", mode="deterministic_baseline")
    odd = _write_run(tmp_path / "out" / "odd-shape", mode="deterministic_baseline")
    (odd / "run_manifest.json").write_text("[]", encoding="utf-8")

    ns = _app_namespace(out, demo, _FakeStreamlit())
    monkeypatch.setenv("FIRECODE_OUTPUT_DIR", str(odd))

    _, demo_mode = ns["artifact_dir"]()

    assert demo_mode is True, "unknown provenance must not read as a real run"


def test_a_truncated_manifest_degrades_to_demo_rather_than_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A half-written manifest shows the DEMO banner, not a traceback.

    Reachable in the scenario the demo depends on: a live re-run interrupted
    mid-write leaves truncated JSON. `load_json` guards a *missing* file but
    lets `json.load` raise on a *malformed* one, so provenance is now read via
    `load_manifest`, which degrades to `{}` — unknown provenance, banner shown.
    """

    demo = _write_run(tmp_path / "demo", mode="demo_fixture")
    out = _write_run(tmp_path / "out", mode="deterministic_baseline")
    broken = _write_run(tmp_path / "out" / "broken", mode="deterministic_baseline")
    (broken / "run_manifest.json").write_text('{"mode": "determin', encoding="utf-8")

    ns = _app_namespace(out, demo, _FakeStreamlit())
    monkeypatch.setenv("FIRECODE_OUTPUT_DIR", str(broken))

    _, demo_mode = ns["artifact_dir"]()

    assert demo_mode is True


# --------------------------------------------------------------------------
# The picker itself — characterising the hazard the rehearsal found
# --------------------------------------------------------------------------


def test_picker_offers_every_cached_run_including_withheld_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Documents *current* behaviour, which is why the demo recipe must pin.

    `artifact_dir()` applies no quality gate: a run withheld by decision — the
    PPSSPP appendix under `REQ-E-01`, whose offline need titles are fabricated
    WordPress vocabulary — is offered next to the primary run and renders
    identically, since both are genuinely `deterministic_baseline`.

    If a gate is ever added this test should be updated deliberately rather than
    quietly deleted; that is the point of pinning it.
    """

    demo = _write_run(tmp_path / "demo", mode="demo_fixture")
    out = _write_run(tmp_path / "out", mode="deterministic_baseline")
    _write_run(tmp_path / "out" / "primary", mode="deterministic_baseline")
    _write_run(tmp_path / "out" / "withheld-appendix", mode="deterministic_baseline")

    st = _FakeStreamlit()
    ns = _app_namespace(out, demo, st)
    monkeypatch.delenv("FIRECODE_OUTPUT_DIR", raising=False)

    ns["artifact_dir"]()

    assert st.sidebar.calls == 1
    offered = {Path(p).name for p in (st.sidebar.offered or [])}
    assert {"primary", "withheld-appendix"} <= offered, (
        "every cached run is offered, withheld ones included — pin "
        "FIRECODE_OUTPUT_DIR for the demo"
    )


def test_single_cached_run_needs_no_picker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With one run there is nothing to choose, so no picker should appear."""

    demo = _write_run(tmp_path / "demo", mode="demo_fixture")
    out = _write_run(tmp_path / "out", mode="deterministic_baseline")

    st = _FakeStreamlit()
    ns = _app_namespace(out, demo, st)
    monkeypatch.delenv("FIRECODE_OUTPUT_DIR", raising=False)

    chosen, demo_mode = ns["artifact_dir"]()

    assert st.sidebar.calls == 0
    assert chosen == out
    assert demo_mode is False


# --------------------------------------------------------------------------
# stale_artifact_reasons: malformed and legacy manifests must fail closed.
# --------------------------------------------------------------------------


def _stale_reasons() -> Any:
    tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "stale_artifact_reasons"
    )
    namespace: dict[str, Any] = {"hashlib": hashlib, "ROOT": ROOT, "Any": Any}
    exec(  # noqa: S102 - our own source
        compile(ast.Module(body=[node], type_ignores=[]), "<app-helper>", "exec"),
        namespace,
    )
    return namespace["stale_artifact_reasons"]


def test_manifest_without_reproducibility_block_should_warn() -> None:
    """The docstring claims the gate fails closed; with no block it does not run.

    Not currently reachable from `src/run.py`, which always writes the block —
    but a legacy or hand-authored manifest would render entirely unchecked, and
    a hand-authored artifact has shipped here once before.
    """

    assert _stale_reasons()({"mode": "deterministic_baseline"})


def test_manifest_with_reproducibility_but_no_hashes_should_warn() -> None:
    assert _stale_reasons()({"reproducibility": {"code_version": "abc123"}})


def test_manifest_with_empty_contract_file_list_should_warn() -> None:
    reasons = _stale_reasons()(
        {
            "reproducibility": {
                "inference_contract_sha256": "0" * 64,
                "inference_contract_files": [],
            }
        }
    )
    assert reasons


def test_absent_manifest_also_reads_as_unknown_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run directory with artifacts but no manifest has unverifiable provenance.

    Closed by the same change: previously an absent manifest resolved to `{}` and
    then to "not demo", i.e. a directory with a `top_gaps.json` and no manifest
    rendered as a real run with no banner.
    """

    demo = _write_run(tmp_path / "demo", mode="demo_fixture")
    out = _write_run(tmp_path / "out", mode="deterministic_baseline")
    bare = tmp_path / "out" / "no-manifest"
    bare.mkdir(parents=True)
    (bare / "top_gaps.json").write_text("[]", encoding="utf-8")

    ns = _app_namespace(out, demo, _FakeStreamlit())
    monkeypatch.setenv("FIRECODE_OUTPUT_DIR", str(bare))

    _, demo_mode = ns["artifact_dir"]()

    assert demo_mode is True, "no manifest means provenance is unknown, not real"
