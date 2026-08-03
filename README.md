<p align="center">
  <img src="examples/brand/logo.svg" alt="The Silent Stakeholder" width="520">
</p>

<p align="center">
  Evidence-grounded product intelligence for the users who never enter the planning room.
</p>

<p align="center">
  <a href="https://gev44.github.io/silent-stakeholder/">Live report</a> ·
  <a href="docs/ARCHITECTURE.md">Architecture</a> ·
  <a href="docs/THREAT_MODEL.md">Threat model</a>
</p>

<p align="center">
  <a href="https://github.com/GEV44/silent-stakeholder/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/GEV44/silent-stakeholder/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Python 3.11–3.13" src="https://img.shields.io/badge/Python-3.11–3.13-3776AB?logo=python&logoColor=white">
  <img alt="Offline-first" src="https://img.shields.io/badge/demo-offline--first-0F766E">
</p>

## What it does

The Silent Stakeholder turns app-store reviews and a public GitHub roadmap into a ranked,
inspectable set of product opportunities. It does not stop at topic clustering: every finding
connects a user job to stable signal IDs, the closest roadmap record, an explicit verdict rule,
verified evidence spans, and a reproducible ranking reason.

The result is designed for skeptical product and engineering teams. A reader can challenge a
finding in the language of their own backlog—and trace every claim back to the artifact that
produced it.

### Release demo, at a glance

The committed demo is a deliberately synthetic ExamplePress fixture that exercises the complete
offline pipeline without credentials or redistributed review data.

| Contract | Release result |
|---|---:|
| Synthetic user signals | 26 |
| Roadmap records inspected | 5 |
| Signals assigned to grounded needs | 26 / 26 |
| Verified evidence spans | 26 / 26 |
| Eligible ranked findings | 4 |
| Near-tie treatment | Top two share display band 1; order is deterministic only |
| Calibration claim | None—scores are explicitly uncalibrated |

The command requests five findings. The manifest reports why only four are shown: this configured
clustering lens emitted four verified, non-covered, uniquely titled candidates. The pipeline never
pads a list to hit a target, and the empty fifth slot is not a claim that no alternate split exists.

## Why this project is different

- **Evidence is a contract.** Stable IDs are assigned at ingestion; generated claims may cite only
  IDs supplied to that stage; every quoted span is re-matched against immutable source text.
- **History is not mistaken for delivery.** Closed issues remain searchable disclosure, but cannot
  prove shipped coverage without a verified merged-change and release chain.
- **Confidence stays honest.** The demo exposes an evidence score, not a probability. Calibration is
  enabled only for an independently adjudicated label set that clears the configured sample floor.
- **Model output is bounded.** Optional Gemini stages use structured schemas and allowlisted IDs.
  Invalid citations or failed calls fall back to deterministic behavior and remain visible in
  metadata.
- **Artifacts prove their origin.** The run manifest binds inputs, configuration, inference code,
  and every declared pipeline JSON artifact by SHA-256. The exporter rejects stale or mixed JSON;
  CI separately requires the committed HTML to equal a deterministic render from current code.
- **Public export is privacy-safe.** The default self-contained report publishes synthetic spans or
  real-run IDs and counts only. Restricted evidence requires explicit acknowledgement and an
  outside-repository destination.
- **The demo works offline.** A deterministic hashing embedder and local inference path exercise the
  full evidence, verdict, verification, ranking, and export pipeline without network access.

## Pipeline

```text
reviews + GitHub roadmap
        │
        ▼
ingest ── stable IDs, deduplication, provenance, temporal scope
        │
        ▼
needs ─── clustering, aspect sentiment, JTBD/Kano framing
        │
        ▼
gaps ──── roadmap retrieval, framing comparison, closed-history guard
        │
        ▼
verify ── citation allowlists, exact span checks, critique filter
        │
        ▼
rank ───── evidence score × review-derived priority proxy
        │
        ├── versioned JSON + hash-bound run manifest
        ├── read-only Streamlit explorer
        └── self-contained, CSP-hardened HTML report
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for stage contracts and
[docs/MODEL_CARD.md](docs/MODEL_CARD.md) for intended use and limitations.

## Quick start

Python 3.11–3.13 is supported.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[demo,dev]"

# Regenerate the synthetic release through the real pipeline.
python -m src.run demo --out-dir out/demo

# Explore the committed synthetic artifacts.
python -m streamlit run app.py
```

To pin the explorer to the fresh run:

```bash
# PowerShell
$env:FIRECODE_OUTPUT_DIR = "out/demo"
python -m streamlit run app.py

# bash/zsh
FIRECODE_OUTPUT_DIR=out/demo python -m streamlit run app.py
```

Generate a validated, portable public report:

```bash
python report.py \
  --artifacts examples/demo \
  --out report.html \
  --profile public
```

The exporter validates the manifest declarations and artifact hashes before atomically replacing
the destination. Its Content Security Policy denies network, image, frame, form, object, and base
capabilities; the fixed report script is authorized by hash.

## Running on your own data

Keep review exports local. The following example ingests a local CSV and an authenticated all-state
GitHub snapshot, then runs the deterministic historical analysis:

```bash
export GITHUB_TOKEN="..."  # PowerShell: $env:GITHUB_TOKEN = "..."

python -m src.ingest \
  --app wordpress \
  --reviews-csv /path/to/reviews.csv \
  --github-state all \
  --out-dir out/wordpress

python -m src.run analyze \
  --input-dir out/wordpress \
  --out-dir out/wordpress \
  --mode historical_archive_check \
  --embedding hashing
```

Add `--llm` only when a supported Gemini backend is configured. Run `python -m src.run doctor`
first to inspect the selected backend without exposing credentials.

Real inputs and outputs are intentionally ignored by Git. The review dataset used during
development has no confirmed redistribution license, so this repository publishes only the
unmistakably synthetic ExamplePress fixture. See [docs/DATA_CARD.md](docs/DATA_CARD.md) and
[docs/DATA_LICENSE_SURVEY.md](docs/DATA_LICENSE_SURVEY.md).

## Verification

The release gate is offline and suppresses subprocess output to reduce accidental evidence or
credential leakage.

```bash
python -m pytest -p no:cacheprovider
python -m ruff check --no-cache .
python -m ruff check --no-cache --select S --ignore S311 src scripts report.py
python -m mypy src scripts report.py --no-incremental
python -m compileall -q src tests scripts examples app.py report.py
python scripts/production_gate.py
```

CI runs the suite on Python 3.11, 3.12, and 3.13 with read-only repository permissions. The tracked
pre-commit hook also blocks raw evidence, processed evidence, real-run outputs, model caches, and
answer keys from entering a commit.

## Repository map

```text
src/                 ingestion, inference, matching, confidence, verification, ranking
config/              versioned application and pipeline configuration
examples/demo/       publishable synthetic inputs and generated release artifacts
tests/               unit, integration, determinism, provenance, privacy, and security tests
docs/                architecture, governance, data/model cards, and live report
scripts/              offline production and reviewer-packet gates
app.py                read-only Streamlit artifact explorer
report.py             validated self-contained HTML exporter
```

## Scope and limitations

- Public GitHub issues are a planning proxy, not proof of internal intent.
- App-store reviews are one evidence channel; cross-source corroboration requires additional inputs.
- Deterministic offline naming uses explicit product-domain vocabulary and is less semantic than the
  optional structured model path.
- Review-derived opportunity scores are prioritization proxies, not survey-validated ODI measures.
- No calibrated probability is shown without a qualifying independent human-label set.
- Need boundaries reflect one deterministic clustering lens. Closely related symptoms may be grouped;
  the demo makes no alternate-clustering omission claim.
- Scores within 1% of a display band's highest score share that band. Within-band order is a
  reproducible tie-break; crossing a band boundary is arithmetic, not validated priority separation.

These limits are carried into generated manifests and reports so they remain visible at the point
of decision—not buried in a methodology appendix.

## License

Copyright © 2026 GEV44. All rights reserved. This repository is publicly viewable but is not
open-source licensed; see [LICENSE](LICENSE).
