# Architecture

## System objective

The Silent Stakeholder ranks three to five latent user needs that are absent, parked, or only
partially addressed in an inspected public roadmap. A result is releasable only when a reviewer can
walk from the verdict and rank back to immutable signal and roadmap records.

The repository follows two architectural rules:

1. analytical stages communicate through inspectable, versioned JSON artifacts; and
2. a generated claim may reference only allowlisted IDs from the exact input batch supplied to that
   stage.

## Data flow

```text
local review export ─┐
                     ├─ ingest ──> signals.json ──> needs ──> needs.json ─┐
GitHub API/fixture ──┘             roadmap.json ───────────────────────────┤
                                                                           ▼
                                                                    gap matching
                                                                           │
                                                                           ▼
human labels ──> optional calibration ──> confidence ──> verify ──> rank
                                                          │          │
                                                          │          └─> top_gaps.json
                                                          └─> verification.json

all stage outputs ──> SHA-256 declarations ──> run_manifest.json
```

The Streamlit explorer and standalone HTML exporter are read-only artifact consumers. They do not
run ingestion or mutate evidence.

## Stage contracts

### 1. Ingestion

`src/ingest.py` normalizes review rows and GitHub issues/milestones.

- Signals and roadmap records receive deterministic IDs after canonical sorting.
- Text-and-rating duplicates collapse without erasing contradictory ratings.
- Pull requests returned by GitHub's issues endpoint are removed.
- Fixture provenance (`repository`, retrieval time, API version, state scope) is preserved; processing
  time is recorded separately.
- Offline mode rejects every network source before a request is attempted.
- Raw and processed evidence directories are excluded from Git and blocked by the tracked hook.

### 2. Need inference

`src/needs.py` groups friction-bearing signals and produces a JTBD statement, symptom, Kano class,
and review-derived priority proxy.

The offline path uses deterministic hashing embeddings and clustering. Its product-domain vocabulary
is explicit: an off-domain cluster is dropped instead of receiving a fabricated product label.
Signals from a neighboring product area can be retrieval candidates, but they cannot support an
offline need unless their vocabulary grounds that frame. Small, below-threshold fragments attach only
when they map unambiguously to one existing offline need.

The optional Gemini extractor uses structured output and an enum of allowed signal IDs. Invalid IDs,
malformed JSON, or failed calls cannot become evidence; the fallback outcome and failure count are
recorded.

### 3. Roadmap comparison

`src/gaps.py` retrieves the closest roadmap candidates with embeddings, then applies auditable gates:

```text
distinctive symptom vocabulary covered, job vocabulary missing  -> MISUNDERSTOOD
similarity below the low gate                                    -> IGNORED
matching item is explicitly or implicitly parked                 -> UNDER-PRIORITIZED
high similarity plus committed priority                          -> COVERED
ambiguous band                                                    -> deterministic midpoint or LLM adjudication
```

The LLM may adjudicate only the ambiguous band and cannot invent a deterministic
`MISUNDERSTOOD`. A closed issue can be searched and cited, but closure alone never proves that a fix
shipped; a would-be `COVERED` match is retained as disclosure-only `IGNORED` unless a future normalized
merged-change and release chain exists.

Each gap records the governing thresholds, framing probes, priority reasons, matched roadmap ID,
decision mechanism, and sensitivity to the gate that actually produced the verdict.

### 4. Confidence and calibration

`src/confidence.py` computes an inspectable evidence score from evidence volume, diversity,
consistency, cohesion, method agreement, and model self-consistency. Sentiment and recency are
priority/current-relevance diagnostics rather than evidence-quality rewards.

The score becomes a probability only after cross-fitted calibration against an independently
adjudicated label set that clears the configured minimum. Platt scaling is used for modest sets;
isotonic calibration is unavailable below 1,000 labels. Without a qualifying receipt, artifacts say
`uncalibrated_evidence_score_not_probability`.

### 5. Verification and ranking

`src/verify.py` resolves every cited ID against the current source artifact and re-matches every span.
Unsupported gaps are filtered before `src/rank.py` orders candidates. The uncalibrated release demo
uses:

```text
evidence score × review-derived priority proxy
```

For uncalibrated output, candidates within 1% of the highest score in a band share that display
band. Their numeric order is retained for deterministic serialization. Bands expose near-ties;
neither within-band order nor cross-band position is presented as validated priority separation.

The requested top-k is a ceiling, not a quota. The manifest reports a shortfall and reason rather
than padding the list with covered, unsupported, or duplicate-title candidates.

## Provenance and report integrity

`src/run.py` writes a manifest containing:

- input artifact hashes;
- pipeline configuration hash;
- the combined hash and file list for every module that can affect candidates, scores, verdicts, or
  ordering;
- exact hashes for every declared pipeline JSON artifact consumed by the report;
- code version, temporal/roadmap scope, observed model-call outcome, counts, and limitations.

`report.py` refuses missing, malformed, duplicate-key, oversized, symlinked, undeclared, or
hash-mismatched inputs. The public profile is an allowlisted projection. It withholds real review
spans, writes through a same-directory temporary file, fsyncs, and atomically replaces the
destination. Its CSP denies external capabilities and authorizes the fixed script by hash.
Because the HTML embeds the manifest, the manifest cannot also hash that HTML without a cycle.
Instead, CI renders the report from current exporter code plus the declared JSON and requires exact
equality with `docs/index.html`.

## Runtime and trust boundaries

- Network access is isolated to explicit ingestion and optional LLM paths.
- Credentials come only from environment variables and are never serialized.
- Cached inputs keep analysis and the synthetic demo offline-capable.
- Public GitHub records are a planning proxy, not proof of private organizational intent.
- Real evidence and real-run outputs are local-only; the repository distributes only the synthetic
  ExamplePress fixture.
- CI uses read-only repository permissions and runs tests, Ruff, security lint, mypy, Pylint,
  compilation, dependency integrity, and protected-file policy on Python 3.11–3.13.
