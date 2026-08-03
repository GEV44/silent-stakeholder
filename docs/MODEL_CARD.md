# Model card

## System

Firecode is a staged research system, not one model. It combines deterministic ingestion and
validation, local text representations, optional clustering, and optional Gemini structured
inference. Every generative output remains subordinate to immutable evidence records.

## Intended use

- exploratory analysis of user-signal themes against a timestamped public-planning proxy;
- historical public-archive matching when all GitHub states are captured;
- current opportunity analysis only when the user evidence is recent;
- hackathon demonstration of evidence-grounded product research.

It is not intended to infer a team's private roadmap, intent, negligence, customer prevalence, or
business value without additional evidence.

## Generative configuration

- default model: `gemini-3.5-flash`;
- serving location: `global`;
- API version: `v1`;
- SDK pin: `google-genai==2.10.0`;
- model, SDK, sampling count, random seed, pipeline-config hash, input hashes,
  and a hash of the prompt/schema adapter source are recorded per analytical run;
- 3.5 is selected because the extraction-stability experiment uses sampling controls that 3.6
  currently ignores.

Run `python -m src.run doctor --live` after credentials are configured. It performs one bounded
structured-output smoke test without storing the response content.

## Deterministic baseline

The hashing/KMeans fallback exists for tests, data profiling, and offline demonstrations. Its need
titles are heuristic frames and must not be presented as production research findings. It does,
however, exercise stable IDs, matching, evidence scoring, quote validation, cautious public states,
and ranking.

The offline demo provides one deterministic clustering lens, not an omission study across alternate
clusterings. Its uncalibrated ordering groups scores within 1% into a shared display band. The band
exposes near-ties; neither within-band order nor cross-band position is a validated separation claim.

## Known failure modes

- clusters can combine distinct workflows or split one workflow into duplicates;
- JTBD, Kano, root-cause, and “partial coverage” are hypotheses;
- historical review language may not represent current users;
- GitHub public artifacts omit private planning;
- self-consistency measures output stability, not correctness;
- structured output prevents malformed IDs but not a semantically weak interpretation.

## Evaluation gates

Evaluate need support, public-artifact match, and final claim defensibility separately. Use two
reviewers plus adjudication, product/time-held-out tests, exact quote validation, threshold
sensitivity, and cluster stability. No displayed percentage is allowed without independently
evaluated calibration.
