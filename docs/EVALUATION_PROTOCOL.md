# Evaluation and calibration protocol

## What humans label

Build a stratified set of need–roadmap pairs across high, medium, and low similarity. Two reviewers
independently label:

1. whether the need is supported by its cited signals;
2. whether the roadmap contains material coverage;
3. the verdict (`IGNORED`, `UNDER_PRIORITIZED`, `MISUNDERSTOOD`, `COVERED`);
4. whether the evidence is sufficient for a public claim.

Disagreements are adjudicated and retained as an ambiguity flag. Report Cohen's kappa before
adjudication.

## Avoiding leakage

Do not tune similarity thresholds, fit the calibrator, and report calibration on the same examples.
For a small hackathon set, use repeated stratified cross-fitting and report out-of-fold predictions.
For a credible final evaluation, reserve a product- or time-held-out test set.

Project release gates:

- under 30 labels: no probability claim; show an uncalibrated score;
- 30–99 labels: pilot error analysis only;
- 100–999 labels: cross-fitted Platt scaling with a large uncertainty warning;
- 1,000+ labels: compare Platt and isotonic by out-of-fold proper scoring rules;
- subgroup reliability requires adequate examples in every reported subgroup.

These are project policies, not universal statistical guarantees.

## Metrics

- Evidence support precision: unsupported top gaps are the most damaging failure.
- Roadmap coverage recall: measures missed “already covered” items.
- Verdict macro F1: prevents `COVERED` or `IGNORED` prevalence from hiding errors.
- Brier score: proper scoring rule for probability quality.
- ECE: descriptive reliability summary, always shown with bin counts.
- Precision@5: judge-facing quality of the final ranked list.
- Evidence trace validity: exact-ID and quote validation rate, target 100%.

Bootstrap the held-out predictions to show 95% intervals for Brier, ECE, and precision@5.

## Temporal scope

The bundled review corpus is historical. A current GitHub snapshot cannot reconstruct a historical
roadmap perfectly. Every run must declare one of these scopes:

- `exploratory_snapshot`: method development only; no current or historical outcome claim;
- `historical_archive_check`: search open and closed GitHub items today for evidence that a
  historical need has a public archive match;
- `current_opportunity`: use recent evidence against a timestamped current public-planning proxy;
- `demo_fixture`: synthetic, contract-testing only.

Never describe `current open issues only` as proof that a 2016 review need was ignored.

## Label file

Copy `eval/dev_labels.example.json` to an ignored working file, replace the examples with real,
double-reviewed labels, and record reviewer/version metadata. The pipeline rejects calibration when
the file does not meet the configured minimum or contains unresolved IDs.
