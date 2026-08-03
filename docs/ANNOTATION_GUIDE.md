# Annotation guide

## Unit of review

One annotation packet contains a latent need, its cited user signals, the closest public planning
artifact, the proposed public state, and the declared time mode. Review source text, not only the
generated summary.

## Independent labels

1. `need_supported`: do the cited signals materially support the latent need?
2. `public_artifact_match`: `none`, `partial`, or `material`.
3. `public_claim_defensible`: can this claim be shown publicly under the declared scope?
4. `verdict`: internal gate label, retained for model evaluation.
5. `notes`: concise reason, including contradictions or missing evidence.

For closed issues, do not label “shipped” without a linked merged change and release/tag evidence.
For missing matches, label only absence from the inspected public sources.

## Process

- Two reviewers label independently.
- Reviewers cannot see the model's evidence score.
- Calculate agreement before discussion.
- Adjudicate disagreements and set `adjudicated=true`.
- Never overwrite an original reviewer label; append the adjudicated record.
- Split by product or time before threshold tuning/calibration.

## Quality examples

- Several reviews say images stall at 99%, while the closest issue only adds a progress indicator:
  `need_supported=true`, `public_artifact_match=partial`.
- A generated need cites mixed praise and unrelated login failures:
  `need_supported=false`.
- No open issue matches, but a closed issue and release show implementation:
  `public_artifact_match=material`; it is not a current gap.

The machine-readable contract is `eval/labels.schema.json`.

