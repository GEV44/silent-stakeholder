# Live defense playbook

## “Why is this ranked first?”

Open the gap and walk through, in order: the latent JTBD, distinct signal count, exact quotes, closest
roadmap match, deterministic verdict inputs, opportunity score, calibrated or uncalibrated confidence,
and final multiplication. Do not start with the model's prose.

## “Defend 71%.”

Only use a percentage when it came from held-out/cross-fitted labels. Show label count, reliability
diagram, ECE, Brier score, and the bin containing 0.71. If the run lacks those labels, say:
“This is an uncalibrated ranking score, not a 71% probability.”

## “You missed gap X.”

Search cached needs and roadmap matches. Show its rank and the specific mechanism that lowered it:
weak support, contradiction, roadmap coverage, low opportunity, failed quote verification, or
calibration. If it was never generated, record it as a recall failure; do not improvise evidence.

## “You just counted complaints.”

Show two reviews with different wording in the same cluster, the extracted symptom, and the separate
JTBD/Kano representation. Then show that frequency is only one confidence feature and cannot create a
gap without roadmap comparison and verified support.

## “How do you know the roadmap ignores it?”

Do not claim knowledge of internal intent. Say exactly which public sources and time scope were
searched, show the closest open and closed items, and phrase the result as “no material public roadmap
evidence found in this scope.”

## “The reviews are from 2016–2017.”

Agree with the limitation. The primary claim is a historical-to-archive analysis, not current market
research. Full current-product conclusions require recent signals. The pipeline is source-agnostic;
the demo corpus proves the method, not present-day prevalence.

## “Could the LLM hallucinate this?”

The prose can still be imperfect, but it cannot introduce a valid evidence trace unnoticed: allowed
IDs are constrained, IDs are revalidated, quotes are checked against immutable text, and unsupported
items are removed before ranking.

