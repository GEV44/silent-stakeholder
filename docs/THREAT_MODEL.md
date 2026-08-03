# Threat model

## Protected assets

- GitHub and Google credentials
- unredacted user-review text
- integrity of evidence IDs and quotes
- calibration labels and metrics
- cached judge-facing outputs

## Main risks and controls

| Risk | Control |
|---|---|
| Secret committed or logged | environment-only credentials, `.gitignore`, startup redaction, secret scan |
| Prompt injection inside a review/issue | source text is inert quoted data; model has no tools; schema-constrained output |
| Invented evidence ID | dynamic ID enum plus code-side subset validation |
| Fabricated or altered quote | normalized exact/fuzzy substring verification against immutable input |
| PR counted as issue | reject API objects containing `pull_request` |
| Duplicate reviews inflate confidence | normalized-text and optional semantic deduplication before feature counts |
| Current roadmap compared as historical truth | explicit temporal scope and full open/closed archive check |
| In-sample calibration looks perfect | cross-fitted predictions only; label count and uncertainty shown |
| Stale/tampered cache | source hashes and schema versions in artifact metadata |
| Public-roadmap absence overclaimed | wording policy: “no evidence found in inspected scope” |

## Operational checklist

Before a demo, rotate any credential previously pasted into a chat, run the tests and secret scan,
freeze cached artifacts, record GitHub rate-limit state, and verify the app works with networking
disabled.

