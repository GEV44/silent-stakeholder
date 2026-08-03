# Cost and quota budget

Firecode does not assume one universal RPM or “tier.” Gemini Developer API and managed Vertex
serving have different quota models. Read the selected surface's current limits and keep retries and
concurrency below the allocation.

## Bounded call plan

With the default maximum of 40 needs:

- need extraction: up to 40 clusters × 5 stability samples = 200 calls;
- ambiguous gap adjudication: at most 40 calls;
- adversarial critique: at most 40 calls;
- startup smoke test: 1 call.

Hard planning ceiling: **281 generation calls per uncached full run**. The actual count is lower when
there are fewer clusters or deterministic verdicts. Responses are cached by model, prompt, schema,
and sample index.

## Release controls

- record request, retry, cache-hit, prompt-token, and output-token counts;
- stop before the configured call/token budget rather than silently overspending;
- use bounded exponential backoff for retryable 429/5xx responses;
- rehearse from frozen cached outputs;
- do not make live inference a dependency of the five-minute demo.

Prices and quotas are intentionally not hardcoded in this repository because they change. Record the
official price sheet and quota snapshot in the run notes when producing a budget.

