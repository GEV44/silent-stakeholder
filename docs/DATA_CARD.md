# Data card

## User signals

The planned source is `sealuzh/app_reviews` on Hugging Face. Its published card reports 288,065
English Android reviews with `package_name`, `review`, `date`, and `star` fields. The dataset is
historical and its card currently lists the license as unknown.

Verified 2026-07-31 against the primary sources, not the card prose: the Hugging Face
`cardData.license` is the literal value `unknown`, and the upstream repository
`github.com/sealuzh/user_quality` publishes **no LICENSE file at all**. The dataset originates with
Grano, Di Sorbo, Mercaldo, Visaggio, Canfora & Panichella (2017), archived at
`zora.uzh.ch/id/eprint/139426`. The only route to resolving the terms is asking those authors.

Consequences:

- Do not redistribute the full dataset in this repository.
- Cache it locally under `data/raw/`, which is ignored by Git.
- Treat quoted reviews as potentially personal or sensitive user content.
- Minimize quotes in reports and avoid usernames or provider identifiers.
- Disclose the historical date range in every result.
- Do not generalize Android app-review findings to current enterprise customers.
- **Do not accept a permissively-tagged mirror as clearance — see below.**

### ⚠️ A laundered MIT copy of this corpus exists

`Sharathhebbar24/app_reviews_modded` on Hugging Face is tagged `license: mit` and contains
**exactly 288,065 rows** with the same four columns plus one addition. Row counts for both were read
from the same `datasets-server/size` endpoint, so the comparison is exact: it is a re-upload of the
unknown-license corpus above.

**An uploader cannot grant MIT on data they do not hold the rights to.** That tag is an assertion
about someone else's data, not a license. Acting on it would let us publish unlicensed review text
while believing the boundary had been cleared — a worse position than being knowingly blocked.

Every other permissively-tagged app-review dataset checked was unusable on the merits (empty,
single-column, or covering only closed-source apps with no public roadmap to diff against), and the
one genuinely MIT-licensed review corpus we found reviews retail companies, so it has no
`package_name` and cannot support the review↔roadmap join. Full survey:
[`DATA_LICENSE_SURVEY.md`](DATA_LICENSE_SURVEY.md).

**Check before trusting any permissive tag:** if the row count matches a known unlicensed upstream,
it is that upstream.

## Roadmap data

Roadmap records come from GitHub's public REST API. The project inspects milestones, issues, labels,
timestamps, and issue relationships available in the configured repository.

GitHub limitations:

- Issues are not a complete internal roadmap.
- State and edited text are current snapshots unless reconstructed from event history.
- The issues endpoint also returns pull requests; ingestion removes them.
- Missing public evidence is not evidence of deliberate neglect.
- Full closed-issue history may require authenticated pagination.

## Demo fixtures

Committed fixtures are synthetic and exist to make tests and the judge-facing UI reproducible without
network access. They must be visibly labeled `DEMO DATA` and must never be presented as research
findings.

## Provenance requirements

Every output records source URLs, retrieval time, record counts, filters, time scope, and hashes of
the canonical input files. A published claim must be reproducible from the retained cached inputs.

