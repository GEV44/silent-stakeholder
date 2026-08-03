# Data & prior-art survey — can we license our way out of the untracked-data problem?

**Surveyed:** 2026-07-31 by block-e. **Method:** Hugging Face API + datasets-server
(authoritative license tags and row counts, not card prose), GitHub API, and the
primary papers. Every claim below was checked against a primary source per
`research/CLAUDE.md` rule 3.

**Answer: no.** No permissively-licensed corpus exists that supports this
project's review↔GitHub-roadmap join. Our current position — untracked `data/`,
gitignored `out/`, a documented ingest recipe, and a synthetic committed fixture
— is not a compromise we settled for. It is the only defensible position
available, and that is now evidenced rather than assumed.

Three findings are actionable. One is a trap that could actively hurt us.

---

## 1. Our own corpus: license genuinely unresolved, confirmed at the source

| source | license |
|---|---|
| HF `sealuzh/app_reviews` — `cardData.license` and tag | **`unknown`** (literal value) |
| upstream `github.com/sealuzh/user_quality` | **no LICENSE file** |

Neither is an oversight we can read past — the HF card has `license: unknown`
explicitly, and the upstream repo publishes no terms at all. The dataset comes
from Grano, Di Sorbo, Mercaldo, Visaggio, Canfora & Panichella (2017), archived
at ZORA (`zora.uzh.ch/id/eprint/139426`). **The only way to resolve this is to
email the authors** — a real option worth taking if anyone wants the boundary
lifted, and the only one that actually works. Everything else below is a
workaround.

---

## 2. ⚠️ The trap: an MIT-licensed copy of our corpus exists, and it is laundered

`Sharathhebbar24/app_reviews_modded` is tagged **`license: mit`** on Hugging
Face. It is a derivative of the unknown-license original:

| | rows | columns |
|---|---|---|
| `sealuzh/app_reviews` | **288,065** | `package_name, review, date, star` |
| `Sharathhebbar24/app_reviews_modded` | **288,065** | same four **+ `products`** |

Row counts measured from the same `datasets-server/size` endpoint, so this is
apples-to-apples: **identical to the row, plus one added column.**

**An uploader cannot grant MIT on data they do not hold the rights to.** That tag
is not a license; it is an assertion someone made about someone else's data. If a
teammate finds it and concludes "there's an MIT version, we can commit the data
now," we would publish unlicensed review text *while believing we were clear* —
strictly worse than today, because today we at least know we're blocked.

**If anyone proposes committing data on the strength of a permissive tag, check
whether the row count matches an unlicensed upstream first.**

## The other permissively-tagged app-review datasets are unusable on the merits

| dataset | license | why it fails |
|---|---|---|
| `AsmaaQ/app_reviews` | apache-2.0 | **0 rows** — empty |
| `RoamingFox/App_Reviews` | apache-2.0 | 7,891 rows, single `example` column, no app identifier |
| `NovaNightshade/AndroidAppReviews` | mit | **1 row**; columns are 26 app *names* (DoorDash, TikTok, Spotify…) — a pivoted junk upload, and all closed-source, so no GitHub roadmap to join |
| `UniqueData/messengers-reviews-google-play` | cc-by-**nc-nd**-4.0 | ND forbids derivatives — our whole pipeline is a derivative |

---

## 3. The two genuinely-licensed datasets in our brief, and why neither helps

**`Kerassy/trustpilot-reviews-123k` — MIT, genuine.** Columns are `category,
company, description, title, review, stars`. Sampled content is consumer retail:
dog drying coats, sofa covers. **No `package_name`, no app id, no GitHub repo** —
the reviewed entities are retail businesses with no public roadmap. The license is
real and useless to us. (Caveat worth noting even so: it is scraped Trustpilot
content, and an uploader's MIT tag does not obviously clear Trustpilot's ToS or
the reviewers' own copyright — the same reasoning that condemns the laundered copy
above applies here in weaker form.)

**`Tobi-Bueck/customer-support-tickets` — `cc-by-nc-4.0`.** Locally usable for a
hackathon (non-commercial, attribution is cheap) and we would not redistribute it
anyway. But **do not import it to fix `source_diversity`.**

That deserves spelling out, because the temptation is concrete. `div` is
**0.0 on every gap in every run** — the lead's own note says *"`div` needs more
than one signal source and we have one … both come alive the moment we add a
source."* Adding an unrelated synthetic ticket corpus would indeed make the
number non-zero. It would not make it *mean* anything: the tickets are not
WordPress users, so a diversity score computed across them measures nothing about
whether a WordPress need is corroborated across channels. That is feature
theater, and it is precisely the class of move this project has rejected
everywhere else (`REQ-main-4`'s retracted calibration claim, REQ-C-02's
"do not hand-curate a second corpus"). **`div = 0.0` with a one-line explanation
is the stronger answer.**

If a *legitimate* second source is wanted, the honest candidate is
**user-filed GitHub issue bodies and comments** — already fetched, public,
no redistribution question. The hazard is circularity: issues are currently the
roadmap side of the diff, so using them as user signal too needs a defensible
split (e.g. non-maintainer authors only) or it becomes self-confirming. Worth a
design discussion; not something to bolt on before a demo.

---

## 4. Prior art: the closest published work uses our exact two sources and does *not* do what we do

**"Can GitHub Issues Help in App Review Classifications?"** — Alizadeh et al.,
ACM TOSEM 2024 ([arXiv:2308.14211](https://arxiv.org/html/2308.14211v3),
[ACM](https://dl.acm.org/doi/full/10.1145/3678170)).

It joins **app-store reviews with GitHub issues** — the same two sides we join —
using 62.7K processed issues from 999 Android repositories to augment training
data for review classification. Findings: +6.3 F1 on bug reports, +7.2 on feature
requests; within-app and within-context augmentation beat random.

**It classifies review intent. It does not compare user needs against a roadmap.**
Verified explicitly: no roadmap-alignment or gap analysis, no verdict over whether
planned work addresses what users need. Its notable limitation is also directly
relevant to us: the "Other" class *"poorly represents emotional content (praise,
complaints) present in reviews but absent in formal GitHub issues"* — i.e. the
signal our praise-embedded-friction lens is built to catch.

Two things follow:

- **A citable novelty claim.** *"The closest published work joins these exact two
  sources and stops at classifying what users said. We diff it against what the
  team planned and return a verdict."* That is a stronger answer to "hasn't this
  been done?" than a generic literature gesture.
- **Its replication package is not a data source for us.**
  `ISE-Research/App-Reviews-Augmentation` — checked via GitHub API: **no
  license**, 928 KB, top level is `README.md` + `src/` only. No redistributable
  corpus, and no license if there were.

A GitHub repository search for review-vs-roadmap gap analysis returned only
keyword-matched noise (awesome-lists, courses) — no comparable open-source project
surfaced.

Also relevant, unresolved: general guidance on scraped Play Store review data is
*use for analysis, not redistribution*, and Play ToS/copyright questions are
genuinely unsettled for research redistribution. That is a second, independent
reason the boundary stands even if the ZORA authors were to reply permissively
about their own collection.

---

## 5. What was added to `data/`

**Nothing.** Deliberately. Each candidate fails on at least one of: license
(laundered or ND), joinability (no app id → no GitHub roadmap), or
methodological honesty (unrelated corpus imported to move a metric). Downloading
any of them would add clutter and, in the MIT case, active risk.

Already present locally from earlier this session and *staying* untracked:
`data/processed/wordpress-open/` (2,974 signals / 774 roadmap items) and
`data/processed/ppsspp-open/` (8,316 / 1,320), both regenerable via the
`README.md` recipe.

## 6. The one real unblock, and it needs no new data

The blocker is not the *analysis* — it is that `out/` artifacts embed **verbatim
review spans**. `CLAUDE.md` says exactly that: *"processed signals and evidence
traces still contain verbatim review text."*

Copyright attaches to that expression, not to facts about it. Verdicts, coverage
percentages, similarity scores, priority reasons, feature vectors, signal **IDs**,
and model-authored need/JTBD statements are not the reviews. A **quote-stripped
artifact variant** — every field except `evidence.quotes[].span` and
`Signal.text` — would be committable, and would let us ship real, inspectable,
reproducible output instead of asking judges to take a local run on trust. Filed
as **`REQ-E-04`** (needs a lane block-e does not own).

## Sources

- https://huggingface.co/datasets/sealuzh/app_reviews · https://github.com/sealuzh/user_quality · https://www.zora.uzh.ch/id/eprint/139426/
- https://huggingface.co/datasets/Sharathhebbar24/app_reviews_modded · https://huggingface.co/datasets/Kerassy/trustpilot-reviews-123k · https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets
- https://arxiv.org/html/2308.14211v3 · https://dl.acm.org/doi/full/10.1145/3678170 · https://github.com/ISE-Research/App-Reviews-Augmentation
- https://www.kaggle.com/datasets/dmytrobuhai/play-market-2025-1m-reviews-500-titles — license **not verified**; the page is JS-rendered and its static HTML exposes no license field. Treat as unknown until someone checks it logged in.
