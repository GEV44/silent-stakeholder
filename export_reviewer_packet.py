"""Produce blind, private labelling packets — and merge them back safely.

Calibration needs two independent humans. Code cannot supply that, and inventing
labels would be worse than shipping uncalibrated. What code *can* do is make the
human step safe to run, which is what this does.

Three contamination routes are closed here, not one:

1. **The answer key.** `eval/_answer_key.DO_NOT_OPEN_BEFORE_LABELING.json` is
   never read by this script and is asserted absent from every packet.
2. **The model's own judgment.** `roadmap_priority` is a *derived* verdict
   (`priority_is_low`), it primes a reviewer straight to UNDER-PRIORITIZED, and
   it is `low` for 99.5% of the corpus anyway. Raw evidence — title, body,
   labels, milestone — is kept; the model's conclusion is stripped.
3. **The other reviewer.** Each packet contains only its own answer slots, so
   reviewer B cannot anchor on reviewer A.

Packets default to a directory **outside the repository**, because a labelling
packet that lands in the working tree is one `git add -A` away from being
published alongside the corpus it quotes.

    python export_reviewer_packet.py export --reviewers alice,bob
    python export_reviewer_packet.py check  --packets ~/silent-stakeholder-labeling
    python export_reviewer_packet.py merge  --packets ~/silent-stakeholder-labeling

Nothing here makes the product calibrated. It stays honestly uncalibrated until
two humans finish and the count clears the configured minimum.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent
WORKLIST = REPO / "eval" / "labeling_worklist.json"
SCHEMA = REPO / "eval" / "labels.schema.json"

# Named so the guard can assert it never reaches a packet. Never opened.
ANSWER_KEY_NAME = "_answer_key.DO_NOT_OPEN_BEFORE_LABELING.json"

DEFAULT_OUT = Path.home() / "silent-stakeholder-labeling"

# Kept: raw evidence a reviewer must weigh for themselves.
REVIEWER_VISIBLE = (
    "pair_id",
    "need_id",
    "latent_need",
    "jtbd",
    "symptom",
    "roadmap_id",
    "roadmap_title",
    "roadmap_body",
    "roadmap_labels",
    "roadmap_milestone",
)

# Stripped, with the reason recorded in the packet manifest so the blinding is
# auditable rather than asserted.
WITHHELD = {
    "roadmap_priority": (
        "model-derived priority judgment; primes UNDER-PRIORITIZED and is 'low' "
        "for 99.5% of the corpus"
    ),
    "reviewer1": "another reviewer's answer slot",
    "reviewer2": "another reviewer's answer slot",
    "adjudicated_verdict": "the answer being sought",
    "adjudicated_notes": "the answer being sought",
}

ANSWER_SLOT = {
    "need_supported": None,
    "public_artifact_match": None,
    "public_claim_defensible": None,
    "verdict": None,
    "notes": "",
}

VERDICTS = ("IGNORED", "UNDER-PRIORITIZED", "MISUNDERSTOOD", "COVERED")
MATCH_LEVELS = ("none", "partial", "material")
MIN_CALIBRATION_LABELS = 100  # mirrors src/run.py --min-calibration-labels


def _fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(2)


def _load_worklist() -> list[dict[str, Any]]:
    if not WORKLIST.exists():
        _fail(f"{WORKLIST} not found")
    payload = json.loads(WORKLIST.read_text(encoding="utf-8"))
    pairs = payload.get("pairs")
    if not pairs:
        _fail("worklist has no 'pairs'")
    return pairs


def _guard_destination(out: Path, force: bool) -> None:
    """Refuse to scatter reviewer packets through the working tree."""

    resolved = out.resolve()
    if resolved == REPO or REPO in resolved.parents:
        if not force:
            _fail(
                f"{resolved} is inside the repository. A packet here can be committed "
                "alongside the corpus it quotes. Choose a path outside the repo, or "
                "pass --force if you have confirmed it is gitignored."
            )
        print(f"WARNING: writing inside the repository at {resolved} (--force).")
        print("         Confirm it is gitignored before committing anything.")


def _blind_item(pair: dict[str, Any]) -> dict[str, Any]:
    item = {key: pair.get(key) for key in REVIEWER_VISIBLE}
    item["answer"] = dict(ANSWER_SLOT)
    return item


def _assert_clean(items: list[dict[str, Any]]) -> None:
    """Prove the blinding rather than trusting the field list."""

    leaked = sorted({key for item in items for key in item if key in WITHHELD})
    if leaked:
        _fail(f"packet would leak withheld fields: {', '.join(leaked)}")
    blob = json.dumps(items)
    if ANSWER_KEY_NAME in blob:
        _fail("packet references the answer key")


def export(args: argparse.Namespace) -> int:
    reviewers = [name.strip() for name in args.reviewers.split(",") if name.strip()]
    if len(reviewers) < 2:
        _fail("at least two reviewers are required; one reviewer is not independent labelling")
    if len(set(reviewers)) != len(reviewers):
        _fail("reviewer names must be distinct")

    pairs = _load_worklist()
    out = args.out
    _guard_destination(out, args.force)
    out.mkdir(parents=True, exist_ok=True)

    if (out / ANSWER_KEY_NAME).exists():
        _fail(f"{out} already contains the answer key. Remove it before labelling.")

    for index, reviewer in enumerate(reviewers):
        items = [_blind_item(pair) for pair in pairs]
        # Independent order per reviewer: identical ordering lets two people drift
        # into the same rhythm and correlate their mistakes.
        random.Random(f"{args.seed}:{reviewer}").shuffle(items)
        _assert_clean(items)

        packet = {
            "schema_version": "1.0",
            "reviewer": reviewer,
            "generated_at": datetime.now(UTC).isoformat(),
            "pair_count": len(items),
            "instructions": [
                "Answer from the evidence shown. Do not open the repository's",
                "answer key, and do not discuss pairs with the other reviewer.",
                "need_supported: does the cited need follow from its own evidence?",
                "public_artifact_match: none | partial | material",
                "public_claim_defensible: would you defend this publicly?",
                f"verdict: one of {', '.join(VERDICTS)}",
                "Record uncertainty in notes rather than guessing.",
            ],
            "withheld_from_this_packet": WITHHELD,
            "items": items,
        }
        path = out / f"packet_{reviewer}.json"
        path.write_text(json.dumps(packet, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path}  ({len(items)} pairs, order seed {args.seed}:{reviewer})")
        if index == 0:
            _write_worksheet(out, reviewer, items)

    (out / "README.txt").write_text(_readme(reviewers, len(pairs)), encoding="utf-8")
    print(f"wrote {out / 'README.txt'}")

    print()
    if len(pairs) < MIN_CALIBRATION_LABELS:
        print(f"NOTE: {len(pairs)} pairs is below the {MIN_CALIBRATION_LABELS}-label minimum.")
        print("      Completing these produces inter-rater agreement and error analysis,")
        print("      NOT a calibrated probability. The product stays uncalibrated.")
    print("The answer key was never read by this command and is not in the packets.")
    return 0


def _write_worksheet(out: Path, reviewer: str, items: list[dict[str, Any]]) -> None:
    """A readable worksheet for whoever would rather not edit JSON."""

    lines = [f"# Labelling worksheet — {reviewer}", ""]
    for number, item in enumerate(items, start=1):
        lines += [
            f"## {number}. {item['pair_id']}",
            "",
            f"**Need** ({item['need_id']}): {item['latent_need']}",
            f"- job: {item['jtbd']}",
            f"- symptom: {item['symptom']}",
            "",
            f"**Roadmap item** ({item['roadmap_id']}): {item['roadmap_title']}",
            f"- labels: {', '.join(item['roadmap_labels'] or []) or '(none)'}",
            f"- milestone: {item['roadmap_milestone'] or '(none)'}",
            "",
            (item["roadmap_body"] or "(no body)")[:600],
            "",
            "| field | answer |",
            "|---|---|",
            "| need_supported | true / false |",
            "| public_artifact_match | none / partial / material |",
            "| public_claim_defensible | true / false |",
            f"| verdict | {' / '.join(VERDICTS)} |",
            "| notes | |",
            "",
            "---",
            "",
        ]
    (out / f"worksheet_{reviewer}.md").write_text("\n".join(lines), encoding="utf-8")


def _readme(reviewers: list[str], count: int) -> str:
    return f"""Silent Stakeholder — labelling packets
{'=' * 38}

{len(reviewers)} reviewers, {count} pairs each: {', '.join(reviewers)}

Rules
-----
1. Do not open eval/_answer_key.DO_NOT_OPEN_BEFORE_LABELING.json in the repo.
2. Do not compare answers with the other reviewer until both are finished.
3. Fill the "answer" block of every item in your own packet_<name>.json.
4. Leave a value null if you genuinely cannot decide, and say why in notes.
   An honest null is worth more than a guess: a guess becomes a number.

Each packet is ordered differently on purpose, so two reviewers do not fall
into the same rhythm and correlate their mistakes.

When both are done
------------------
    python export_reviewer_packet.py merge --packets <this directory>

That reports agreement, writes disagreements out for adjudication, and only
then produces eval/dev_labels.json.

{count} pairs is below the {MIN_CALIBRATION_LABELS}-label minimum for calibration.
Finishing them yields inter-rater agreement and error analysis. It does not make
the product calibrated, and nothing should say otherwise.
"""


def _answered(item: dict[str, Any]) -> bool:
    answer = item.get("answer") or {}
    return answer.get("verdict") is not None


def _validate_answer(answer: dict[str, Any], pair_id: str, reviewer: str) -> list[str]:
    problems: list[str] = []
    verdict = answer.get("verdict")
    if verdict is not None and verdict not in VERDICTS:
        problems.append(f"{reviewer}/{pair_id}: verdict {verdict!r} not in {VERDICTS}")
    match = answer.get("public_artifact_match")
    if match is not None and match not in MATCH_LEVELS:
        problems.append(f"{reviewer}/{pair_id}: public_artifact_match {match!r} invalid")
    for field in ("need_supported", "public_claim_defensible"):
        value = answer.get(field)
        if value is not None and not isinstance(value, bool):
            problems.append(f"{reviewer}/{pair_id}: {field} must be true/false")
    return problems


def merge(args: argparse.Namespace) -> int:
    packets = sorted(args.packets.glob("packet_*.json"))
    if len(packets) < 2:
        _fail(f"need at least two packets in {args.packets}; found {len(packets)}")

    loaded: dict[str, dict[str, dict[str, Any]]] = {}
    problems: list[str] = []
    for path in packets:
        payload = json.loads(path.read_text(encoding="utf-8"))
        reviewer = payload["reviewer"]
        by_pair: dict[str, dict[str, Any]] = {}
        for item in payload["items"]:
            by_pair[item["pair_id"]] = item
            problems.extend(_validate_answer(item.get("answer") or {}, item["pair_id"], reviewer))
        loaded[reviewer] = by_pair
        done = sum(1 for item in payload["items"] if _answered(item))
        print(f"{reviewer}: {done}/{len(payload['items'])} answered")

    if problems:
        for problem in problems:
            print(f"  invalid: {problem}", file=sys.stderr)
        _fail(f"{len(problems)} invalid answer(s); fix them before merging")

    names = sorted(loaded)
    first, second = loaded[names[0]], loaded[names[1]]
    shared = sorted(set(first) & set(second))

    agreed: list[dict[str, Any]] = []
    disputed: list[dict[str, Any]] = []
    both_answered = 0
    for pair_id in shared:
        a, b = first[pair_id], second[pair_id]
        if not (_answered(a) and _answered(b)):
            continue
        both_answered += 1
        (agreed if a["answer"]["verdict"] == b["answer"]["verdict"] else disputed).append(
            {
                "pair_id": pair_id,
                "need_id": a["need_id"],
                "roadmap_id": a["roadmap_id"],
                names[0]: a["answer"],
                names[1]: b["answer"],
            }
        )

    if both_answered == 0:
        _fail("no pair has been answered by both reviewers yet")

    rate = len(agreed) / both_answered
    print(f"\nboth answered : {both_answered}")
    print(f"agreement     : {len(agreed)}/{both_answered} = {rate:.0%}")
    print(f"disputed      : {len(disputed)}")
    if disputed:
        counts = Counter(
            (row[names[0]]["verdict"], row[names[1]]["verdict"]) for row in disputed
        )
        for (left, right), n in counts.most_common(5):
            print(f"   {left} vs {right}: {n}")

    dispute_path = args.packets / "disputed_for_adjudication.json"
    dispute_path.write_text(json.dumps(disputed, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {dispute_path}")

    labels = [
        {
            "need_id": row["need_id"],
            "roadmap_id": row["roadmap_id"],
            "need_supported": bool(row[names[0]]["need_supported"]),
            "public_artifact_match": row[names[0]]["public_artifact_match"],
            "public_claim_defensible": bool(row[names[0]]["public_claim_defensible"]),
            "verdict": row[names[0]]["verdict"],
            "adjudicated": False,
            "notes": " | ".join(
                filter(None, (row[names[0]].get("notes"), row[names[1]].get("notes")))
            ),
        }
        for row in agreed
    ]

    out_labels = {
        "schema_version": "1.0",
        "status": (
            "AGREED_ONLY_NOT_YET_ADJUDICATED"
            if disputed
            else "AGREED_PENDING_MINIMUM"
        ),
        "reviewers": names,
        "generated_at": datetime.now(UTC).isoformat(),
        "agreement_rate": round(rate, 4),
        "disputed_count": len(disputed),
        "labels": labels,
    }

    if args.out is None:
        print("\nDry run — pass --out eval/dev_labels.json to write.")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out_labels, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out} ({len(labels)} agreed labels)")

    print()
    if disputed:
        print(f"{len(disputed)} pair(s) still need adjudication and are NOT in the label set.")
    if len(labels) < MIN_CALIBRATION_LABELS:
        print(f"{len(labels)} labels < {MIN_CALIBRATION_LABELS} minimum — the calibrator will")
        print("refuse, and every run stays uncalibrated. That is the correct outcome.")
    return 0


def check(args: argparse.Namespace) -> int:
    """Audit a packet directory for contamination before labelling begins."""

    findings: list[str] = []
    if not args.packets.exists():
        _fail(f"{args.packets} does not exist")

    if (args.packets / ANSWER_KEY_NAME).exists():
        findings.append(f"CONTAMINATED: the answer key is present in {args.packets}")

    packets = sorted(args.packets.glob("packet_*.json"))
    if not packets:
        findings.append(f"no packets found in {args.packets}")

    for path in packets:
        payload = json.loads(path.read_text(encoding="utf-8"))
        blob = json.dumps(payload["items"])
        for field in WITHHELD:
            if f'"{field}"' in blob:
                findings.append(f"{path.name}: leaks withheld field {field!r}")
        if ANSWER_KEY_NAME in blob:
            findings.append(f"{path.name}: references the answer key")

    resolved = args.packets.resolve()
    if resolved == REPO or REPO in resolved.parents:
        findings.append(f"{resolved} is inside the repository — packets can be committed")

    if findings:
        for finding in findings:
            print(f"  {finding}")
        print(f"\n{len(findings)} finding(s).")
        return 1
    print(f"{len(packets)} packet(s) clean: no answer key, no withheld fields, outside the repo.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    export_parser = sub.add_parser("export", help="create blind packets")
    export_parser.add_argument("--reviewers", required=True, help="comma-separated, min 2")
    export_parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    export_parser.add_argument("--seed", default="firecode", help="order-shuffle seed")
    export_parser.add_argument("--force", action="store_true", help="allow writing inside the repo")
    export_parser.set_defaults(func=export)

    merge_parser = sub.add_parser("merge", help="combine completed packets")
    merge_parser.add_argument("--packets", type=Path, default=DEFAULT_OUT)
    merge_parser.add_argument("--out", type=Path, default=None, help="omit for a dry run")
    merge_parser.set_defaults(func=merge)

    check_parser = sub.add_parser("check", help="audit a packet directory")
    check_parser.add_argument("--packets", type=Path, default=DEFAULT_OUT)
    check_parser.set_defaults(func=check)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
