"""Stage 4: reference integrity, quote-back verification, and red-team critique."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Protocol

_SPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"\S+")
_CRITIQUE_SEVERITY = {"DEFENSIBLE": 0, "WEAK": 1, "UNSUPPORTED": 2}


class GapCritic(Protocol):
    def critique(
        self,
        *,
        gap: Mapping[str, Any],
        evidence_text: str,
        allowed_signal_ids: Sequence[str],
        allowed_roadmap_ids: Sequence[str],
    ) -> Mapping[str, Any]:
        """Return a critique verdict and grounded rationale."""


class GeminiGapCritic:
    """Optional structured Gemini skeptic used after deterministic verification."""

    def __init__(self, client: Any, model: str, *, temperature: float = 0.2):
        self.client = client
        self.model = model
        self.temperature = temperature

    def critique(
        self,
        *,
        gap: Mapping[str, Any],
        evidence_text: str,
        allowed_signal_ids: Sequence[str],
        allowed_roadmap_ids: Sequence[str],
    ) -> Mapping[str, Any]:
        schema = {
            "type": "object",
            "properties": {
                "verdict": {
                    "type": "string",
                    "enum": ["DEFENSIBLE", "WEAK", "UNSUPPORTED"],
                },
                "rationale": {"type": "string"},
                "signal_ids": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": list(allowed_signal_ids) or ["S0000"],
                    },
                },
                "roadmap_ids": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": list(allowed_roadmap_ids) or ["R0000"],
                    },
                },
            },
            "required": ["verdict", "rationale", "signal_ids", "roadmap_ids"],
            "additionalProperties": False,
        }
        prompt = (
            "Act as an adversarial product-review judge. Decide whether the gap "
            "is supported by the supplied evidence, whether its roadmap verdict "
            "follows, and whether its confidence is proportionate. Do not invent "
            "facts or IDs.\n"
            f"Gap: {json.dumps(dict(gap), default=str)}\nEvidence:\n{evidence_text}"
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "temperature": self.temperature,
                "response_mime_type": "application/json",
                "response_json_schema": schema,
            },
        )
        return json.loads(response.text)


@dataclass(frozen=True, slots=True)
class QuoteVerification:
    valid: bool
    exact: bool
    score: float
    matched_span: str | None
    start: int | None
    end: int | None


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    raise TypeError(f"expected mapping or model, got {type(value).__name__}")


def _nested_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    return _mapping(value)


def _normalize(value: str) -> str:
    return _SPACE_RE.sub(" ", value).strip().casefold()


def verify_quote(
    source_text: str,
    quote: str,
    *,
    fuzzy_threshold: float = 0.90,
    start: int | None = None,
    end: int | None = None,
) -> QuoteVerification:
    """Verify an exact substring or locate the best fuzzy token window."""

    text = str(source_text)
    span = str(quote)
    if not 0.0 <= fuzzy_threshold <= 1.0:
        raise ValueError("fuzzy_threshold must be in [0, 1]")
    if not span.strip():
        return QuoteVerification(False, False, 0.0, None, None, None)
    if start is not None or end is not None:
        if start is None or end is None or start < 0 or end < start or end > len(text):
            return QuoteVerification(False, False, 0.0, None, None, None)
        if text[start:end] == span:
            return QuoteVerification(True, True, 1.0, span, start, end)

    exact_start = text.find(span)
    if exact_start >= 0:
        return QuoteVerification(
            True, True, 1.0, span, exact_start, exact_start + len(span)
        )

    target = _normalize(span)
    if not target:
        return QuoteVerification(False, False, 0.0, None, None, None)
    matches = list(_TOKEN_RE.finditer(text))
    if not matches:
        return QuoteVerification(False, False, 0.0, None, None, None)
    quote_tokens = max(1, len(target.split()))
    best_score = 0.0
    best: tuple[str, int, int] | None = None
    # Small length variation catches punctuation/word insertions without the
    # quadratic character-by-character scan that long review text would cause.
    for width in range(max(1, quote_tokens - 2), quote_tokens + 3):
        for token_start in range(0, len(matches) - width + 1):
            char_start = matches[token_start].start()
            char_end = matches[token_start + width - 1].end()
            candidate = text[char_start:char_end]
            score = SequenceMatcher(None, target, _normalize(candidate)).ratio()
            tie_key = (char_start, char_end)
            best_key = (best[1], best[2]) if best is not None else None
            if score > best_score or (
                score == best_score and best_key is not None and tie_key < best_key
            ):
                best_score = score
                best = (candidate, char_start, char_end)
    valid = best is not None and best_score >= fuzzy_threshold
    return QuoteVerification(
        valid=valid,
        exact=False,
        score=round(best_score, 6),
        matched_span=best[0] if valid and best is not None else None,
        start=best[1] if valid and best is not None else None,
        end=best[2] if valid and best is not None else None,
    )


def validate_references(
    gap: Mapping[str, Any] | Any,
    *,
    signal_ids: set[str],
    roadmap_ids: set[str],
    need_ids: set[str] | None = None,
) -> list[str]:
    row = _mapping(gap)
    evidence = _nested_mapping(row.get("evidence"))
    cited = list(map(str, evidence.get("signal_ids") or []))
    issues: list[str] = []
    unknown_signals = sorted(set(cited).difference(signal_ids))
    if unknown_signals:
        issues.append("unknown signal IDs: " + ", ".join(unknown_signals))
    if len(cited) != len(set(cited)):
        issues.append("duplicate signal IDs in evidence")
    for quote_value in evidence.get("quotes") or []:
        quote = _mapping(quote_value)
        quote_id = str(quote.get("id") or "")
        if quote_id not in cited:
            issues.append(f"quote ID {quote_id!r} is not declared in evidence.signal_ids")
    matched = row.get("matched_roadmap_id")
    if matched is not None and str(matched) not in roadmap_ids:
        issues.append(f"unknown roadmap ID: {matched}")
    if need_ids is not None and str(row.get("need_id") or "") not in need_ids:
        issues.append(f"unknown need ID: {row.get('need_id')}")
    return issues


def verify_gap(
    gap: Mapping[str, Any] | Any,
    *,
    signals: Sequence[Mapping[str, Any] | Any],
    roadmap: Sequence[Mapping[str, Any] | Any],
    needs: Sequence[Mapping[str, Any] | Any] = (),
    fuzzy_threshold: float = 0.90,
) -> dict[str, Any]:
    """Return a non-mutating verification report for one gap."""

    row = _mapping(gap)
    signal_lookup = {
        str(signal_row["id"]): signal_row
        for signal_row in (_mapping(signal) for signal in signals)
    }
    roadmap_lookup = {
        str(roadmap_row["id"]): roadmap_row
        for roadmap_row in (_mapping(item) for item in roadmap)
    }
    need_lookup = {
        str(need_row["id"]): need_row for need_row in (_mapping(need) for need in needs)
    }
    issues = validate_references(
        row,
        signal_ids=set(signal_lookup),
        roadmap_ids=set(roadmap_lookup),
        need_ids=set(need_lookup) if needs else None,
    )
    evidence = _nested_mapping(row.get("evidence"))
    quote_reports: list[dict[str, Any]] = []
    for quote_value in evidence.get("quotes") or []:
        quote = _mapping(quote_value)
        signal_id = str(quote.get("id") or "")
        source = signal_lookup.get(signal_id)
        if source is None:
            continue
        verification = verify_quote(
            str(source.get("text") or ""),
            str(quote.get("span") or ""),
            fuzzy_threshold=fuzzy_threshold,
            start=quote.get("start"),
            end=quote.get("end"),
        )
        quote_reports.append(
            {
                "id": signal_id,
                "valid": verification.valid,
                "exact": verification.exact,
                "score": verification.score,
                "matched_span": verification.matched_span,
                "start": verification.start,
                "end": verification.end,
            }
        )
        if not verification.valid:
            issues.append(
                f"quote for {signal_id} did not meet fuzzy threshold {fuzzy_threshold:.2f}"
            )

    metadata = _nested_mapping(row.get("metadata"))
    roadmap_quote_value = evidence.get("roadmap_quote") or metadata.get("roadmap_quote")
    roadmap_quote_report: dict[str, Any] | None = None
    if roadmap_quote_value:
        roadmap_quote = _mapping(roadmap_quote_value)
        roadmap_id = str(roadmap_quote.get("id") or "")
        item = roadmap_lookup.get(roadmap_id)
        if item is None:
            issues.append(f"roadmap quote cites unknown ID: {roadmap_id}")
        else:
            source_text = "\n".join(
                [
                    str(item.get("title") or ""),
                    str(item.get("body") or ""),
                    " ".join(map(str, item.get("labels") or [])),
                    str(item.get("milestone") or ""),
                ]
            )
            verification = verify_quote(
                source_text,
                str(roadmap_quote.get("span") or ""),
                fuzzy_threshold=fuzzy_threshold,
                start=roadmap_quote.get("start"),
                end=roadmap_quote.get("end"),
            )
            roadmap_quote_report = {
                "id": roadmap_id,
                "valid": verification.valid,
                "exact": verification.exact,
                "score": verification.score,
                "matched_span": verification.matched_span,
                "start": verification.start,
                "end": verification.end,
            }
            if not verification.valid:
                issues.append("roadmap quote did not meet the fuzzy threshold")

    cited_count = len(set(map(str, evidence.get("signal_ids") or [])))
    valid_quotes = sum(bool(report["valid"]) for report in quote_reports)
    if issues:
        critique = "UNSUPPORTED"
    elif cited_count < 2 or valid_quotes == 0 or (
        str(row.get("verdict") or "") != "IGNORED"
        and row.get("matched_roadmap_id") is None
    ):
        critique = "WEAK"
    else:
        critique = "DEFENSIBLE"
    return {
        "gap_id": str(row.get("id") or ""),
        "valid": not issues,
        "issues": issues,
        "quote_reports": quote_reports,
        "roadmap_quote_report": roadmap_quote_report,
        "critique": critique,
    }


def _critic_result(
    result: Mapping[str, Any],
    allowed_signal_ids: set[str],
    allowed_roadmap_ids: set[str],
) -> tuple[str, str]:
    verdict = str(result.get("verdict") or "").upper()
    if verdict not in _CRITIQUE_SEVERITY:
        raise ValueError(f"invalid critique verdict: {verdict!r}")
    signal_ids = set(map(str, result.get("signal_ids") or []))
    roadmap_ids = set(map(str, result.get("roadmap_ids") or []))
    if not signal_ids.issubset(allowed_signal_ids):
        raise ValueError("critic cited an unknown signal ID")
    if not roadmap_ids.issubset(allowed_roadmap_ids):
        raise ValueError("critic cited an unknown roadmap ID")
    rationale = str(result.get("rationale") or "").strip()
    if not rationale:
        raise ValueError("critic rationale is empty")
    return verdict, rationale


def verify_gaps(
    gaps: Sequence[Mapping[str, Any] | Any],
    *,
    signals: Sequence[Mapping[str, Any] | Any],
    roadmap: Sequence[Mapping[str, Any] | Any],
    needs: Sequence[Mapping[str, Any] | Any] = (),
    fuzzy_threshold: float = 0.90,
    critic: GapCritic | None = None,
    drop_unsupported: bool = False,
    canonicalize_fuzzy_quotes: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Verify, optionally red-team, and annotate gaps.

    An LLM critic may downgrade a deterministic result but can never upgrade it.
    """

    signal_rows = [_mapping(signal) for signal in signals]
    roadmap_rows = [_mapping(item) for item in roadmap]
    signal_lookup = {str(row["id"]): row for row in signal_rows}
    verified: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    for gap_value in gaps:
        gap = _mapping(gap_value)
        report = verify_gap(
            gap,
            signals=signal_rows,
            roadmap=roadmap_rows,
            needs=needs,
            fuzzy_threshold=fuzzy_threshold,
        )
        evidence = _nested_mapping(gap.get("evidence"))
        quote_values = [_mapping(quote) for quote in evidence.get("quotes") or []]
        if canonicalize_fuzzy_quotes:
            report_by_id = {
                str(item["id"]): item for item in report["quote_reports"] if item["valid"]
            }
            for quote in quote_values:
                detail = report_by_id.get(str(quote.get("id") or ""))
                if detail and not detail["exact"]:
                    quote["span"] = detail["matched_span"]
                    quote["start"] = detail["start"]
                    quote["end"] = detail["end"]
        evidence["quotes"] = quote_values
        gap["evidence"] = evidence

        final_critique = str(report["critique"])
        metadata = _nested_mapping(gap.get("metadata"))
        metadata["verification"] = {
            "valid": report["valid"],
            "issues": report["issues"],
            "quote_count": len(report["quote_reports"]),
            "valid_quote_count": sum(
                bool(item["valid"]) for item in report["quote_reports"]
            ),
        }
        if critic is not None and report["valid"]:
            cited_ids = list(map(str, evidence.get("signal_ids") or []))
            matched_id = gap.get("matched_roadmap_id")
            allowed_roads = [str(matched_id)] if matched_id is not None else []
            evidence_text = "\n".join(
                f"{signal_id}: {signal_lookup[signal_id].get('text', '')}"
                for signal_id in cited_ids
                if signal_id in signal_lookup
            )
            try:
                llm_verdict, llm_rationale = _critic_result(
                    critic.critique(
                        gap=gap,
                        evidence_text=evidence_text,
                        allowed_signal_ids=cited_ids,
                        allowed_roadmap_ids=allowed_roads,
                    ),
                    set(cited_ids),
                    set(allowed_roads),
                )
                if _CRITIQUE_SEVERITY[llm_verdict] > _CRITIQUE_SEVERITY[final_critique]:
                    final_critique = llm_verdict
                metadata["critic"] = {
                    "verdict": llm_verdict,
                    "rationale": llm_rationale,
                }
            except (ValueError, TypeError, KeyError, json.JSONDecodeError, RuntimeError) as exc:
                metadata["critic"] = {
                    "verdict": "not-applied",
                    "reason": type(exc).__name__,
                }
        gap["critique"] = final_critique
        gap["metadata"] = metadata
        report["critique"] = final_critique
        reports.append(report)
        if not (drop_unsupported and final_critique == "UNSUPPORTED"):
            verified.append(gap)
    return verified, reports
