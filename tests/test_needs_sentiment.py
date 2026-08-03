"""Aspect-sentiment scoring — the input to importance and opportunity scores.

`analyze_sentiment` feeds ODI importance/satisfaction, so its output propagates
all the way to the ranked list a judge reads. It is also the one place in
`needs.py` that must survive whatever ingestion hands it: a missing rating, a
non-numeric rating, an empty body.

Ranges matter as much as values. `opportunity_score` assumes sentiment is in
[-1, 1]; an out-of-range value would push an opportunity score past its own
documented ceiling without raising anything.

Offline and deterministic: this path is pure lexicon arithmetic, no model.
"""

from __future__ import annotations

import pytest

from src.needs import analyze_sentiment


@pytest.mark.parametrize(
    ("rating", "expected"),
    [(1, -1.0), (2, -0.5), (3, 0.0), (4, 0.5), (5, 1.0)],
)
def test_a_rating_alone_maps_linearly_onto_the_sentiment_scale(rating, expected):
    """Every star value, including the neutral midpoint that should score zero.

    "the screen" carries no lexicon word, so this isolates the rating term.
    """

    assert analyze_sentiment({"id": "S1", "text": "the screen", "rating": rating})[
        "sentiment"
    ] == pytest.approx(expected)


def test_wording_alone_decides_when_no_rating_was_supplied():
    """GitHub-sourced signals have no star rating; they must still score."""

    result = analyze_sentiment({"id": "S1", "text": "Export fails constantly"})
    assert result["sentiment"] == pytest.approx(-1.0)


def test_wording_outweighs_the_rating_when_the_two_disagree():
    """A one-star review saying "I love it" is dominated by the words, 70/30.

    Star ratings are noisy — people rate low for reasons the text contradicts —
    so the text is weighted higher on purpose. Pinning the blend keeps that a
    decision rather than an accident.
    """

    result = analyze_sentiment({"id": "S1", "text": "I love it", "rating": 1})
    assert result["sentiment"] == pytest.approx(0.7 * 1.0 + 0.3 * -1.0)


def test_a_non_numeric_rating_is_ignored_rather_than_fatal():
    """Ingestion cannot guarantee a clean rating column; one bad row must not
    abort the whole analysis stage."""

    result = analyze_sentiment({"id": "S1", "text": "crash", "rating": "bogus"})
    assert result["sentiment"] == pytest.approx(-1.0)


def test_a_signal_with_nothing_to_score_is_neutral_not_negative():
    """Absence of evidence is not evidence of dissatisfaction."""

    result = analyze_sentiment({"id": "S1", "text": "", "rating": None})
    assert result["sentiment"] == 0.0
    assert result["intensity"] == 0.0
    assert result["aspect"] == "product use"


@pytest.mark.parametrize(
    "signal",
    [
        {"id": "S1", "text": "terrible awful broken useless crash fail", "rating": 1},
        {"id": "S2", "text": "great perfect amazing excellent love", "rating": 5},
        {"id": "S3", "text": "absolutely extremely very really totally broken", "rating": 1},
    ],
)
def test_scores_stay_inside_the_ranges_downstream_scoring_assumes(signal):
    """Piling on polarity words and intensifiers must not escape the bounds."""

    result = analyze_sentiment(signal)
    assert -1.0 <= float(result["sentiment"]) <= 1.0
    assert 0.0 <= float(result["intensity"]) <= 1.0


def test_the_aspect_is_never_a_polarity_word():
    """The aspect names *what* the user is talking about, not how they feel.

    "crashes" as an aspect would make the need title restate the sentiment.
    """

    result = analyze_sentiment({"id": "S1", "text": "the editor crashes badly", "rating": 1})
    assert result["aspect"] == "editor"


def test_the_aspect_is_never_an_intensifier():
    result = analyze_sentiment(
        {"id": "S1", "text": "absolutely terrible really broken", "rating": 1}
    )
    assert result["aspect"] == "product use"


def test_intensifier_still_increases_intensity():
    plain = analyze_sentiment({"id": "S1", "text": "broken editor"})
    intensified = analyze_sentiment({"id": "S2", "text": "absolutely broken editor"})
    assert plain["aspect"] == intensified["aspect"] == "editor"
    assert float(intensified["intensity"]) > float(plain["intensity"])
