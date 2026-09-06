"""Small, explainable pre-Refiner router for Chinese ASR text.

The router deliberately uses only the current raw ASR text.  It is not fitted
on the manually reviewed Gate2 set: its role is to avoid paying the quality
and GPU cost of a generative Refiner when there is no explicit disfluency or
self-correction evidence in the text.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re


_CJK = r"[\u4e00-\u9fff]"
_SELF_CORRECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # Standalone "不对" normally means a judgement ("这就不对了") rather
    # than a retraction.  It becomes useful evidence when it introduces a new
    # utterance, for example "不对，我想说…".
    ("explicit_self_correction", re.compile(r"(?:不对|说错了?|讲错了?|口误)[，,。！？!?]\s*(?:我|是|应该|想|要)")),
    ("explicit_rephrasing", re.compile(r"(?:重新说|重说|我的意思)")),
    # Require a nearby corrective "是", so ordinary usages such as "不是一般的"
    # do not cause a call by themselves.
    ("not_but_is_correction", re.compile(r"不是[^。！？，,!?]{0,24}[，,!?]\s*是(?!不)")),
    # "应该是…" is often simply a hedge.  Use it only when it explicitly
    # replaces a preceding formulation, such as "不能说是水光感，应该是油光感".
    ("should_be_correction", re.compile(r"(?:不能说|不该说|不应该说|不是)[^。！？，,!?]{0,24}[，,!?]\s*(?:应该|应当)是")),
)

# A triple is a stronger ASR/disfluency signal than a normal Chinese doubled
# word.  The allow-list intentionally excludes lexical reduplications such as
# "好好好吃" and "慢慢"; it can be extended only after a separate validation set
# demonstrates a net benefit.
_TRIPLE_DISFLUENCY_CHARS = frozenset("不是这那我他她它啊嗯呃死")
_DOUBLE_LEAD_DISFLUENCY_CHARS = frozenset("不也是其这那我你他她它")
_DOUBLE_CHAR = re.compile(rf"(?=({_CJK})\1)")
_TRIPLE_CHAR = re.compile(rf"({_CJK})\1{{2,}}")
_REPEATED_BIGRAM = re.compile(rf"(?=({_CJK}{{2}})\1)")


@dataclass(frozen=True)
class RouteDecision:
    """A serializable decision made before submitting a window to the Refiner."""

    call_refiner: bool
    score: int
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["reasons"] = list(self.reasons)
        return result


def route(raw_text: str) -> RouteDecision:
    """Return whether the current raw ASR text merits a Refiner call.

    A score of two is needed.  Explicit self-correction language is sufficient
    on its own; repetition needs either an allowed triple run or independent
    duplicate-leading characters at two locations.  Digits and Latin letters
    are intentionally not features, because their presence alone says nothing
    about whether this generative Refiner can improve the transcription.
    """

    text = raw_text.strip()
    if not text:
        return RouteDecision(False, 0, ())

    score = 0
    reasons: list[str] = []
    for name, pattern in _SELF_CORRECTION_PATTERNS:
        if pattern.search(text):
            score += 2
            reasons.append(name)

    triple_chars = sorted({match.group(1) for match in _TRIPLE_CHAR.finditer(text)})
    selected_triples = [char for char in triple_chars if char in _TRIPLE_DISFLUENCY_CHARS]
    if selected_triples:
        score += 2
        reasons.append("triple_disfluency:" + "".join(selected_triples))

    # A single doubled character is often a legitimate word (拜拜、慢慢、好好).
    # Two different doubled leading characters in one window are much less
    # likely to be intentional and cover patterns such as "其其实也也是".
    doubled_chars = {
        match.group(1)
        for match in _DOUBLE_CHAR.finditer(text)
        if match.group(1) in _DOUBLE_LEAD_DISFLUENCY_CHARS
    }
    if len(doubled_chars) >= 2:
        score += 2
        reasons.append("multiple_double_leads:" + "".join(sorted(doubled_chars)))
    else:
        repeated_bigrams = {match.group(1) for match in _REPEATED_BIGRAM.finditer(text)}
        # ASR may emit "其其实也也是" or "其其实也是也是" for the same
        # dysfluency.  Requiring the function-word double lead as well as a
        # repeated bigram avoids selecting a repeated phrase by itself.
        if doubled_chars and repeated_bigrams:
            score += 2
            reasons.append(
                "double_lead_with_repeated_bigram:"
                + "".join(sorted(doubled_chars))
                + "/"
                + ",".join(sorted(repeated_bigrams))
            )

    return RouteDecision(score >= 2, score, tuple(reasons))
