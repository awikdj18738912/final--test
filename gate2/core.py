#!/usr/bin/env python3
"""Deterministic source-window, diff, validation, and CAS primitives."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from typing import Iterable


STRONG_BOUNDARY = set("。！？.!?；;\n")
SOFT_BOUNDARY = set("，,、：: ")
NON_CONTENT_CHARS = set(" \t\r\n，。！？,.!?；;、：:‘’“”\"'（）()【】[]{}<>《》…—-_~`")
SENSITIVE_PATTERN = re.compile(
    r"\d+(?:\.\d+)?|[零〇一二两三四五六七八九十百千万亿]+|"
    r"不是|不|没有|没|无|否|未"
)
CHINESE_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
CHINESE_UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10000, "亿": 100000000}


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_number(token: str) -> int | None:
    """Canonicalize an isolated Arabic/Chinese integer; never infer time or units."""
    if token.isdigit():
        return int(token)
    if not token or any(char not in CHINESE_DIGITS and char not in CHINESE_UNITS for char in token):
        return None
    if all(char in CHINESE_DIGITS for char in token):
        return int("".join(str(CHINESE_DIGITS[char]) for char in token))
    total = section = number = 0
    for char in token:
        if char in CHINESE_DIGITS:
            number = CHINESE_DIGITS[char]
            continue
        unit = CHINESE_UNITS[char]
        if unit < 10000:
            section += (number or 1) * unit
        else:
            section = (section + number) * unit
            total += section
            section = 0
        number = 0
    return total + section + number


def sensitive_token_supported(token: str, evidence_text: str) -> bool:
    if token in evidence_text:
        return True
    canonical = canonical_number(token)
    return canonical is not None and any(canonical_number(item) == canonical for item in SENSITIVE_PATTERN.findall(evidence_text))


@dataclass(frozen=True)
class SourceSpan:
    span_id: str
    text: str
    close_reason: str


@dataclass(frozen=True)
class SourceWindow:
    window_id: str
    span_ids: tuple[str, ...]
    source_text: str
    current_text: str
    read_only_prefix: str
    base_version: int
    base_hash: str
    mutable_tail_start: int = 0


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    tenant_id: str
    text: str
    status: str = "confirmed"
    trust: float = 1.0
    source_type: str = "user"


@dataclass(frozen=True)
class Patch:
    start_char: int
    end_char: int
    expected_source: str
    replacement: str


@dataclass(frozen=True)
class PatchSet:
    window_id: str
    base_version: int
    base_hash: str
    patches: tuple[Patch, ...]
    evidence_ids: tuple[str, ...] = ()
    memory_ids: tuple[str, ...] = ()
    correction_type: str = "rewrite"
    patch_hash: str = ""


class ContentConsistencyChecker:
    """Reject ungrounded lexical additions before a streaming patch is committed.

    This conservative check is for the no-evidence K=1 path. It does not block
    deletions or punctuation/number formatting, while requiring novel CJK or
    alphanumeric characters to occur in the source or trusted evidence.
    """

    def __init__(self, *, max_novel_chars: int = 0) -> None:
        if max_novel_chars < 0:
            raise ValueError("max_novel_chars must be non-negative")
        self.max_novel_chars = max_novel_chars

    @staticmethod
    def novel_characters(source_text: str, clean_text: str, evidence_text: str = "") -> tuple[str, ...]:
        grounded = set(source_text + evidence_text)
        novel = {
            char for char in clean_text
            if char not in grounded and char not in NON_CONTENT_CHARS and not char.isdigit()
        }
        return tuple(sorted(novel))

    def reasons(self, *, source_text: str, clean_text: str, evidence_text: str = "") -> tuple[str, ...]:
        novel = self.novel_characters(source_text, clean_text, evidence_text)
        if len(novel) <= self.max_novel_chars:
            return ()
        return ("ungrounded_content_addition",)


@dataclass(frozen=True)
class ValidationResult:
    accepted: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class RevisionEvent:
    event: str
    window_id: str
    base_version: int
    result_version: int
    base_hash: str
    result_hash: str
    text: str
    patch_hash: str
    idempotent_replay: bool = False


class BoundaryManager:
    """Close immutable source spans on punctuation, endpoints, or a hard cap."""

    def __init__(self, max_chars: int = 80, read_only_prefix_chars: int = 120) -> None:
        if max_chars <= 0 or read_only_prefix_chars < 0:
            raise ValueError("invalid boundary configuration")
        self.max_chars = max_chars
        self.read_only_prefix_chars = read_only_prefix_chars
        self.buffer = ""
        self.spans: list[SourceSpan] = []

    def _close(self, cutoff: int, reason: str) -> SourceSpan:
        text = self.buffer[:cutoff]
        self.buffer = self.buffer[cutoff:]
        span = SourceSpan(span_id=f"seg_{len(self.spans) + 1:04d}", text=text, close_reason=reason)
        self.spans.append(span)
        return span

    def _emit_stable(self, stable_chars: int) -> list[SourceSpan]:
        closed: list[SourceSpan] = []
        stable_chars = min(max(stable_chars, 0), len(self.buffer))
        while self.buffer and stable_chars > 0:
            inspect_chars = min(self.max_chars, stable_chars)
            strong = next(
                (index + 1 for index, char in enumerate(self.buffer[:inspect_chars]) if char in STRONG_BOUNDARY),
                None,
            )
            if strong is not None:
                closed.append(self._close(strong, "punctuation"))
                stable_chars -= strong
                continue
            if stable_chars >= self.max_chars and len(self.buffer) >= self.max_chars:
                search_from = max(1, self.max_chars // 2)
                candidates = [
                    index + 1
                    for index, char in enumerate(self.buffer[: self.max_chars])
                    if char in SOFT_BOUNDARY and index + 1 >= search_from
                ]
                cutoff = candidates[-1] if candidates else self.max_chars
                closed.append(self._close(cutoff, "max_chars"))
                stable_chars -= cutoff
                continue
            break
        return closed

    def feed(self, text_delta: str, *, is_endpoint: bool = False, is_final: bool = False) -> list[SourceSpan]:
        self.buffer += text_delta
        closed = self._emit_stable(len(self.buffer))
        if (is_endpoint or is_final) and self.buffer:
            closed.append(self._close(len(self.buffer), "final" if is_final else "endpoint"))
        return closed

    @property
    def committed_source(self) -> str:
        return "".join(span.text for span in self.spans)

    def update_hypothesis(
        self,
        hypothesis: str,
        *,
        is_endpoint: bool = False,
        is_final: bool = False,
        stable_prefix_chars: int | None = None,
    ) -> list[SourceSpan]:
        """Replace the mutable suffix of a cumulative ASR hypothesis.

        Text already emitted as a source span is immutable. Revisions are
        allowed only inside the pending suffix; callers must safely disable
        refinement if the ASR rewrites committed source text.
        """
        committed = self.committed_source
        if not hypothesis.startswith(committed):
            raise ValueError("ASR revised committed source text")
        self.buffer = hypothesis[len(committed) :]
        stable = len(self.buffer) if stable_prefix_chars is None else stable_prefix_chars
        closed = self._emit_stable(stable)
        if (is_endpoint or is_final) and self.buffer:
            closed.append(self._close(len(self.buffer), "final" if is_final else "endpoint"))
        return closed

    def select_window(self, *, k: int = 3, max_chars: int = 240, base_version: int = 0, current_text: str | None = None) -> SourceWindow:
        if k <= 0 or max_chars <= 0:
            raise ValueError("k and max_chars must be positive")
        selected: list[SourceSpan] = []
        total = 0
        for span in reversed(self.spans):
            if selected and (len(selected) >= k or total + len(span.text) > max_chars):
                break
            if not selected and len(span.text) > max_chars:
                raise ValueError("latest source span exceeds window max_chars")
            selected.append(span)
            total += len(span.text)
            if len(selected) >= k:
                break
        selected.reverse()
        if not selected:
            raise ValueError("no closed source spans")
        first_index = self.spans.index(selected[0])
        prefix = "".join(span.text for span in self.spans[:first_index])[-self.read_only_prefix_chars :]
        source = "".join(span.text for span in selected)
        rendered = source if current_text is None else current_text
        ids = tuple(span.span_id for span in selected)
        window_id = "win_" + hashlib.sha256("|".join(ids).encode("utf-8")).hexdigest()[:12]
        return SourceWindow(
            window_id=window_id,
            span_ids=ids,
            source_text=source,
            current_text=rendered,
            read_only_prefix=prefix,
            base_version=base_version,
            base_hash=text_hash(rendered),
        )


class PatchCompiler:
    """Compile clean text into auditable, window-local Unicode code-point edits."""

    @staticmethod
    def apply(old_text: str, patches: Iterable[Patch]) -> str:
        result = old_text
        ordered = sorted(patches, key=lambda item: (item.start_char, item.end_char), reverse=True)
        previous_start = len(old_text) + 1
        for patch in ordered:
            if patch.end_char > previous_start:
                raise ValueError("overlapping patches")
            if not 0 <= patch.start_char <= patch.end_char <= len(old_text):
                raise ValueError("patch outside source window")
            if old_text[patch.start_char : patch.end_char] != patch.expected_source:
                raise ValueError("expected_source mismatch")
            result = result[: patch.start_char] + patch.replacement + result[patch.end_char :]
            previous_start = patch.start_char
        return result

    def diff(
        self,
        *,
        window: SourceWindow,
        clean_text: str,
        evidence_ids: Iterable[str] = (),
        memory_ids: Iterable[str] = (),
        correction_type: str = "rewrite",
    ) -> PatchSet:
        patches = tuple(
            Patch(start, end, window.current_text[start:end], clean_text[new_start:new_end])
            for tag, start, end, new_start, new_end in SequenceMatcher(None, window.current_text, clean_text, autojunk=False).get_opcodes()
            if tag != "equal"
        )
        payload = {
            "window_id": window.window_id,
            "base_version": window.base_version,
            "base_hash": window.base_hash,
            "patches": [asdict(item) for item in patches],
            "evidence_ids": sorted(evidence_ids),
            "memory_ids": sorted(memory_ids),
            "correction_type": correction_type,
        }
        digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        patch_set = PatchSet(
            window_id=window.window_id,
            base_version=window.base_version,
            base_hash=window.base_hash,
            patches=patches,
            evidence_ids=tuple(payload["evidence_ids"]),
            memory_ids=tuple(payload["memory_ids"]),
            correction_type=correction_type,
            patch_hash=digest,
        )
        if self.apply(window.current_text, patches) != clean_text:
            raise AssertionError("compiled patches do not reproduce clean text")
        return patch_set


class PatchValidator:
    def __init__(
        self,
        *,
        max_patches: int = 16,
        max_changed_chars: int = 120,
        max_change_ratio: float = 0.60,
        content_checker: ContentConsistencyChecker | None = None,
    ) -> None:
        self.max_patches = max_patches
        self.max_changed_chars = max_changed_chars
        self.max_change_ratio = max_change_ratio
        self.content_checker = content_checker

    def validate(
        self,
        *,
        patch_set: PatchSet,
        window: SourceWindow,
        clean_text: str,
        tenant_id: str,
        evidence: Iterable[Evidence] = (),
        pass_through: bool = False,
        allow_self_correction_deletion: bool = False,
    ) -> ValidationResult:
        reasons: list[str] = []
        evidence_by_id = {item.evidence_id: item for item in evidence}
        if patch_set.window_id != window.window_id:
            reasons.append("window_id_mismatch")
        if patch_set.base_version != window.base_version:
            reasons.append("stale_base_version")
        if patch_set.base_hash != window.base_hash or window.base_hash != text_hash(window.current_text):
            reasons.append("base_hash_mismatch")
        if len(patch_set.patches) > self.max_patches:
            reasons.append("too_many_patches")
        changed = sum(max(item.end_char - item.start_char, len(item.replacement)) for item in patch_set.patches)
        if changed > self.max_changed_chars:
            reasons.append("change_span_too_large")
        self_correction_deletion = (
            allow_self_correction_deletion
            and bool(clean_text.strip())
            and clean_text in window.current_text
        )
        if (
            window.current_text
            and changed / len(window.current_text) > self.max_change_ratio
            and not self_correction_deletion
        ):
            reasons.append("change_ratio_too_large")
        try:
            applied = PatchCompiler.apply(window.current_text, patch_set.patches)
            if applied != clean_text:
                reasons.append("clean_text_mismatch")
        except ValueError as exc:
            reasons.append(str(exc).replace(" ", "_"))
        for patch in patch_set.patches:
            if patch.start_char < window.mutable_tail_start:
                reasons.append("outside_mutable_tail")

        allowed_evidence = []
        for evidence_id in (*patch_set.evidence_ids, *patch_set.memory_ids):
            item = evidence_by_id.get(evidence_id)
            if item is None:
                reasons.append("unknown_evidence")
            elif item.tenant_id != tenant_id:
                reasons.append("cross_tenant_evidence")
            elif item.status in {"quarantined", "rejected", "superseded"}:
                reasons.append("untrusted_evidence_status")
            else:
                allowed_evidence.append(item)
        evidence_text = window.source_text + "".join(item.text for item in allowed_evidence)
        for token in SENSITIVE_PATTERN.findall(clean_text):
            if not sensitive_token_supported(token, evidence_text):
                reasons.append("unsupported_sensitive_token")
                break
        if self.content_checker is not None:
            reasons.extend(
                self.content_checker.reasons(
                    source_text=window.source_text,
                    clean_text=clean_text,
                    evidence_text="".join(item.text for item in allowed_evidence),
                )
            )
        if pass_through and patch_set.patches and not allowed_evidence:
            reasons.append("pass_through_changed_without_evidence")
        return ValidationResult(accepted=not reasons, reasons=tuple(dict.fromkeys(reasons)))


@dataclass
class _WindowRecord:
    text: str
    version: int
    history: dict[str, RevisionEvent] = field(default_factory=dict)


class VersionedWindowStore:
    """Single-writer whole-window CAS with idempotent patch replay."""

    def __init__(self) -> None:
        self._records: dict[str, _WindowRecord] = {}

    def create(self, window_id: str, text: str, *, version: int = 0) -> SourceWindow:
        if window_id in self._records:
            raise ValueError("window already exists")
        self._records[window_id] = _WindowRecord(text=text, version=version)
        return self.snapshot(window_id)

    def snapshot(self, window_id: str) -> SourceWindow:
        record = self._records[window_id]
        return SourceWindow(window_id, (), record.text, record.text, "", record.version, text_hash(record.text))

    def commit(self, patch_set: PatchSet, clean_text: str) -> RevisionEvent:
        record = self._records[patch_set.window_id]
        if patch_set.patch_hash in record.history:
            previous = record.history[patch_set.patch_hash]
            return RevisionEvent(**{**asdict(previous), "idempotent_replay": True})
        if patch_set.base_version != record.version or patch_set.base_hash != text_hash(record.text):
            raise ValueError("stale CAS")
        applied = PatchCompiler.apply(record.text, patch_set.patches)
        if applied != clean_text:
            raise ValueError("clean text does not match patches")
        if not patch_set.patches:
            event = RevisionEvent(
                event="keep",
                window_id=patch_set.window_id,
                base_version=record.version,
                result_version=record.version,
                base_hash=patch_set.base_hash,
                result_hash=patch_set.base_hash,
                text=record.text,
                patch_hash=patch_set.patch_hash,
            )
            record.history[patch_set.patch_hash] = event
            return event
        event = RevisionEvent(
            event="replace_window",
            window_id=patch_set.window_id,
            base_version=record.version,
            result_version=record.version + 1,
            base_hash=patch_set.base_hash,
            result_hash=text_hash(clean_text),
            text=clean_text,
            patch_hash=patch_set.patch_hash,
        )
        record.text = clean_text
        record.version += 1
        record.history[patch_set.patch_hash] = event
        return event
