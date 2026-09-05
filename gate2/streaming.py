"""K=1 state bridge between cumulative streaming ASR and safe Refiner patches."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .core import (
    BoundaryManager,
    ContentConsistencyChecker,
    Patch,
    PatchCompiler,
    PatchValidator,
    SourceSpan,
    SourceWindow,
    ValidationResult,
    text_hash,
)


@dataclass(frozen=True)
class SpanDecision:
    event: str
    window: SourceWindow
    clean_text: str
    patches: tuple[Patch, ...]
    validation: ValidationResult


class K1RefinementState:
    """Keep immutable raw spans separate from their replaceable rendering."""

    def __init__(self, session_id: str, *, max_span_chars: int = 80) -> None:
        self.session_id = session_id
        self.boundaries = BoundaryManager(max_chars=max_span_chars)
        self.sources: dict[str, str] = {}
        self.outputs: dict[str, str] = {}
        self.compiler = PatchCompiler()
        self.validator = PatchValidator(content_checker=ContentConsistencyChecker())
        self.previous_hypothesis = ""

    def update_hypothesis(self, hypothesis: str, *, is_final: bool = False) -> list[SourceSpan]:
        common = 0
        for left, right in zip(self.previous_hypothesis, hypothesis):
            if left != right:
                break
            common += 1
        committed_chars = len(self.boundaries.committed_source)
        stable_pending_chars = max(0, common - committed_chars)
        closed = self.boundaries.update_hypothesis(
            hypothesis,
            is_final=is_final,
            stable_prefix_chars=stable_pending_chars,
        )
        self.previous_hypothesis = hypothesis
        for span in closed:
            self.sources[span.span_id] = span.text
            self.outputs.setdefault(span.span_id, span.text)
        return closed

    def render(self) -> str:
        closed = "".join(self.outputs.get(span.span_id, span.text) for span in self.boundaries.spans)
        return closed + self.boundaries.buffer

    def window(self, span_id: str) -> SourceWindow:
        source = self.sources[span_id]
        index = next(index for index, span in enumerate(self.boundaries.spans) if span.span_id == span_id)
        prefix = "".join(span.text for span in self.boundaries.spans[:index])[-120:]
        digest = hashlib.sha256(f"{self.session_id}:{span_id}".encode("utf-8")).hexdigest()[:16]
        return SourceWindow(
            window_id=f"win_{digest}",
            span_ids=(span_id,),
            source_text=source,
            current_text=source,
            read_only_prefix=prefix,
            base_version=0,
            base_hash=text_hash(source),
        )

    def apply(
        self,
        *,
        span_id: str,
        clean_text: str,
        tenant_id: str,
        generation_complete: bool = True,
    ) -> SpanDecision:
        window = self.window(span_id)
        patch_set = self.compiler.diff(window=window, clean_text=clean_text, correction_type="streaming_k1")
        validation = self.validator.validate(
            patch_set=patch_set,
            window=window,
            clean_text=clean_text,
            tenant_id=tenant_id,
            evidence=[],
        )
        if not generation_complete:
            validation = ValidationResult(
                False, tuple(dict.fromkeys((*validation.reasons, "generation_max_tokens")))
            )
        if validation.accepted:
            self.outputs[span_id] = clean_text
        event = "reject" if not validation.accepted else "keep" if not patch_set.patches else "replace"
        return SpanDecision(event, window, clean_text, patch_set.patches, validation)
