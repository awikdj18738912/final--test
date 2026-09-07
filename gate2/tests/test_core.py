from __future__ import annotations

import unittest

from gate2.core import (
    BoundaryManager,
    ContentConsistencyChecker,
    Evidence,
    PatchCompiler,
    PatchValidator,
    VersionedWindowStore,
)
from gate2.refiner import TransformersRefiner
from gate2.streaming_text_eval import chunk_oral, compose_update, join_chunks, window_keys
from gate2.streaming import K1RefinementState


class BoundaryTests(unittest.TestCase):
    def test_punctuation_endpoint_and_k_window(self) -> None:
        manager = BoundaryManager(max_chars=12)
        self.assertEqual([item.text for item in manager.feed("第一句。第二句")], ["第一句。"])
        self.assertEqual([item.text for item in manager.feed("结束", is_endpoint=True)], ["第二句结束"])
        window = manager.select_window(k=1)
        self.assertEqual(window.source_text, "第二句结束")
        self.assertEqual(window.read_only_prefix, "第一句。")

    def test_hard_cap_prefers_soft_boundary(self) -> None:
        manager = BoundaryManager(max_chars=10)
        closed = manager.feed("甲乙丙丁戊，己庚辛壬癸子")
        self.assertEqual(closed[0].text, "甲乙丙丁戊，")
        self.assertEqual(closed[0].close_reason, "max_chars")

    def test_ascii_period_closes_english_source_span(self) -> None:
        manager = BoundaryManager()
        closed = manager.feed("First sentence. Next")
        self.assertEqual([item.text for item in closed], ["First sentence."])
        self.assertEqual(manager.buffer, " Next")

    def test_cumulative_hypothesis_can_revise_only_pending_suffix(self) -> None:
        manager = BoundaryManager()
        self.assertEqual(manager.update_hypothesis("第一句。第二"), [manager.spans[0]])
        self.assertEqual(manager.buffer, "第二")
        manager.update_hypothesis("第一句。第二句")
        self.assertEqual(manager.buffer, "第二句")
        with self.assertRaisesRegex(ValueError, "revised committed"):
            manager.update_hypothesis("被修改的第一句。第二句")


class PatchTests(unittest.TestCase):
    def setUp(self) -> None:
        manager = BoundaryManager()
        manager.feed("地点在三号楼，呃不对，是四号楼。", is_final=True)
        self.window = manager.select_window(k=3, base_version=7)
        self.compiler = PatchCompiler()

    def test_unicode_diff_roundtrip_and_cas_idempotency(self) -> None:
        clean = "地点在四号楼。"
        patch_set = self.compiler.diff(window=self.window, clean_text=clean, correction_type="self_correction")
        self.assertEqual(PatchCompiler.apply(self.window.current_text, patch_set.patches), clean)

        store = VersionedWindowStore()
        snapshot = store.create(self.window.window_id, self.window.current_text, version=7)
        patch_set = self.compiler.diff(window=snapshot, clean_text=clean)
        event = store.commit(patch_set, clean)
        self.assertEqual(event.result_version, 8)
        self.assertFalse(event.idempotent_replay)
        self.assertTrue(store.commit(patch_set, clean).idempotent_replay)

    def test_noop_is_keep_without_version_increment(self) -> None:
        store = VersionedWindowStore()
        snapshot = store.create("win-keep", "原文", version=4)
        patch_set = self.compiler.diff(window=snapshot, clean_text="原文")
        event = store.commit(patch_set, "原文")
        self.assertEqual(event.event, "keep")
        self.assertEqual(event.result_version, 4)

    def test_stale_cas_rejected(self) -> None:
        store = VersionedWindowStore()
        snapshot = store.create("win", "原文", version=1)
        first = self.compiler.diff(window=snapshot, clean_text="新文")
        store.commit(first, "新文")
        stale = self.compiler.diff(window=snapshot, clean_text="另一个文本")
        with self.assertRaisesRegex(ValueError, "stale CAS"):
            store.commit(stale, "另一个文本")

    def test_sensitive_change_needs_source_or_tenant_evidence(self) -> None:
        manager = BoundaryManager()
        manager.feed("会议在三楼。", is_final=True)
        window = manager.select_window()
        clean = "会议在四楼。"
        patch_set = self.compiler.diff(window=window, clean_text=clean)
        rejected = PatchValidator().validate(
            patch_set=patch_set, window=window, clean_text=clean, tenant_id="tenant-a"
        )
        self.assertFalse(rejected.accepted)
        self.assertIn("unsupported_sensitive_token", rejected.reasons)

        evidence = Evidence("ev-1", "tenant-a", "会议地点是四楼")
        supported = self.compiler.diff(window=window, clean_text=clean, evidence_ids=[evidence.evidence_id])
        accepted = PatchValidator().validate(
            patch_set=supported, window=window, clean_text=clean, tenant_id="tenant-a", evidence=[evidence]
        )
        self.assertTrue(accepted.accepted, accepted.reasons)

    def test_grounded_chinese_to_arabic_digit_format_is_allowed(self) -> None:
        manager = BoundaryManager()
        manager.feed("地点在四号楼。", is_final=True)
        window = manager.select_window()
        clean = "地点在4号楼。"
        patch_set = self.compiler.diff(window=window, clean_text=clean)
        result = PatchValidator().validate(
            patch_set=patch_set, window=window, clean_text=clean, tenant_id="tenant-a"
        )
        self.assertTrue(result.accepted, result.reasons)

    def test_cross_tenant_and_pass_through_changes_rejected(self) -> None:
        manager = BoundaryManager()
        manager.feed("这是干净文本。", is_final=True)
        window = manager.select_window()
        clean = "这是非常干净的文本。"
        foreign = Evidence("ev-x", "tenant-b", clean)
        patch_set = self.compiler.diff(window=window, clean_text=clean, evidence_ids=[foreign.evidence_id])
        result = PatchValidator().validate(
            patch_set=patch_set,
            window=window,
            clean_text=clean,
            tenant_id="tenant-a",
            evidence=[foreign],
            pass_through=True,
        )
        self.assertFalse(result.accepted)
        self.assertIn("cross_tenant_evidence", result.reasons)
        self.assertIn("pass_through_changed_without_evidence", result.reasons)

    def test_content_checker_rejects_ungrounded_word_but_allows_deletion(self) -> None:
        checker = ContentConsistencyChecker()
        self.assertEqual(
            checker.reasons(source_text="我真的是不小心的呀。", clean_text="我真的是不动的呀。"),
            ("ungrounded_content_addition",),
        )
        self.assertEqual(
            checker.reasons(source_text="其其实也是也是。", clean_text="其实也是。"),
            (),
        )

    def test_streaming_k1_rejects_ungrounded_content_addition(self) -> None:
        state = K1RefinementState("ses_content")
        span = state.update_hypothesis("我真的是不小心的呀。", is_final=True)[0]
        decision = state.apply(
            span_id=span.span_id,
            clean_text="我真的是不动的呀。",
            tenant_id="tenant-a",
        )
        self.assertEqual(decision.event, "reject")
        self.assertIn("ungrounded_content_addition", decision.validation.reasons)
        self.assertEqual(state.render(), "我真的是不小心的呀。")


class RefinerAdapterTests(unittest.TestCase):
    def test_key_suffix_is_audit_metadata_not_clean_window(self) -> None:
        text, keys = TransformersRefiner.clean_generated("地点在四号楼。 <KEY>[四号楼、研发中心]")
        self.assertEqual(text, "地点在四号楼。")
        self.assertEqual(keys, ["四号楼", "研发中心"])

    def test_unterminated_key_suffix_is_not_transcript_content(self) -> None:
        raw = "正文。<KEY>[实体、实体、"
        text, keys = TransformersRefiner.clean_generated(raw)
        self.assertEqual(text, "正文。")
        self.assertEqual(keys, [])
        self.assertTrue(TransformersRefiner.has_key_suffix(raw))
        self.assertTrue(TransformersRefiner.has_key_suffix("正文。<KEY>[实体]"))
        self.assertFalse(TransformersRefiner.has_key_suffix("正文。"))
        self.assertTrue(TransformersRefiner.has_incomplete_key_suffix(raw))
        self.assertFalse(TransformersRefiner.has_incomplete_key_suffix("正文。<KEY>[实体]"))

    def test_generation_inputs_drop_unsupported_token_type_ids(self) -> None:
        inputs = {"input_ids": "ids", "attention_mask": "mask", "token_type_ids": "segments"}
        self.assertEqual(
            TransformersRefiner.generation_inputs(inputs),
            {"input_ids": "ids", "attention_mask": "mask"},
        )

    def test_generation_metadata_recognizes_im_end_before_batch_padding(self) -> None:
        class FakeTokenizer:
            @staticmethod
            def decode(token_ids, *, skip_special_tokens=False):
                if skip_special_tokens:
                    raise AssertionError("generation metadata must preserve special tokens")
                return ",".join(str(value) for value in token_ids)

        class FakeContinuation:
            @staticmethod
            def tolist():
                return [10, 11, 130073, 1, 1]

        adapter = object.__new__(TransformersRefiner)
        adapter.stop_token_ids = [1, 130073]
        adapter.tokenizer = FakeTokenizer()
        metadata = adapter.generation_metadata(FakeContinuation())
        self.assertEqual(metadata["finish_reason"], "stop")
        self.assertEqual(metadata["stop_token_id"], 130073)
        self.assertEqual(metadata["generated_tokens"], 3)
        self.assertEqual(metadata["tail_token_ids"], [10, 11, 130073])

    def test_generation_metadata_reports_real_length_exhaustion(self) -> None:
        class FakeTokenizer:
            @staticmethod
            def decode(token_ids, *, skip_special_tokens=False):
                return "output"

        class FakeContinuation:
            @staticmethod
            def tolist():
                return [10, 11]

        adapter = object.__new__(TransformersRefiner)
        adapter.stop_token_ids = [1, 130073]
        adapter.tokenizer = FakeTokenizer()
        metadata = adapter.generation_metadata(FakeContinuation())
        self.assertEqual(metadata["finish_reason"], "length")
        self.assertIsNone(metadata["stop_token_id"])
        self.assertEqual(metadata["generated_tokens"], 2)


class StreamingTextTests(unittest.TestCase):
    def test_chunk_and_window_keys(self) -> None:
        self.assertEqual(chunk_oral("First. Second.", 80), ["First.", "Second."])
        self.assertEqual(window_keys(3, (1, 2, 3)), [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)])

    def test_official_style_window_replacement(self) -> None:
        chunks = ["甲。", "乙。", "丙。"]
        outputs = {
            (0, 0): {"model_text": "A"},
            (1, 1): {"model_text": "B"},
            (2, 2): {"model_text": "C"},
            (0, 1): {"model_text": "AB"},
            (1, 2): {"model_text": "BC"},
        }
        self.assertEqual(join_chunks(["A", "BC"]), "A BC")
        self.assertEqual(compose_update(chunks, outputs, end=2, k=2, field="model_text"), "A BC")

    def test_k1_state_preserves_refined_span_when_raw_tail_advances(self) -> None:
        state = K1RefinementState("ses_1")
        self.assertEqual(state.update_hypothesis("这个方案，呃呃，后面再讨论。下一句"), [])
        closed = state.update_hypothesis("这个方案，呃呃，后面再讨论。下一句继续")
        decision = state.apply(
            span_id=closed[0].span_id,
            clean_text="这个方案，后面再讨论。",
            tenant_id="tenant-a",
        )
        self.assertEqual(decision.event, "replace")
        state.update_hypothesis("这个方案，呃呃，后面再讨论。下一句继续增长")
        self.assertEqual(state.render(), "这个方案，后面再讨论。下一句继续增长")

    def test_k1_does_not_commit_ephemeral_partial_punctuation(self) -> None:
        state = K1RefinementState("ses_1")
        self.assertEqual(state.update_hypothesis("白云。"), [])
        self.assertEqual(state.update_hypothesis("白云区。"), [])
        self.assertIsNone(state.sources.get("seg_0001"))
        closed = state.update_hypothesis("白云区，钟落潭。", is_final=True)
        self.assertTrue(closed)

    def test_k1_generation_failure_keeps_source(self) -> None:
        state = K1RefinementState("ses_1")
        span = state.update_hypothesis("原文。", is_final=True)[0]
        decision = state.apply(
            span_id=span.span_id,
            clean_text="很长的失败输出",
            tenant_id="tenant-a",
            generation_complete=False,
        )
        self.assertEqual(decision.event, "reject")
        self.assertEqual(state.render(), "原文。")

    def test_explicit_self_correction_allows_grounded_large_deletion(self) -> None:
        source = "我是一个苹果，嗯，不对，我是一个梨。"
        state = K1RefinementState("self-correction")
        span = state.update_hypothesis(source, is_final=True)[0]

        conservative = state.apply(
            span_id=span.span_id,
            clean_text="我是一个梨。",
            tenant_id="tenant-a",
        )
        self.assertEqual(conservative.event, "reject")
        self.assertIn("change_ratio_too_large", conservative.validation.reasons)

        allowed = state.apply(
            span_id=span.span_id,
            clean_text="我是一个梨。",
            tenant_id="tenant-a",
            allow_self_correction_deletion=True,
        )
        self.assertEqual(allowed.event, "replace")
        self.assertTrue(allowed.validation.accepted)
        self.assertEqual(state.render(), "我是一个梨。")

    def test_self_correction_exception_does_not_allow_new_content(self) -> None:
        source = "我是一个苹果，嗯，不对，我是一个梨。"
        state = K1RefinementState("self-correction-new-content")
        span = state.update_hypothesis(source, is_final=True)[0]

        decision = state.apply(
            span_id=span.span_id,
            clean_text="我是一只梨。",
            tenant_id="tenant-a",
            allow_self_correction_deletion=True,
        )

        self.assertEqual(decision.event, "reject")
        self.assertIn("change_ratio_too_large", decision.validation.reasons)
        self.assertIn("ungrounded_content_addition", decision.validation.reasons)


if __name__ == "__main__":
    unittest.main()
