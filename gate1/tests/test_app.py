import asyncio

from gate1.app import SessionRecord, refine_span, transcript_event
from gate2.streaming import K1RefinementState


def test_whole_window_events_are_versioned_and_hashed() -> None:
    session = SessionRecord("ses_1", "tenant", "Chinese", "balanced", 0.0)
    first = transcript_event(session, "partial", "你好", False)
    second = transcript_event(session, "final", "你好，世界。", True)

    assert first["base_version"] == 0
    assert first["result_version"] == 1
    assert second["base_version"] == 1
    assert second["result_version"] == 2
    assert second["base_hash"] != first["base_hash"]
    assert session.final is True


def test_session_lock_is_asyncio_lock() -> None:
    session = SessionRecord("ses_1", "tenant", "Chinese", "balanced", 0.0)

    async def use_lock() -> None:
        async with session.lock:
            session.input_seq_no += 1

    asyncio.run(use_lock())
    assert session.input_seq_no == 1


def test_async_refiner_emits_revision_and_failure_keeps_source() -> None:
    class FakeRefiner:
        def __init__(self, *, fail_generation: bool = False) -> None:
            self.fail_generation = fail_generation

        async def rpc(self, payload, timeout=30):  # noqa: ANN001, ARG002
            assert payload["active_source_window"]
            return {
                "text": "这个方案，后面再讨论。" if not self.fail_generation else "错误输出",
                "keys": [],
                "finish_reason": "length" if self.fail_generation else "eos",
                "stop_token_id": None if self.fail_generation else 130073,
                "generated_tokens": 12,
                "key_suffix_present": False,
                "inference_sec": 0.001,
            }

    class FakeState:
        def __init__(self, refiner) -> None:  # noqa: ANN001
            self.refiner = refiner

    async def run_case(fail_generation: bool) -> tuple[SessionRecord, dict]:
        session = SessionRecord("ses_refine", "tenant", "Chinese", "balanced", 0.0)
        session.refinement = K1RefinementState(session.session_id)
        span = session.refinement.update_hypothesis("这个方案，呃呃，后面再讨论。", is_final=True)[0]
        session.raw_text = session.refinement.render()
        session.text = session.raw_text
        session.result_version = 1
        session.final = True
        queue = asyncio.Queue()
        await refine_span(FakeState(FakeRefiner(fail_generation=fail_generation)), session, span.span_id, queue)
        return session, queue.get_nowait()

    revised, revision_event = asyncio.run(run_case(False))
    assert revision_event["event"] == "revision"
    assert revision_event["base_version"] == 1
    assert revision_event["result_version"] == 2
    assert revised.text == "这个方案，后面再讨论。"

    rejected, reject_event = asyncio.run(run_case(True))
    assert reject_event["event"] == "refiner_reject"
    assert "generation_max_tokens" in reject_event["validation"]["reasons"]
    assert reject_event["base_version"] == reject_event["result_version"] == 1
    assert rejected.text == "这个方案，呃呃，后面再讨论。"


if __name__ == "__main__":
    test_whole_window_events_are_versioned_and_hashed()
    test_session_lock_is_asyncio_lock()
    test_async_refiner_emits_revision_and_failure_keeps_source()
    print("gate1 state tests passed")
