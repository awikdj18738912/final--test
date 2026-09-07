import asyncio

from gate1.app import (
    FRONTEND_PATH,
    Gpu1SwitchConflict,
    ServiceState,
    SessionRecord,
    refine_span,
    schedule_refinements,
    transcript_event,
    wait_for_refinements,
)
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


def test_frontend_is_bundled_with_realtime_and_offline_controls() -> None:
    html = FRONTEND_PATH.read_text(encoding="utf-8")

    assert 'id="start-recording"' in html
    assert 'id="offline-form"' in html
    assert 'new WebSocket' in html
    assert '/offline/jobs' in html
    assert '/runtime/gpu1-role' in html


def test_idle_gpu1_worker_can_switch_without_restarting_gpu0() -> None:
    created = []

    class FakeWorker:
        def __init__(self, role: str) -> None:
            self.role = role
            self.started = False
            self.stopped = False
            created.append(self)

        async def start(self) -> None:
            self.started = True

        def stop(self) -> None:
            self.stopped = True

    class FakeState(ServiceState):
        def __init__(self) -> None:
            self.gpu1_role = "offline"
            self.gpu1_lock = asyncio.Lock()
            self.offline = FakeWorker("offline")
            self.refiner = None
            self.sessions = {}
            self.jobs = {}

        def build_gpu1_process(self, role: str):  # noqa: ANN201
            return FakeWorker(role)

    state = FakeState()
    previous = state.offline
    changed = asyncio.run(state.switch_gpu1_role("refiner"))

    assert changed is True
    assert previous.stopped is True
    assert state.gpu1_role == "refiner"
    assert state.offline is None
    assert state.refiner is created[-1]
    assert state.refiner.started is True


def test_gpu1_switch_rejects_an_active_realtime_session() -> None:
    class FakeState(ServiceState):
        def __init__(self) -> None:
            self.gpu1_role = "offline"
            self.gpu1_lock = asyncio.Lock()
            self.offline = None
            self.refiner = None
            session = SessionRecord("active", "tenant", "Chinese", "balanced", 0.0)
            session.active = True
            self.sessions = {session.session_id: session}
            self.jobs = {}

    try:
        asyncio.run(FakeState().switch_gpu1_role("refiner"))
    except Gpu1SwitchConflict:
        pass
    else:
        raise AssertionError("GPU1 switch must reject an active realtime session")


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


def test_router_skips_clean_span_without_an_rpc() -> None:
    class MustNotCallRefiner:
        async def rpc(self, payload, timeout=30):  # noqa: ANN001, ARG002
            raise AssertionError(f"router should have skipped this RPC: {payload}")

    class FakeState:
        refiner = MustNotCallRefiner()

    async def run_case() -> tuple[SessionRecord, dict]:
        session = SessionRecord("ses_skip", "tenant", "Chinese", "balanced", 0.0)
        session.refinement = K1RefinementState(session.session_id)
        span = session.refinement.update_hypothesis("爸爸妈妈都来了。", is_final=True)[0]
        session.raw_text = session.refinement.render()
        session.text = session.raw_text
        queue = asyncio.Queue()
        schedule_refinements(FakeState(), session, [span], queue)
        await wait_for_refinements(session)
        return session, queue.get_nowait()

    session, event = asyncio.run(run_case())
    assert event["event"] == "refiner_skipped"
    assert event["router"]["call_refiner"] is False
    assert event["router"]["reasons"] == []
    assert session.refinement_results == [event]


def test_router_submits_explicit_retraction_and_records_decision() -> None:
    class FakeRefiner:
        def __init__(self) -> None:
            self.calls = 0

        async def rpc(self, payload, timeout=30):  # noqa: ANN001, ARG002
            self.calls += 1
            return {
                "text": payload["active_source_window"],
                "keys": [],
                "finish_reason": "eos",
                "stop_token_id": 130073,
                "generated_tokens": 1,
                "key_suffix_present": False,
                "inference_sec": 0.001,
            }

    class FakeState:
        def __init__(self) -> None:
            self.refiner = FakeRefiner()

    async def run_case() -> tuple[SessionRecord, dict, FakeState]:
        state = FakeState()
        session = SessionRecord("ses_call", "tenant", "Chinese", "balanced", 0.0)
        session.refinement = K1RefinementState(session.session_id)
        span = session.refinement.update_hypothesis("不对，我想重新说一遍。", is_final=True)[0]
        session.raw_text = session.refinement.render()
        session.text = session.raw_text
        queue = asyncio.Queue()
        schedule_refinements(state, session, [span], queue)
        await wait_for_refinements(session)
        return session, queue.get_nowait(), state

    session, event, state = asyncio.run(run_case())
    assert state.refiner.calls == 1
    assert event["event"] == "refiner_keep"
    assert event["router"]["call_refiner"] is True
    assert "explicit_self_correction" in event["router"]["reasons"]
    assert session.refinement_results[0]["router"] == event["router"]


if __name__ == "__main__":
    test_whole_window_events_are_versioned_and_hashed()
    test_session_lock_is_asyncio_lock()
    test_frontend_is_bundled_with_realtime_and_offline_controls()
    test_idle_gpu1_worker_can_switch_without_restarting_gpu0()
    test_gpu1_switch_rejects_an_active_realtime_session()
    test_async_refiner_emits_revision_and_failure_keeps_source()
    test_router_skips_clean_span_without_an_rpc()
    test_router_submits_explicit_retraction_and_records_decision()
    print("gate1 state tests passed")
