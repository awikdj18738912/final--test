from types import SimpleNamespace

from gate1.worker import HypothesisTelemetry, LogprobTelemetry


def test_hypothesis_telemetry_counts_only_replaced_prefix_characters() -> None:
    telemetry = HypothesisTelemetry()
    first = telemetry.observe("你好", chunk_count=1)
    extended = telemetry.observe("你好世界", chunk_count=2)
    revised = telemetry.observe("你好同学", chunk_count=3)

    assert first == {
        "chunk_count": 1,
        "hypothesis_updates": 1,
        "revision_chars": 0,
        "appended_chars": 2,
        "instability_ratio": 0.0,
    }
    assert extended["revision_chars"] == 0
    assert extended["appended_chars"] == 4
    assert revised["revision_chars"] == 2
    assert revised["appended_chars"] == 6
    assert revised["instability_ratio"] == 0.5


def test_logprob_telemetry_keeps_only_scalar_summary() -> None:
    telemetry = LogprobTelemetry()
    outputs = [SimpleNamespace(outputs=[SimpleNamespace(
        token_ids=[101, 102],
        logprobs=[
            {101: SimpleNamespace(logprob=-0.1), 999: SimpleNamespace(logprob=-1.1)},
            {102: SimpleNamespace(logprob=-0.5), 998: SimpleNamespace(logprob=-0.7)},
        ],
    )])]

    telemetry.observe_outputs(outputs)

    assert telemetry.metrics() == {
        "logprob_available": True,
        "logprob_generation_count": 1,
        "logprob_token_count": 2,
        "mean_token_logprob": -0.3,
        "min_token_logprob": -0.5,
        "mean_top1_margin": 0.6,
    }


def test_logprob_telemetry_fails_closed_when_vllm_does_not_return_it() -> None:
    telemetry = LogprobTelemetry()
    telemetry.observe_outputs([SimpleNamespace(outputs=[SimpleNamespace(token_ids=[], logprobs=None)])])

    assert telemetry.metrics() == {
        "logprob_available": False,
        "logprob_generation_count": 1,
        "logprob_token_count": 0,
    }


def test_logprob_telemetry_reset_excludes_worker_warmup() -> None:
    telemetry = LogprobTelemetry()
    telemetry.observe_outputs([SimpleNamespace(outputs=[SimpleNamespace(token_ids=[], logprobs=None)])])
    telemetry.reset()

    assert telemetry.metrics() == {
        "logprob_available": False,
        "logprob_generation_count": 0,
        "logprob_token_count": 0,
    }


if __name__ == "__main__":
    test_hypothesis_telemetry_counts_only_replaced_prefix_characters()
    test_logprob_telemetry_keeps_only_scalar_summary()
    test_logprob_telemetry_fails_closed_when_vllm_does_not_return_it()
    test_logprob_telemetry_reset_excludes_worker_warmup()
    print("gate1 worker telemetry tests passed")
