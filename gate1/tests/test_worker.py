from gate1.worker import HypothesisTelemetry


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
