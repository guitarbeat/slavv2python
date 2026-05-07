import json

from workspace.scripts.analysis.compare_execution_traces import compare_traces


def test_compare_traces_perfect_match(tmp_path):
    trace1_path = tmp_path / "trace1.jsonl"
    trace2_path = tmp_path / "trace2.jsonl"

    events = [
        {
            "event": "iteration_start",
            "iteration": 1,
            "current_linear": 100,
            "current_energy": -10.0,
        },
        {"event": "seed_selected", "seed_idx": 1, "selected_linear": 101, "selected_energy": -9.0},
        {"event": "join", "start_vertex": 1, "end_vertex": 2, "half_1_len": 5, "half_2_len": 5},
    ]

    with trace1_path.open("w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")

    with trace2_path.open("w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")

    divergences = compare_traces(trace1_path, trace2_path)
    assert len(divergences) == 0


def test_compare_traces_energy_divergence(tmp_path):
    trace1_path = tmp_path / "trace1.jsonl"
    trace2_path = tmp_path / "trace2.jsonl"

    e1 = {"event": "seed_selected", "seed_idx": 1, "selected_linear": 101, "selected_energy": -9.0}
    e2 = {"event": "seed_selected", "seed_idx": 1, "selected_linear": 101, "selected_energy": -9.1}

    trace1_path.write_text(json.dumps(e1) + "\n")
    trace2_path.write_text(json.dumps(e2) + "\n")

    divergences = compare_traces(trace1_path, trace2_path)
    assert len(divergences) == 1
    assert divergences[0].key == "selected_energy"


def test_compare_traces_multiple_divergences(tmp_path):
    trace1_path = tmp_path / "trace1.jsonl"
    trace2_path = tmp_path / "trace2.jsonl"

    events1 = [
        {
            "event": "iteration_start",
            "iteration": 1,
            "current_linear": 100,
            "current_energy": -10.0,
        },
        {"event": "seed_selected", "seed_idx": 1, "selected_linear": 101, "selected_energy": -9.0},
        {"event": "seed_selected", "seed_idx": 2, "selected_linear": 201, "selected_energy": -8.0},
    ]
    events2 = [
        {
            "event": "iteration_start",
            "iteration": 1,
            "current_linear": 100,
            "current_energy": -10.0,
        },
        {"event": "seed_selected", "seed_idx": 1, "selected_linear": 101, "selected_energy": -9.1},
        {"event": "seed_selected", "seed_idx": 2, "selected_linear": 201, "selected_energy": -8.1},
    ]

    with trace1_path.open("w") as f:
        for e in events1:
            f.write(json.dumps(e) + "\n")

    with trace2_path.open("w") as f:
        for e in events2:
            f.write(json.dumps(e) + "\n")

    divergences = compare_traces(trace1_path, trace2_path)
    assert len(divergences) == 2
    assert all(d.key == "selected_energy" for d in divergences)
