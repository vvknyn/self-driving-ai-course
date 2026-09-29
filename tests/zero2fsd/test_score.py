import numpy as np
import pytest

from zero2fsd.car import Car, run
from zero2fsd.dashboard import dashboard
from zero2fsd.score import driving_score, metrics, metrics_table


def _tel(make_tel):
    return make_tel(lat=[0.0, 0.5, -1.0, 0.9], progress=[0, 25, 50, 100], steer=[0.0, 0.05, 0.05, -0.05])


def test_metrics_from_hand_built_telemetry(make_tel):
    m = metrics(_tel(make_tel))
    assert m["completion"] == pytest.approx(0.5)
    assert m["mean_abs_lat"] == pytest.approx(0.6) and m["max_abs_lat"] == pytest.approx(1.0)
    assert m["steps_off_lane"] == 1  # 0.9 is not beyond the 0.9 limit; 1.0 is
    assert m["rms_steer_rate"] == pytest.approx(np.sqrt((1**2 + 0**2 + 2**2) / 3))  # rates 1, 0, -2 rad/s
    assert set(m) == {"completion", "mean_abs_lat", "max_abs_lat", "steps_off_lane", "rms_steer_rate"}


def test_driving_score_is_exact_and_an_int(make_tel):
    score = driving_score(_tel(make_tel))
    assert score == 17 and isinstance(score, int)  # round(100 * 0.5 * (1 - 0.6/0.9))


@pytest.mark.parametrize(
    "lat, progress, length, expected_completion, expected_score",
    [
        ([0.0, 0.0], [0, 250], 200.0, 1.0, 100),  # completion is capped at 1
        ([0.0, 0.0], [0, 200], 200.0, 1.0, 100),
        ([1.0, 1.0], [0, 200], 200.0, 1.0, 0),  # mean |lat| beyond 0.9 clips the accuracy term to 0
        ([0.45, -0.45], [0, 100], 200.0, 0.5, 25),  # 100 * 0.5 * (1 - 0.45/0.9)
        ([0.0, 0.0], [0, -5], 200.0, 0.0, 0),  # going backwards is not negative completion
    ],
)
def test_completion_cap_and_score_clipping(make_tel, lat, progress, length, expected_completion, expected_score):
    tel = make_tel(lat=lat, progress=progress, route_length=length)
    assert metrics(tel)["completion"] == pytest.approx(expected_completion)
    assert driving_score(tel) == expected_score


def test_a_single_sample_has_no_steer_rate(make_tel):
    assert metrics(make_tel(lat=[0.2], progress=[0]))["rms_steer_rate"] == 0.0


def test_metrics_table_is_fixed_width_with_one_row_per_run(make_tel):
    a, b = _tel(make_tel), make_tel(lat=[0.0, 0.0], progress=[0, 200])
    lines = metrics_table(a, b, labels=["worse", "perfect run"]).splitlines()
    assert len(lines) == 4 and len({len(line) for line in lines}) == 1  # header, rule, two rows, all one width
    assert lines[0].split()[:2] == ["run", "completion"] and set(lines[1]) <= {"-", " "}
    assert "worse" in lines[2] and "50%" in lines[2] and lines[2].split()[-1] == "17"
    assert "perfect run" in lines[3] and "100%" in lines[3] and lines[3].split()[-1] == "100"


def test_metrics_table_defaults_labels_to_scenario_names_and_checks_label_count(make_tel):
    tel = make_tel(lat=[0.0, 0.0], progress=[0, 1], scenario="gentle")
    assert "gentle" in metrics_table(tel).splitlines()[2]
    with pytest.raises(ValueError):
        metrics_table(tel, tel, labels=["only one"])


def test_dashboard_returns_a_four_panel_figure_for_real_runs():
    a, b = run(Car(), "straight", max_steps=40), run(Car(controller=lambda est, obs: 0.02), "straight", max_steps=40)
    fig = dashboard(a, b, labels=["provided", "drifting left"])
    assert len(fig.axes) == 4
    assert [t.get_text() for t in fig.axes[0].get_legend().get_texts()] == ["provided", "drifting left"]
    assert "drifting left" in "".join(t.get_text() for t in fig.axes[3].texts)  # the metrics table panel


def test_dashboard_needs_runs_on_one_scenario(make_tel):
    with pytest.raises(ValueError, match="gentle.*straight"):
        dashboard(make_tel([0.0], [0], scenario="straight"), make_tel([0.0], [0], scenario="gentle"))
    with pytest.raises(ValueError, match="no runs"):
        dashboard()
