"""Checks for src/leaf_width.py: photo selection, the measuring loop and the pre-declared rule."""

import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src import leaf_width as lw
from src.config import load_config


def fake_inventory(sides):
    return pd.DataFrame({"photo_id": [f"C/{i:04d}.jpg" for i in range(len(sides))], "class": "healthy",
                         "width": sides, "height": sides, "opened_ok": True})


UNIFORM = fake_inventory(np.random.default_rng(1).integers(700, 3200, 300))


def test_selection_is_reproducible_unique_and_spans_the_range():
    first = lw.select_photos(UNIFORM, 30, seed=0)
    again = lw.select_photos(UNIFORM.sample(frac=1, random_state=3), 30, seed=0)
    assert first["photo_id"].tolist() == again["photo_id"].tolist()  # input order does not matter
    assert len(first) == 30 and first["photo_id"].is_unique
    assert first["short_side"].min() <= UNIFORM["width"].min() * 1.05
    assert first["short_side"].max() >= UNIFORM["width"].max() / 1.05


@pytest.mark.parametrize("seed", range(40))
def test_empty_bands_never_repeat_a_photo(seed):
    gappy = fake_inventory([700, 705, 710, 715, 720, 3000, 3050, 3100])  # most bands in between are empty
    chosen = lw.select_photos(gappy, 6, seed=seed)
    assert len(chosen) == 6 and chosen["photo_id"].is_unique


def test_asking_for_more_photos_than_exist_fails():
    with pytest.raises(ValueError):
        lw.select_photos(fake_inventory([800, 900]), 3, seed=0)


@pytest.fixture
def setup(tmp_path):
    config = load_config("configs/inventory.yaml")
    settings = config["pixel_scale"]  # absolute paths replace the repo-relative ones
    settings.update(selection_file=str(tmp_path / "selection.csv"), measurements_dir=str(tmp_path / "lw"),
                    measurements_file=str(tmp_path / "all.csv"),
                    decision_file=str(tmp_path / "decision.json"))
    selected = lw.select_photos(UNIFORM, 30, config["seed"])
    selected.to_csv(tmp_path / "selection.csv", index=False)
    return config, selected, tmp_path


def write_measurements(tmp_path, selected, widths_by_measurer, status="accept"):
    (tmp_path / "lw").mkdir(exist_ok=True)
    for who, widths in widths_by_measurer.items():
        rows = [{"dataset": "primary", "photo_id": pid, "measurer": who, "x1": 0, "y1": 0, "x2": w, "y2": 0,
                 "width_px": w, "status": status, "measured_at": "2026-11-05T00:00:00+00:00"}
                for pid, w in zip(selected["photo_id"], widths)]
        pd.DataFrame(rows, columns=lw.COLUMNS).to_csv(tmp_path / "lw" / f"{who}.csv", index=False)


def test_incomplete_until_both_measurers_answered_every_photo(setup):
    config, selected, tmp_path = setup
    assert not lw.decide(config)["complete"]
    write_measurements(tmp_path, selected.iloc[:29], {"ahsan": [100] * 29, "harris": [100] * 29})
    assert not lw.decide(config)["complete"]
    assert not (tmp_path / "decision.json").exists()


def test_leaf_width_growing_with_photo_size_means_rescale(setup):
    config, selected, tmp_path = setup
    widths = selected["short_side"] * 0.2  # resized photos: leaf width scales with the photo
    write_measurements(tmp_path, selected, {"ahsan": widths, "harris": widths * 1.02})
    result = lw.decide(config)
    assert result["complete"] and result["rescale"] and result["spearman_rho"] > 0.9
    assert result["median_relative_gap"] == pytest.approx(0.02 / 1.01, abs=1e-4)
    decision = json.loads((tmp_path / "decision.json").read_text())
    assert decision["rescale"] and decision["n_used"] == 30
    assert decision["size_fit"]["slope"] == pytest.approx(1.0)  # pure resize: width proportional to size
    width, side = widths.iloc[0] * 1.01, selected["short_side"].iloc[0]  # 1.01: mean of the two measurers
    median = decision["median_leaf_width_px"]  # a pure resize is undone: every leaf becomes the median width
    assert width * lw.rescale_factor(side, decision) == pytest.approx(median, rel=1e-3)
    assert len(pd.read_csv(tmp_path / "all.csv")) == 60


def test_constant_leaf_width_means_native_resolution(setup):
    config, selected, tmp_path = setup
    widths = 400 + np.random.default_rng(2).normal(0, 20, len(selected))  # crops: same zoom everywhere
    write_measurements(tmp_path, selected, {"ahsan": widths, "umer": widths})
    result = lw.decide(config)
    assert result["complete"] and not result["rescale"]
    assert result["size_fit"] is None and lw.rescale_factor(1000, result) == 1.0
    assert result["median_leaf_width_px"] == pytest.approx(np.median(widths), abs=0.1)


def test_unmeasurable_photos_are_left_out_and_counted(setup):
    config, selected, tmp_path = setup
    write_measurements(tmp_path, selected, {"ahsan": selected["short_side"] * 0.2,
                                            "harris": selected["short_side"] * 0.2})
    harris = pd.read_csv(tmp_path / "lw" / "harris.csv")
    harris.loc[:2, ["x1", "y1", "x2", "y2", "width_px"]] = np.nan
    harris.loc[:2, "status"] = "unmeasurable"
    harris.to_csv(tmp_path / "lw" / "harris.csv", index=False)
    result = lw.decide(config)
    assert result["n_used"] == 27 and result["n_unmeasurable"] == 3


@pytest.mark.parametrize("names", [["ahsan", "test"], ["ahsan", "harris", "umer"]])
def test_unknown_or_extra_measurers_are_refused(setup, names):
    config, selected, tmp_path = setup
    write_measurements(tmp_path, selected, {n: [100.0] * 30 for n in names})
    with pytest.raises(ValueError, match="measurers"):
        lw.decide(config)


def test_too_few_usable_photos_means_no_decision(setup):
    config, selected, tmp_path = setup
    write_measurements(tmp_path, selected, {"ahsan": selected["short_side"] * 0.2,
                                            "harris": selected["short_side"] * 0.2})
    harris = pd.read_csv(tmp_path / "lw" / "harris.csv")
    harris.loc[:10, "status"] = "unmeasurable"  # 11 left out -> 19 usable, below the minimum of 20
    harris.to_csv(tmp_path / "lw" / "harris.csv", index=False)
    with pytest.raises(ValueError, match="usable"):
        lw.decide(config)
    assert not (tmp_path / "decision.json").exists()


def test_agreement_is_null_not_nan_when_one_measurer_is_constant(setup):
    config, selected, tmp_path = setup
    write_measurements(tmp_path, selected, {"ahsan": selected["short_side"] * 0.2, "harris": [300.0] * 30})
    result = lw.decide(config)
    assert result["agreement_spearman"] is None and result["rho_per_measurer"]["harris"] is None
    assert json.loads((tmp_path / "decision.json").read_text())["agreement_spearman"] is None


def test_model_rater_is_allowed_and_recorded_with_the_verdicts_fragility(setup):
    config, selected, tmp_path = setup
    widths = selected["short_side"] * 0.2
    write_measurements(tmp_path, selected, {"ahsan": widths, "model_rater_a": widths * 0.98})
    lw.decide(config)
    decision = json.loads((tmp_path / "decision.json").read_text())
    assert decision["model_raters"] == ["model_rater_a"] and decision["measurers"] == ["ahsan", "model_rater_a"]
    assert decision["rho_per_measurer"] == {"ahsan": 1.0, "model_rater_a": 1.0}
    assert decision["leave_one_out_rho"] == [1.0, 1.0] and decision["leave_one_out_flips"] == 0


def test_constant_widths_are_refused_not_silently_native(setup):
    config, selected, tmp_path = setup
    write_measurements(tmp_path, selected, {"ahsan": [100.0] * 30, "harris": [100.0] * 30})
    with pytest.raises(ValueError, match="constant"):
        lw.decide(config)


def test_measure_saves_answers_resumes_and_shuffles_per_measurer(setup, monkeypatch):
    config, selected, tmp_path = setup
    root = tmp_path / "photos"
    for photo_id in selected["photo_id"]:
        (root / photo_id).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64)).save(root / photo_id)
    monkeypatch.setattr(lw, "dataset_dir", lambda config, name: root)
    shown = []

    def fake(answers):
        def ask(image, title):
            action, points = next(answers)
            if action != "stop":
                shown.append(title.split()[1])
            return action, points
        return ask

    answers = iter([("accept", [(10, 20), (13, 24)])] * 4 + [("unmeasurable", [])] + [("stop", [])])
    lw.measure(config, "ahsan", ask=fake(answers))
    saved = pd.read_csv(tmp_path / "lw" / "ahsan.csv")
    assert len(saved) == 5 and saved["width_px"].tolist()[:4] == [5.0] * 4  # hypot(3, 4)
    assert saved["status"].tolist() == ["accept"] * 4 + ["unmeasurable"]
    lw.measure(config, "ahsan", ask=fake(iter([("accept", [(0, 0), (6, 8)])] * 25)))  # resumes
    saved = pd.read_csv(tmp_path / "lw" / "ahsan.csv")
    assert len(saved) == 30 and saved["photo_id"].is_unique
    assert shown != selected["photo_id"].tolist()  # not in size order
    ahsan_order, shown[:] = list(shown), []
    lw.measure(config, "harris", ask=fake(iter([("accept", [(0, 0), (6, 8)])] * 30)))
    assert shown != ahsan_order  # each measurer gets their own order


def drive_window(monkeypatch, script, close_like_tk=True):
    """Run click_leaf_edges headless on ``script``: ("click", x, y, button[, dblclick]) or ("key", k).

    With ``close_like_tk``, plt.close fires a close_event first, as the Tk backend does on a real desktop.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import CloseEvent, FigureCanvasBase, KeyEvent, MouseEvent

    def fake_loop(canvas, timeout=0):
        canvas.draw()  # as on screen: the photo's aspect ratio leaves blank margins around it
        ax = canvas.figure.axes[0]
        for step in script:
            if step[0] == "click":
                _, x, y, button, *dbl = step
                # None means a click in the blank margin next to the photo
                px, py = ax.transData.transform((x, y)) if x is not None else (1, 1)
                MouseEvent("button_press_event", canvas, px, py, button=button, dblclick=bool(dbl))._process()
            else:
                KeyEvent("key_press_event", canvas, step[1])._process()

    real_close = plt.close

    def tk_close(fig):
        CloseEvent("close_event", fig.canvas)._process()
        real_close(fig)

    monkeypatch.setattr(FigureCanvasBase, "start_event_loop", fake_loop)
    monkeypatch.setattr(matplotlib, "get_backend", lambda: "tkagg")
    if close_like_tk:
        monkeypatch.setattr(plt, "close", tk_close)
    return lw.click_leaf_edges(Image.fromarray(np.zeros((100, 200, 3), dtype=np.uint8)), "test")


WINDOW = pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")


@WINDOW
def test_window_accepts_two_points_after_undo_and_ignores_stray_input(monkeypatch):
    action, points = drive_window(monkeypatch, [
        ("click", None, None, 1),     # blank margin next to the photo: ignored
        ("click", 250, 50, 1),        # beyond the photo's right edge: ignored
        ("click", 10, 20, 3),         # right button: ignored
        ("click", 10, 20, 1), ("key", "enter"),  # one point only: Enter does nothing yet
        ("click", 10, 20, 1, True),   # the second half of a double-click: ignored
        ("click", 11, 21, 1), ("key", "enter"),  # two points 1.4 px apart: refused as a slip
        ("key", "backspace"),
        ("click", 50, 20, 1), ("key", "backspace"),  # undone again
        ("click", 40, 60, 1), ("click", 90, 90, 1),  # a third click is ignored (two points max)
        ("key", "s"),                 # matplotlib's own shortcuts are switched off
        ("key", "enter"),
    ])
    assert action == "accept"  # and the close_event fired by plt.close afterwards does not undo it
    assert np.allclose(points, [(10, 20), (40, 60)], atol=0.6)


@WINDOW
def test_window_unmeasurable_and_stop(monkeypatch):
    assert drive_window(monkeypatch, [("click", 10, 20, 1), ("key", "u")]) == ("unmeasurable", [])
    assert drive_window(monkeypatch, [("click", 10, 20, 1), ("key", "escape")]) == ("stop", [])
    assert drive_window(monkeypatch, [], close_like_tk=False) == ("stop", [])  # loop ended without an answer


def test_window_refuses_to_run_without_a_screen(monkeypatch):
    import matplotlib

    monkeypatch.setattr(matplotlib, "get_backend", lambda: "agg")
    with pytest.raises(RuntimeError, match="screen"):
        lw.click_leaf_edges(Image.new("RGB", (10, 10)), "test")
