import json

import pandas as pd

from scripts import deadline_core
from src.config import load_config


def test_report_freezes_before_evaluation_and_reuses_saved_test_results(tmp_path, monkeypatch):
    """Exercise reporting using synthetic photos only; no real held-out data are read."""
    config = load_config("configs/pipeline.yaml")
    config["evaluation"]["bootstrap"]["resamples"] = 20
    classes = config["classes"]
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "deadline_core_training_complete.json").write_text("{}")
    items = [{"name": method, "runs": [f"{method}_r{r}" for r in range(1, 6)]}
             for method in ("attention", "reference")]

    def predictions(name, partition):
        round_ = int(name[-1])
        rows = []
        for index, true in enumerate(classes):
            pred = classes[1] if name.startswith("reference") and index == 0 else true
            row = {"photo_id": f"synthetic_{partition}_{round_}_{index}", "round": round_,
                   "partition": partition, "true_class": true, "pred_class": pred}
            row.update({f"prob_{c}": .7 if c == pred else .1 for c in classes})
            rows.append(row)
        return pd.DataFrame(rows)

    for item in items:
        for name in item["runs"]:
            (runs / name).mkdir()
            predictions(name, "val").to_csv(runs / name / "predictions.csv", index=False)

    evaluated = []

    def freeze(names, plan):
        assert len(names) == 10
        plan.write_text(json.dumps(names))

    def evaluate(name, plan):
        assert plan.is_file()
        evaluated.append(name)
        predictions(name, "test").to_csv(runs / name / "test_predictions.csv", index=False)
        (runs / name / "test_metrics.json").write_text("{}")

    monkeypatch.setattr(deadline_core, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(deadline_core, "get_default_runs_dir", lambda: runs)
    monkeypatch.setattr(deadline_core, "load_config", lambda _: config)
    monkeypatch.setattr(deadline_core, "configurations", lambda: (items, []))
    monkeypatch.setattr(deadline_core.evaluate, "freeze_plan", freeze)
    monkeypatch.setattr(deadline_core.evaluate, "evaluate", evaluate)
    deadline_core.report()
    deadline_core.report()
    assert len(evaluated) == len(set(evaluated)) == 10
    scores = pd.read_csv(tmp_path / "results/tables/deadline_core_scores.csv")
    paired = pd.read_csv(tmp_path / "results/tables/deadline_core_paired.csv")
    assert set(scores.method) == {"attention", "reference"}
    assert set(paired.metric) == {"macro_f1", "balanced_accuracy"}
    assert "pooled_macro_f1_low" in scores.columns
    assert "95% CI" in (tmp_path / "docs/reports/deadline_core_report.md").read_text()
    assert (tmp_path / "results/tables/deadline_report_provenance.json").is_file()
