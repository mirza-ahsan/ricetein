"""Stage 14 tables: built from records only, pending when a source is missing, LaTeX that is plain and balanced."""

import ast
import json
import re

import pandas as pd
import pytest
from final_fixture import CLASSES, METHODS, make_fixture

from src import make_tables as mt

REPO = mt.REPO_ROOT


def table(root, name):
    return pd.read_csv(root / "results/tables" / f"final_{name}.csv", dtype=str, keep_default_na=False)


@pytest.fixture
def fixture(tmp_path):
    return make_fixture(tmp_path)


def test_every_table_is_built_from_the_records(fixture):
    provenance = mt.build_all(fixture)["tables"]
    for spec in mt.SPECS:
        assert (fixture / f"results/tables/final_{spec.name}.csv").exists()
        assert (fixture / f"results/tables/final_{spec.name}.tex").exists()
    built = {n for n, t in provenance.items() if t["status"] != "pending"}
    assert built == {"main", "conditions", "per_class", "confusion", "paired_session", "paired_locked", "ablations",
                     "ablations_test", "partition_order", "shortcuts"}
    for name in built:
        for rel, digest in provenance[name]["sources"].items():
            assert mt.sha256(fixture / rel) == digest


def test_main_table_values_come_from_the_source_files(fixture):
    mt.build_all(fixture)
    main = table(fixture, "main").set_index("Method")
    session = pd.read_csv(fixture / "results/tables/session_final.csv").set_index("method")
    for method in METHODS:
        row = session.loc[method]
        assert main.loc[mt.NAMES[method], "Session macro-F1"] == f"{100 * row.macro_f1_mean:.1f} ± {100 * row.macro_f1_sd:.1f}"
    assert main.loc[mt.NAMES["whole224"], "Locked macro-F1"] == "–"      # not in the locked table: no value invented
    # change a source number: the table follows it
    path = fixture / "results/tables/session_final.csv"
    frame = pd.read_csv(path)
    frame.loc[frame.method == "attention", "macro_f1_mean"] = 0.5
    frame.to_csv(path, index=False)
    mt.build_all(fixture)
    assert table(fixture, "main").set_index("Method").loc[mt.NAMES["attention"], "Session macro-F1"].startswith("50.0")


def test_paddy_columns_are_pending_until_stage_10_then_filled(tmp_path):
    root = make_fixture(tmp_path / "a")
    assert mt.build_all(root)["tables"]["main"]["status"] == "partly pending"
    main = table(root, "main")
    assert (main["Paddy native macro-F1"] == "pending: Stage 10").all() and "Paddy other zooms macro-F1" in main
    root = make_fixture(tmp_path / "b", paddy=True)
    mt.build_all(root)
    main = table(root, "main").set_index("Method")
    assert main.loc[mt.NAMES["attention"], "Paddy native macro-F1"] == "60.0 ± 5.0"
    assert main.loc[mt.NAMES["attention"], "Paddy matched zoom macro-F1"] == "60.0 ± 5.0"
    assert main.loc[mt.NAMES["swin_tiny_224"], "Paddy matched zoom macro-F1"] == "–"     # no zoom levels for this method
    assert main.loc[mt.NAMES["swin_tiny_224"], "Paddy native macro-F1"] == "60.0 ± 5.0"
    assert main.loc[mt.NAMES["mean"], "Paddy native macro-F1"] == "–"                    # not in the Paddy table: a dash, not a guess


def test_a_missing_source_gives_a_pending_table_never_no_table(fixture):
    (fixture / "results/tables/session_ablations.csv").unlink()
    provenance = mt.build_all(fixture)["tables"]
    assert provenance["ablations"]["status"] == "pending"
    pending = table(fixture, "ablations")
    assert list(pending.columns) == ["status"] and pending.status[0].startswith("pending: Stage 8")
    assert "pending: Stage 8" in (fixture / "results/tables/final_ablations.tex").read_text()
    for name in ("committee", "explain", "explain_ratings", "cost", "paired_conditions", "paired_committee"):
        assert provenance[name]["status"] == "pending"
    assert table(fixture, "committee").status[0] == "pending: Stage 11"


def test_later_stage_tables_appear_when_their_source_exists(tmp_path):
    root = make_fixture(tmp_path, explain=True, cost=True)
    root_tables = root / "results/tables"
    pd.DataFrame({"method": ["x"], "macro_f1": [0.5], "latency": [0.123456]}).to_csv(root_tables / "committee_results.csv", index=False)
    provenance = mt.build_all(root)["tables"]
    assert {provenance[n]["status"] for n in ("committee", "explain", "cost")} == {"present"}
    assert table(root, "committee").loc[0, "macro_f1"] == "50.0"
    assert table(root, "explain")["Verdict"].iloc[0] == "top-k matters more than random-k"


def test_the_ratings_table_is_pending_until_the_human_ratings_report_exists_then_shows_both_kinds(tmp_path):
    root = make_fixture(tmp_path, explain=True)
    assert mt.build_all(root)["tables"]["explain_ratings"]["status"] == "pending"
    assert table(root, "explain_ratings").status[0] == "pending: Stage 12 (human ratings)"
    pd.DataFrame({"kind": ["human", "model"], "raters": [3, 3], "maps": [50, 50], "yes": [0.4, 0.393], "partly": [0.3, 0.32],
                  "no": [0.3, 0.287], "all_agree": [0.7, 0.88], "kappa": [0.55, 0.88]}).to_csv(
        root / "results/tables/explain_ratings_by_kind.csv", index=False)
    assert mt.build_all(root)["tables"]["explain_ratings"]["status"] == "present"
    shown = table(root, "explain_ratings")
    assert shown["Raters"].tolist() == ["human (3)", "model (3)"]
    assert shown.loc[0, ["Yes (%)", "All raters agree (%)", "Fleiss' kappa"]].tolist() == ["40.0", "70.0", "0.55"]
    assert shown.loc[1, "Yes (%)"] == "39.3"


def test_committee_cost_comes_from_the_stage_13_table(tmp_path):
    root = make_fixture(tmp_path, cost=True)
    tables = root / "results/tables"
    names = ["1 best single model", "2 committee with tile models", "3 committee of whole-image models"]
    pd.DataFrame({"row": names, "test_macro_f1_mean": [0.78, 0.82, 0.80], "cost": ["pending: Stage 13"] * 3}).to_csv(
        tables / "committee_results.csv", index=False)
    pd.DataFrame({"method": ["attention", "committee"], "seconds_per_photo": [0.31, 0.357], "seconds_per_photo_cpu": [5.36, 5.47],
                  "disk_mb": [327.7, 433.9]}).to_csv(tables / "cost_results.csv", index=False)
    provenance = mt.build_all(root)["tables"]["committee"]
    shown = table(root, "committee")
    assert shown["cost"].tolist() == ["0.31 s per photo (GPU), 5.36 s (CPU), 328 MB",
                                      "0.36 s per photo (GPU), 5.47 s (CPU), 434 MB", "not measured"]
    assert provenance["status"] == "present" and "results/tables/cost_results.csv" in provenance["sources"]


def test_paired_tables_copy_the_verdict_words(fixture):
    mt.build_all(fixture)
    source = pd.read_csv(fixture / "results/tables/session_paired.csv")
    shown = table(fixture, "paired_session")
    assert list(shown["Verdict (photo range)"]) == list(source.verdict_photo)
    assert list(shown["Verdict (block range)"]) == list(source.verdict_block)
    assert shown["Rounds won by attention"].iloc[0] == f"{source.round_wins.iloc[0]}/5"


def test_paired_conditions_reads_the_stage_5_branch_columns(fixture):
    frame = pd.DataFrame({"condition": ["convnext_tiny_448"] * 2, "metric": ["macro_f1", "balanced_accuracy"], "difference": [-0.01, -0.02],
                          "photo_low": [-0.03, -0.04], "photo_high": [0.01, 0.0], "block_low": [-0.03, -0.04], "block_high": [0.01, 0.0],
                          "attention_round_wins": [1, 0], "condition_round_wins": [4, 5], "rounds": [5, 5],
                          "verdict_photo": ["no clear difference"] * 2, "verdict_block": ["no clear difference"] * 2})
    frame.to_csv(fixture / "results/tables/paired_conditions.csv", index=False)
    mt.build_all(fixture)
    shown = table(fixture, "paired_conditions")
    assert shown["Rounds won by attention"].tolist() == ["1/5", "0/5"] and shown["Rounds won by the other"].tolist() == ["4/5", "5/5"]
    assert shown["Against"][0] == "ConvNeXt-T, 448 px"


def test_paired_committee_names_both_sides_and_both_round_counts(fixture):
    frame = pd.DataFrame({"A": ["row2_plain"] * 2, "B": ["convnext_tiny_448"] * 2, "metric": ["macro_f1", "balanced_accuracy"],
                          "difference": [0.022, 0.023], "photo_low": [0.012, 0.013], "photo_high": [0.032, 0.033],
                          "block_low": [0.01, 0.009], "block_high": [0.038, 0.041], "round_wins": [4, 4], "rounds": [5, 5],
                          "round_losses": [1, 1], "verdict_photo": ["A beats B", "A beats B"],
                          "verdict_block": ["A beats B", "no clear difference"]})
    frame.to_csv(fixture / "results/tables/paired_committee.csv", index=False)
    provenance = mt.build_all(fixture)["tables"]
    shown = table(fixture, "paired_committee")
    assert shown["Rounds won by the committee"].tolist() == ["4/5", "4/5"] and shown["Rounds won by ConvNeXt-T"].tolist() == ["1/5", "1/5"]
    assert shown["Verdict (photo range)"][0] == "committee beats ConvNeXt-T" and shown["Verdict (block range)"][1] == "no clear difference"
    assert shown["Difference, pooled (points)"][0] == "+2.2" and shown["Photo range"][0] == "[1.2, 3.2]"
    assert "results/tables/paired_committee.csv" in provenance["paired_committee"]["sources"]


def test_per_class_and_confusion_agree_with_the_predictions(fixture):
    mt.build_all(fixture)
    frames = [pd.read_csv(p) for p in sorted((fixture / "runs").glob("stage7_session_attention_r*/test_predictions.csv"))]
    pooled = pd.concat(frames)
    hispa = pooled[pooled.true_class == "hispa"]
    recall = (hispa.pred_class == "hispa").mean()
    per_class = table(fixture, "per_class").set_index("Method")
    assert per_class.loc[mt.NAMES["attention"], "hispa (diagnostic)"] == f"{100 * recall:.1f}"
    confusion = table(fixture, "confusion")
    block = confusion[confusion.Method == mt.NAMES["attention"]].set_index("True class")
    assert int(block.loc["hispa", "pred. hispa"]) == int((hispa.pred_class == "hispa").sum())
    assert per_class.loc["Test photos (n)", "healthy"] == str(int((pooled.true_class == "healthy").sum()))
    assert "(\\textbf" not in (fixture / "results/tables/final_per_class.tex").read_text()  # header is plain, hispa cells are bold
    assert r"\textbf{" in (fixture / "results/tables/final_per_class.tex").read_text()


def test_test_files_of_a_run_outside_its_frozen_plan_are_never_read(fixture):
    plan = fixture / "runs/session_evaluation_plan.json"
    runs = json.loads(plan.read_text())
    runs["runs"].pop("stage7_session_attention_r3")
    plan.write_text(json.dumps(runs))
    with pytest.raises(RuntimeError, match="must not be read"):
        mt.build_all(fixture)
    assert table(fixture, "per_class").status[0].startswith("error:")


def test_a_planned_run_without_test_predictions_is_pending(fixture):
    (fixture / "runs/stage7_session_attention_r2/test_predictions.csv").unlink()
    assert mt.build_all(fixture)["tables"]["per_class"]["status"] == "pending"


def test_ablation_tables_label_validation_and_test(fixture):
    mt.build_all(fixture)
    assert set(table(fixture, "ablations")["Partition"]) == {"validation"}
    test = table(fixture, "ablations_test")
    assert set(test.Variant) == {"otsu_on", "native_scale"}
    assert test["Reading (photo range)"].eq("main beats variant").all() and "Main minus variant, pooled (points)" in test
    values = [json.loads(p.read_text())["macro_f1"] for p in sorted((fixture / "runs").glob("stage7_session_otsu_on_r*/test_metrics.json"))]
    assert not values                                                    # the variants' runs are the 'ablate' ones
    own = [json.loads(p.read_text())["macro_f1"] for p in sorted((fixture / "runs").glob("stage7_session_ablate_otsu_on_r*/test_metrics.json"))]
    shown = test[(test.Variant == "otsu_on") & (test.Metric == "macro-F1")]["Variant test score"].iloc[0]
    assert shown == f"{100 * pd.Series(own).mean():.1f} ± {100 * pd.Series(own).std():.1f}"


def test_ablation_test_rows_are_pending_when_the_report_has_no_test_section(fixture):
    path = fixture / "docs/reports/session_ablation_report.md"
    path.write_text(path.read_text().split("## 2.")[0] + "## 2. Headline ablations on test (main against variant)\n\nThe headline variants are not tested yet.\n")
    assert mt.build_all(fixture)["tables"]["ablations_test"]["status"] == "pending"


def test_latex_is_plain_balanced_and_escaped(tmp_path):
    root = make_fixture(tmp_path, paddy=True, explain=True, cost=True)
    mt.build_all(root)
    for path in sorted((root / "results/tables").glob("final_*.tex")):
        text = path.read_text()
        body = [line for line in text.splitlines() if not line.startswith("%")]
        joined = "\n".join(body)
        stripped = re.sub(r"\\[{}\\]", "", joined)
        assert stripped.count("{") == stripped.count("}"), path.name
        assert not re.search(r"(?<!\\)[%#_^~$](?![a-z])", re.sub(r"\$\\pm\$|\$-\$", "", joined).replace("\\$", "")), path.name
        begin = joined.count("\\begin{table"), joined.count("\\end{table")
        assert begin[0] == begin[1] == 1
        rows = [line for line in body if line.endswith("\\\\")]
        counts = {line.count(" & ") for line in rows}
        assert len(counts) == 1, f"{path.name}: rows with different column counts"
    assert mt.escape("a_b 50% & #1") == r"a\_b 50\% \& \#1" and mt.escape("1 ± 2") == r"1 $\pm$ 2"


def test_no_number_is_typed_in_the_table_code():
    tree = ast.parse((REPO / "src/make_tables.py").read_text())
    floats = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, float)]
    assert floats == []
    big = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, int) and not isinstance(n.value, bool) and n.value > 100]
    assert big == []


def test_readiness_lists_every_commitment_and_writes_the_json(fixture):
    mt.build_all(fixture)
    ready = mt.write_readiness(fixture)
    assert json.loads((fixture / "results/tables/final_readiness.json").read_text()) == ready
    by_id = {c["id"]: c for c in ready["commitments"]}
    assert set(by_id) == {"P1-6", "P1-10", "P1-9", "O6"}
    assert by_id["P1-10"]["status"] == "present" and by_id["P1-9"]["status"] == "manual"
    assert by_id["O6"]["status"] == "pending" and "table:explain" in by_id["O6"]["missing"]
    assert ready["tables"]["cost"] == {"status": "pending", "stage": "Stage 13"}
    assert ready["figures"]["deletion"]["status"] == "not built"        # make_figures has not run in the fixture


def test_every_real_commitment_is_listed():
    ids = {c["id"] for c in mt.commitments(REPO)}
    assert ids == {*(f"P1-{i}" for i in range(1, 18)), *(f"O{i}" for i in range(1, 8))}
    assert set(mt.EVIDENCE) <= ids


def test_the_real_tracked_records_build_every_session_table(tmp_path):
    out = tmp_path / "tables"
    provenance = mt.build_all(REPO, out)["tables"]
    for name in ("main", "per_class", "confusion", "paired_session", "ablations_test", "partition_order", "shortcuts"):
        assert provenance[name]["status"] in ("present", "partly pending"), name
    main = pd.read_csv(out / "final_main.csv", dtype=str).set_index("Method")
    session = pd.read_csv(REPO / "results/tables/session_final.csv").set_index("method")
    assert main.loc[mt.NAMES["attention"], "Session macro-F1"] == mt.pm(session.loc["attention", "macro_f1_mean"], session.loc["attention", "macro_f1_sd"])
    # the pooled per-class recalls, recomputed here from the tracked test predictions
    runs = json.loads((REPO / "runs/session_selection.json").read_text())["methods"]["attention"]["runs"]
    pooled = pd.concat(pd.read_csv(REPO / "runs" / r / "test_predictions.csv") for r in runs)
    expected = f"{100 * (pooled[pooled.true_class == 'hispa'].pred_class == 'hispa').mean():.1f}"
    assert pd.read_csv(out / "final_per_class.csv", dtype=str).set_index("Method").loc[mt.NAMES["attention"], "hispa (diagnostic)"] == expected
    assert list(CLASSES) == mt.CLASSES


def test_a_wrong_run_count_stops_the_table_but_the_others_are_still_written(fixture):
    plan = json.loads((fixture / "runs/session_ablations_plan.json").read_text())
    plan["runs"].pop("stage7_session_ablate_otsu_on_r5")
    (fixture / "runs/session_ablations_plan.json").write_text(json.dumps(plan))
    # the run folder of r5 is still there, but only four runs are in the frozen plan: the name filter finds four
    with pytest.raises(RuntimeError, match="ablations_test"):
        mt.build_all(fixture)
    assert table(fixture, "ablations_test").status[0].startswith("error:")
    assert (fixture / "results/tables/final_main.csv").exists() and (fixture / "results/tables/final_shortcuts.csv").exists()
    assert json.loads((fixture / "results/tables/final_provenance.json").read_text())["tables"]["ablations_test"]["status"] == "error"


def test_the_main_table_shows_convnext_at_448_and_names_conditions(fixture):
    mt.build_all(fixture)
    assert mt.NAMES["convnext_tiny_448"] in set(table(fixture, "main").Method)
    assert mt.condition_name("efficientnetv2s_224_otsu") == "EfficientNetV2-S, 224 px + Otsu"
    assert mt.condition_name("inceptionv3_128") == "InceptionV3, 128 px" and mt.condition_name("something_else") == "something else"


def test_partition_order_tex_defines_its_letters_and_minus_signs_are_math(fixture):
    mt.build_all(fixture)
    tex = (fixture / "results/tables/final_partition_order.tex").read_text()
    assert "A: correct" not in tex and "A: set-up A" in tex and "L: set-up L" in tex
    assert mt.escape("[-1.0, 2.0] top-k") == "[$-$1.0, 2.0] top-k"


def test_the_readiness_check_notices_a_changed_source_and_counts_a_partly_pending_main(fixture):
    mt.build_all(fixture)
    assert mt.write_readiness(fixture)["tables"]["main"]["status"] == "partly pending"
    path = fixture / "results/tables/session_final.csv"
    path.write_text(path.read_text().replace("0.7", "0.6", 1))
    ready = mt.write_readiness(fixture)
    assert ready["tables"]["main"]["status"] == "stale: rebuild" and ready["tables"]["ablations_test"]["status"] == "stale: rebuild"
    assert ready["present_counts"]["tables"] < 10


def test_every_evidence_token_names_a_real_file_table_or_figure():
    ids = {c["id"] for c in mt.commitments(REPO)}
    for needs in mt.EVIDENCE.values():
        for token in needs:
            if token.startswith("table:"):
                assert token[6:] in {spec.name for spec in mt.SPECS}, token
            elif token.startswith("figure:"):
                assert token[7:] in mt.PROMISED_FIGURES, token
            else:
                assert (REPO / token).exists() or token.startswith(("docs/reports/paddy", "docs/reports/explain")), token
    assert set(mt.EVIDENCE) <= ids
