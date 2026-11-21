"""Human heat-map ratings: strict import, no double import, model rows untouched, separate summaries."""

import pandas as pd
import pytest

from src import heatmap_ratings as hr

ITEMS = hr.ITEMS
MODEL_ROWS = "item_id,rater,rater_kind,rating,note\n" + "".join(
    f'{item},model_rater_{r},model,{answer},"x, y"\n'
    for r in (1, 2, 3)
    for item, answer in zip(ITEMS, ["yes", "partly", "no", "yes", "no"] * 10)
)


NAMES = ("a", "b", "c")   # the returned files are ratings_<anything>.csv; the code never knows who is who


def write_returned(folder, answers=None, **overrides):
    """Three returned files. ``answers`` maps a file name to its 50 answers; ``overrides`` replaces whole texts."""
    folder.mkdir(parents=True, exist_ok=True)
    default = ["yes", "partly", "no", "yes", "no"] * 10
    for name in NAMES:
        rows = (answers or {}).get(name, default)
        text = "item_id,rating,note\n" + "".join(f"{i},{a},\n" for i, a in zip(ITEMS, rows))
        (folder / f"ratings_{name}.csv").write_text(overrides.get(name, text))


@pytest.fixture
def setup(tmp_path):
    ratings = tmp_path / "ratings.csv"
    ratings.write_text(MODEL_ROWS)
    return tmp_path / "returned", ratings, tmp_path / "kit"


def run_import(setup):
    returned, ratings, kit = setup
    return hr.import_ratings(returned, ratings, kit)


def test_import_appends_150_human_rows_and_keeps_the_file_to_id_map_in_the_kit_only(setup):
    write_returned(setup[0], {"b": ["PARTLY "] * 50})
    assert run_import(setup) == 150
    ratings = pd.read_csv(setup[1], dtype=str, keep_default_na=False)
    human = ratings[ratings["rater_kind"] == "human"]
    assert sorted(human["rater"].unique()) == ["human_rater_1", "human_rater_2", "human_rater_3"]
    mapping = pd.read_csv(setup[2] / "rater_map.csv").set_index("file")["rater"]
    assert set(mapping) == set(hr.HUMAN_IDS) and set(human["rater"]) == set(hr.HUMAN_IDS)
    assert (human[human["rater"] == mapping["ratings_b.csv"]]["rating"] == "partly").all()   # case and spaces normalised
    assert not any(name in open(hr.__file__).read().lower() for name in ("ahsan", "umer", "harris"))


@pytest.mark.parametrize(
    "bad, message",
    [
        ({"a": ["yes"] * 49 + ["maybe"]}, r"ratings_a.csv, row 51 \(H50\).*'maybe'"),
        ({"a": ["yes"] * 3 + [""] + ["yes"] * 46}, r"row 5 \(H04\)"),
    ],
)
def test_a_bad_answer_stops_the_import_and_names_the_row(setup, bad, message):
    write_returned(setup[0], bad)
    before = setup[1].read_text()
    with pytest.raises(ValueError, match=message):
        run_import(setup)
    assert setup[1].read_text() == before and not setup[2].exists()


def test_wrong_rows_header_or_missing_file_stop_the_import(setup):
    returned = setup[0]
    write_returned(returned)
    good = (returned / "ratings_c.csv").read_text()
    for text, message in [
        (good.replace("H07,", "H08,", 1), "twice"),
        (good.replace("\nH50,no,", "\n"), "missing"),
        (good + "H51,yes,\n", "unknown"),
        (good.replace("item_id,rating,note", "id,rating,note"), "header"),
    ]:
        (returned / "ratings_c.csv").write_text(text)
        with pytest.raises(ValueError, match=message):
            run_import(setup)
    with pytest.raises(ValueError, match="not found"):
        hr.read_returned(returned / "missing.csv")


def test_odd_excel_exports_are_read_or_refused_clearly(setup):
    returned = setup[0]
    good = "item_id,rating,note\n" + "".join(f"{i},yes,\n" for i in ITEMS)
    write_returned(returned)
    cases = {
        "cp1252": good.replace("H01,yes,", "H01,yes,it\u2019s a lesion").encode("cp1252"),
        "bom_blank_rows_extra_columns": ("\ufeff" + good.replace(",\n", ",,\n").replace("rating,note", "rating,note,")
                                       + ",,,\n,,,\n").encode("utf-8"),
        "newline_in_note": good.replace("H01,yes,", 'H01,yes,"two\r\nlines"').encode(),
    }
    for name, raw in cases.items():
        (returned / "ratings_c.csv").write_bytes(raw)
        frame = hr.read_returned(returned / "ratings_c.csv")
        assert len(frame) == 50 and "\n" not in "".join(frame["note"]), name
    assert hr.read_returned(returned / "ratings_c.csv")["note"][0] == "two lines"
    (returned / "ratings_c.csv").write_bytes(good.encode("utf-16"))
    with pytest.raises(ValueError, match="UTF-16"):
        hr.read_returned(returned / "ratings_c.csv")
    (returned / "ratings_c.csv").write_text(good.replace("H03,yes,", "H03,yes,a, b"))
    with pytest.raises(ValueError, match="not a 3-column CSV"):
        hr.read_returned(returned / "ratings_c.csv")
    (returned / "ratings_c.csv").write_text(good.replace("H03,yes", "H03,nope") + ",,\n")
    with pytest.raises(ValueError, match=r"row 4 \(H03\)"):
        hr.read_returned(returned / "ratings_c.csv")


def test_two_files_instead_of_three_stop_the_import(setup):
    write_returned(setup[0])
    (setup[0] / "ratings_c.csv").unlink()
    with pytest.raises(ValueError, match="expected 3 files"):
        run_import(setup)


def test_a_failed_write_leaves_the_ratings_file_and_the_kit_untouched(setup, monkeypatch):
    write_returned(setup[0])
    before = setup[1].read_text()

    def boom(*args):
        raise OSError("disk full")

    monkeypatch.setattr(hr.os, "replace", boom)
    with pytest.raises(OSError):
        run_import(setup)
    assert setup[1].read_text() == before and not (setup[2] / "rater_map.csv").exists()
    assert not list(setup[1].parent.glob("*.tmp"))


def test_the_same_rater_is_never_imported_twice_and_model_rows_stay_byte_identical(setup):
    write_returned(setup[0])
    run_import(setup)
    after_first = setup[1].read_text()
    assert after_first.startswith(MODEL_ROWS)
    with pytest.raises(ValueError, match="already imported"):
        run_import(setup)
    assert setup[1].read_text() == after_first


def test_a_missing_final_newline_does_not_merge_rows(setup):
    setup[1].write_text(MODEL_ROWS.rstrip("\n"))
    write_returned(setup[0])
    run_import(setup)
    assert len(pd.read_csv(setup[1])) == 300


def sheet():
    return pd.DataFrame({"item_id": ITEMS, "category": ["correct", "wrong", "random", "extra", "correct"] * 10})


def kind_rows(kind, answers):
    return pd.DataFrame(
        [
            {"item_id": i, "rater": f"{kind}_rater_{n}", "rater_kind": kind, "rating": a, "note": ""}
            for n, rows in enumerate(answers, start=1)
            for i, a in zip(ITEMS, rows)
        ]
    )


def test_majority_marks_a_three_way_tie_and_the_comparison_counts_it(setup):
    human = kind_rows("human", [["yes"] * 50, ["yes"] * 25 + ["no"] * 25, ["partly"] * 10 + ["no"] * 40])
    model = kind_rows("model", [["yes"] * 50, ["no"] * 50, ["no"] * 50])
    majority = hr.majority(hr.wide_of(human))
    assert list(majority) == ["yes"] * 25 + ["no"] * 25   # yes,yes,partly / yes,yes,no / yes,no,no
    link = hr.human_against_model(human, model)
    assert link["human_ties"] == 0 and link["model_ties"] == 0
    assert link["equal"] == 25 and link["with_majority"] == 50
    tied = kind_rows("human", [["yes"] * 50, ["partly"] * 50, ["no"] * 50])
    link = hr.human_against_model(tied, model)
    assert (link["human_ties"], link["with_majority"], link["equal"]) == (50, 0, 0)


def test_report_summarises_humans_and_models_separately(setup, tmp_path, monkeypatch):
    human = kind_rows("human", [["yes"] * 50, ["yes"] * 50, ["no"] * 50])
    model = kind_rows("model", [["partly"] * 50, ["partly"] * 50, ["partly"] * 50])
    ratings, examples = tmp_path / "ratings.csv", tmp_path / "examples.csv"
    pd.concat([model, human]).to_csv(ratings, index=False)
    sheet().to_csv(examples, index=False)
    for name, value in (("REPORT", tmp_path / "report.md"), ("BY_KIND", tmp_path / "by_kind.csv"), ("TABLES", tmp_path)):
        monkeypatch.setattr(hr, name, value)
    hr.report(ratings, examples)
    by_kind = pd.read_csv(tmp_path / "by_kind.csv").set_index("kind")
    assert by_kind.loc["human", ["yes", "partly", "no"]].tolist() == pytest.approx([2 / 3, 0, 1 / 3])
    assert by_kind.loc["model", ["yes", "partly", "no"]].tolist() == [0, 1, 0]
    assert by_kind.loc["model", "all_agree"] == 1 and by_kind.loc["human", "all_agree"] == 0
    text = (tmp_path / "report.md").read_text()
    assert "equals the model majority on 0 of 50" in text and "human raters" in text and "model raters" in text
    pd.concat([model, human[human["rater"] != "human_rater_3"]]).to_csv(ratings, index=False)
    assert "| maps |" in text or "maps" in text.split("by example category")[1]
    with pytest.raises(ValueError, match="three human raters"):
        hr.report(ratings, examples)
