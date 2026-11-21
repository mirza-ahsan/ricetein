"""Stage 11: does a committee of different models beat the best single model? (declared in handbook §24, 2026-11-21)

    uv run python -m src.committee select     # validation files only: choose the members, write runs/committee_selection.json
    uv run python -m src.committee plan       # freeze runs/committee_plan.json, before any test file is read
    uv run python -m src.committee evaluate   # test the chosen E2/E3 heads once, then each committee once
    uv run python -m src.committee paddy      # score chosen E2/E3 heads on Paddy once; committees on Paddy
    uv run python -m src.committee tta        # eight-view test-time augmentation of the best whole-image member, once
    uv run python -m src.committee report     # tables and docs/reports/committee_report.md

A committee averages, for each round r, the round-r models of its members. Members and weights are chosen on
session-split validation predictions only. Each committee is tested once (``test_started.json`` marker, never
deleted); nothing is chosen on test or on Paddy Doctor.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import evaluate, metrics, stats
from src import followup_report as fr
from src.config import REPO_ROOT, load_config
from src.followup import read_json
from src.inventory_report import md_table
from src.records import get_default_runs_dir, require_clean_code

CONFIG = "configs/committee.yaml"
TABLE_COLUMNS = ["round", "partition", "photo_id", "true_class", "pred_class"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def path_of(config: dict[str, Any], key: str) -> Path:
    return REPO_ROOT / config["committee"]["files"][key]


def committee_dir(cid: str) -> Path:
    return get_default_runs_dir() / f"committee_{cid}"


# ----------------------------------------------------------------------------------------------- members
def member_runs(config: dict[str, Any]) -> dict[str, list[str]]:
    """Member -> its five run names, round 1 to 5, from the selection files or the declared pattern."""
    c, root = config["committee"], get_default_runs_dir()
    session = read_json(root / "session_selection.json")["methods"]
    conditions = read_json(root / "session_conditions_selection.json")["configurations"]
    found = {}
    for name, spec in c["members"].items():
        if "session" in spec:
            runs = session[spec["session"]]["runs"]
        elif "conditions" in spec:
            runs = conditions[spec["conditions"]]
        else:
            runs = [spec["pattern"].format(round=r) for r in c["rounds"]]
        if len(runs) != len(c["rounds"]):
            raise ValueError(f"{name}: expected {len(c['rounds'])} runs, found {len(runs)}")
        found[name] = list(runs)
    return found


def selected_runs(config: dict[str, Any]) -> dict[str, list[str]]:
    """The member run lists as the validation choice recorded them (what the plan froze)."""
    return read_json(path_of(config, "selection"))["member_runs"]


def read_round(run: str, filename: str, round_: int, partition: str) -> pd.DataFrame:
    """One run's prediction table in a fixed photo order, checked to be one round of one partition."""
    table = pd.read_csv(get_default_runs_dir() / run / filename)
    if set(table["round"]) != {round_} or set(table["partition"]) != {partition}:
        raise ValueError(f"{run}/{filename}: expected round {round_}, partition {partition}")
    return table.sort_values("photo_id", ignore_index=True)


def read_member(runs: list[str], filename: str, partition: str, rounds: list[int]) -> dict[int, pd.DataFrame]:
    return {r: read_round(run, filename, r, partition) for r, run in zip(rounds, runs)}


def average(tables: list[pd.DataFrame], weights: list[float] | None, classes: list[str]) -> pd.DataFrame:
    """The weighted mean of the members' probabilities for the same photos, as a prediction table."""
    first = tables[0]
    for other in tables[1:]:
        if other["photo_id"].tolist() != first["photo_id"].tolist() or other["true_class"].tolist() != first["true_class"].tolist():
            raise ValueError("the members must predict the same photos in the same round")
    mean = np.average([metrics.probabilities(t, classes) for t in tables], axis=0, weights=weights)
    out = first[["round", "partition", "photo_id", "true_class"]].copy()
    out["pred_class"] = np.array(classes)[mean.argmax(axis=1)]
    for k, c in enumerate(classes):
        out[f"prob_{c}"] = mean[:, k]
    return out[[*TABLE_COLUMNS, *[f"prob_{c}" for c in classes]]]


# ----------------------------------------------------------------------------------------------- validation choice
class Validation:
    """Validation predictions of every member: true labels per round, probabilities per member and round."""

    def __init__(self, tables: dict[str, dict[int, pd.DataFrame]], classes: list[str]) -> None:
        self.classes, self.rounds = classes, sorted(next(iter(tables.values())))
        self.members = list(tables)
        first = tables[self.members[0]]
        self.true = {r: first[r]["true_class"].to_numpy() for r in self.rounds}
        for m in self.members:
            for r in self.rounds:
                if tables[m][r]["photo_id"].tolist() != first[r]["photo_id"].tolist():
                    raise ValueError(f"{m} round {r}: other validation photos than {self.members[0]}")
        self.probs = {m: {r: metrics.probabilities(tables[m][r], classes) for r in self.rounds} for m in self.members}
        self.f1 = stats.macro_f1(classes)

    def score(self, members: list[str], weights: list[float] | None = None) -> float:
        """S: mean over the rounds of the validation macro-F1 of the (weighted) probability average."""
        labels, values = np.array(self.classes), []
        for r in self.rounds:
            mean = np.average([self.probs[m][r] for m in members], axis=0, weights=weights)
            values.append(self.f1(self.true[r], labels[mean.argmax(axis=1)]))
        return float(np.mean(values))


def first_best(scored: list[tuple[float, str]], tie: float) -> tuple[float, str]:
    """The highest score; among scores within ``tie`` of it, the member listed first."""
    top = max(s for s, _ in scored)
    return next((s, m) for s, m in scored if s >= top - tie)


def greedy(val: Validation, pool: list[str], choice: dict[str, Any], first: str | None = None) -> dict[str, Any]:
    """Greedy forward selection on S (declared rule): the second member is always added, later ones need
    ``min_gain``, at most ``max_members``."""
    chosen, steps = ([first] if first else []), []
    current = val.score(chosen) if chosen else None
    while len(chosen) < choice["max_members"]:
        options = [m for m in pool if m not in chosen]
        if not options:
            break
        s, best = first_best([(val.score([*chosen, m]), m) for m in options], choice["tie"])
        if len(chosen) >= choice["min_members"] and s - current < choice["min_gain"]:
            break
        steps.append({"added": best, "score": s, "gain": None if current is None else s - current})
        chosen.append(best)
        current = s
    return {"members": chosen, "score": current, "steps": steps}


def weighted(val: Validation, members: list[str], choice: dict[str, Any]) -> dict[str, Any]:
    """Integer weights from the declared grid. The smallest weights win ties; the plain average stays unless
    the best weights raise S by ``min_gain``."""
    plain, best, best_weights = val.score(members), None, None
    for w in sorted(itertools.product(choice["weights"], repeat=len(members)), key=lambda w: (sum(w), w)):
        s = val.score(members, list(w))
        if best is None or s > best + choice["tie"]:
            best, best_weights = s, list(w)
    if best - plain < choice["min_gain"]:
        return {"weights": [1] * len(members), "score": plain, "same_as_plain": True}
    return {"weights": best_weights, "score": best, "same_as_plain": False}


def disagreement(tables: dict[str, dict[int, pd.DataFrame]], committees: dict[str, list[str]]) -> pd.DataFrame:
    """Pairwise disagreement and double-fault rates of the members on validation photos (rounds pooled), and
    for each committee the mean pairwise disagreement and the share of photos where some member is right."""
    pooled = {m: pd.concat([t[r] for r in sorted(t)], ignore_index=True) for m, t in tables.items()}
    true = next(iter(pooled.values()))["true_class"].to_numpy()
    wrong = {m: p["pred_class"].to_numpy() != true for m, p in pooled.items()}
    pred = {m: p["pred_class"].to_numpy() for m, p in pooled.items()}
    rows = [{"kind": "pair", "a": a, "b": b, "disagreement": float((pred[a] != pred[b]).mean()),
             "double_fault": float((wrong[a] & wrong[b]).mean())} for a, b in itertools.combinations(pooled, 2)]
    for cid, members in committees.items():
        pairs = [(pred[a] != pred[b]).mean() for a, b in itertools.combinations(members, 2)]
        rows.append({"kind": "committee", "a": cid, "b": "+".join(members), "disagreement": float(np.mean(pairs)),
                     "double_fault": float(np.logical_and.reduce([wrong[m] for m in members]).mean())})
    return pd.DataFrame(rows)


def choose(config: dict[str, Any]) -> dict[str, Any]:
    """The whole validation-only choice: members of rows 2 and 3, weights, TTA member, disagreement."""
    c, classes = config["committee"], config["classes"]
    runs = member_runs(config)
    tables = {m: read_member(r, "predictions.csv", "val", c["rounds"]) for m, r in runs.items()}
    val, choice = Validation(tables, classes), c["choice"]
    row2 = greedy(val, c["pools"]["A"], choice)
    forced = not any(c["members"][m]["tile"] for m in row2["members"])
    if forced:
        row2 = greedy(val, c["pools"]["A"], choice, first=c["forced_first"])
    row3 = greedy(val, c["pools"]["B"], choice)
    committees = {}
    for row, found in ((2, row2), (3, row3)):
        plain = {"row": row, "variant": "plain", "members": found["members"], "weights": [1] * len(found["members"]),
                 "score": found["score"], "steps": found["steps"], "forced_first": row == 2 and forced}
        tuned = weighted(val, found["members"], choice)
        committees[f"row{row}_plain"] = plain
        committees[f"row{row}_weighted"] = {**plain, "variant": "weighted", **tuned}
    singles = [(val.score([m]), m) for m in c["pools"][c["tta"]["pool"]]]
    tta_member = first_best(singles, choice["tie"])[1]
    table = disagreement(tables, {cid: e["members"] for cid, e in committees.items() if e["variant"] == "plain"})
    return {"partition": "val", "split": c["split_file"], "pools": c["pools"],
            "member_runs": runs, "member_scores": {m: val.score([m]) for m in val.members},
            "committees": committees, "tta_member": tta_member,
            "disagreement": table.to_dict("records")}


def without_version(record: dict[str, Any]) -> dict[str, Any]:
    """A record as JSON would store it, minus the code version (the same choice may be re-run at another commit)."""
    return json.loads(json.dumps({k: v for k, v in record.items() if k != "code_version"}))


def select(config: dict[str, Any]) -> None:
    result = {"code_version": require_clean_code(), **choose(config)}
    path = path_of(config, "selection")
    if not path.exists():
        path.write_text(json.dumps(result, indent=2))
    elif without_version(read_json(path)) != without_version(result):
        raise ValueError(f"{path} exists with another choice; the declared choice is never replaced")
    pd.DataFrame(result["disagreement"]).to_csv(path_of(config, "disagreement"), index=False)
    for cid, e in result["committees"].items():
        print(f"{cid}: {', '.join(f'{m} x{w}' for m, w in zip(e['members'], e['weights']))}; validation S {e['score']:.4f}"
              + ("; same as plain" if e.get("same_as_plain") else ""))
    print(f"TTA member: {result['tta_member']}")


def tested_committees(selection: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The committees that are tested: a weighted variant equal to its plain committee is not tested again."""
    return {cid: e for cid, e in selection["committees"].items() if not e.get("same_as_plain")}


# ----------------------------------------------------------------------------------------------- plan and test
def has_test(run: str) -> bool:
    return (get_default_runs_dir() / run / "test_predictions.csv").exists()


def head_entry(run: str) -> dict[str, str]:
    folder = get_default_runs_dir() / run
    record = read_json(folder / "record.json")
    if record["status"] != "completed" or record["code_version"] in ("unknown",) or record["code_version"].endswith("-dirty"):
        raise ValueError(f"{run}: incomplete or untraceable training run")
    return {"record_sha256": evaluate.digest(folder / "record.json"),
            "checkpoint_sha256": evaluate.digest(evaluate.checkpoint(folder)),
            "split_checksum": record["split_checksum"]}


def plan(config: dict[str, Any]) -> None:
    """Freeze what is tested, before any test file is opened. Heads that were never tested (the E2 and E3
    heads) enter in the ``src.evaluate`` plan format; already tested members are pinned by file checksum."""
    version, selection_path, path = require_clean_code(), path_of(config, "selection"), path_of(config, "plan")
    if path.exists():   # frozen once; later calls only check that the choice it rests on is unchanged
        if read_json(path)["selection_sha256"] != sha256(selection_path):
            raise ValueError(f"{path} was frozen for another selection; it is never replaced")
        print(f"{path}: already frozen")
        return
    selection = read_json(selection_path)
    committees = {cid: {"members": e["members"], "weights": e["weights"],
                        "runs": {m: selection["member_runs"][m] for m in e["members"]}}
                  for cid, e in tested_committees(selection).items()}
    root = get_default_runs_dir()
    used = sorted({run for e in committees.values() for runs in e["runs"].values() for run in runs})
    untested = [run for run in used if not has_test(run)]
    result = {"selection_partition": "val", "code_version": version, "selection_sha256": sha256(selection_path),
              "tta_member": selection["tta_member"], "committees": committees,
              "runs": {run: head_entry(run) for run in untested},
              "tested_members_test_sha256": {run: sha256(root / run / "test_predictions.csv") for run in used if has_test(run)},
              "validation_sha256": {run: sha256(root / run / "predictions.csv") for run in used}}
    with path.open("x") as out:
        json.dump(result, out, indent=2)
    print(f"{path}: {len(committees)} committees, {len(untested)} untested head runs")


def summary_of(table: pd.DataFrame, classes: list[str]) -> dict[str, Any]:
    return {"partition": "test", **metrics.scores(table["true_class"], table["pred_class"], classes)}


def test_committee(config: dict[str, Any], cid: str, spec: dict[str, Any], plan_path: Path) -> None:
    """Test one committee, once. The marker is created before any test file is read."""
    folder, classes, rounds = committee_dir(cid), config["classes"], config["committee"]["rounds"]
    if (folder / "test_predictions.csv").exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    try:
        with (folder / "test_started.json").open("x") as out:
            json.dump({"evaluation_code": require_clean_code(), "plan_sha256": sha256(plan_path)}, out)
    except FileExistsError:
        raise ValueError(f"{cid}: started but has no test predictions; stop and report, do not retry") from None
    plan = read_json(plan_path)
    for run, digest in plan["tested_members_test_sha256"].items():
        if sha256(get_default_runs_dir() / run / "test_predictions.csv") != digest:
            raise ValueError(f"{run}: test predictions changed after the plan was frozen")
    members = {m: {part: read_member(spec["runs"][m], name, part, rounds)
                   for part, name in (("val", "predictions.csv"), ("test", "test_predictions.csv"))}
               for m in spec["members"]}
    out_tables = {part: pd.concat([average([members[m][part][r] for m in spec["members"]], spec["weights"], classes)
                                   for r in rounds], ignore_index=True) for part in ("val", "test")}
    out_tables["val"].to_csv(folder / "predictions.csv", index=False)
    out_tables["test"].to_csv(folder / "test_predictions.csv", index=False)
    (folder / "test_metrics.json").write_text(json.dumps(
        {**summary_of(out_tables["test"], classes), "per_round": {
            r: metrics.scores(g["true_class"], g["pred_class"], classes)["macro_f1"]
            for r, g in out_tables["test"].groupby("round")}}, indent=2))
    (folder / "committee.json").write_text(json.dumps({"id": cid, **spec}, indent=2))


def evaluate_all(config: dict[str, Any]) -> None:
    require_clean_code()
    plan_path = path_of(config, "plan")
    plan = read_json(plan_path)
    if plan["selection_sha256"] != sha256(path_of(config, "selection")):
        raise ValueError("the selection changed after the plan was frozen")
    for run in plan["runs"]:
        if has_test(run):
            continue
        evaluate.evaluate(run, plan_path)
        print(f"{run}: test evaluated", flush=True)
    for cid, spec in plan["committees"].items():
        test_committee(config, cid, spec, plan_path)
        print(f"{cid}: tested", flush=True)


# ----------------------------------------------------------------------------------------------- Paddy Doctor
def two_way(low: float, high: float, wins: int, rounds: int, comparison: dict[str, Any]) -> str:
    """The declared rule read in both directions (A is the committee, B the single model)."""
    need = comparison["min_round_wins"]
    above = low > 0 if comparison["ci_excludes_zero"] else True
    below = high < 0 if comparison["ci_excludes_zero"] else True
    if above and wins >= need:
        return "A beats B"
    if below and wins <= rounds - need:
        return "B beats A"
    return "no clear difference"


def paddy_path(run: str, tile: bool, zoom: str) -> Path:
    from src import paddy_zero_shot as zp

    return zp.paths(run, zoom if tile else None)[0]


def paddy_member_rounds(config: dict[str, Any], member: str, runs: list[str]) -> dict[int, pd.DataFrame]:
    """Round -> the member's Paddy predictions at the declared zoom (Stage 10 files, or the new E2/E3 files)."""
    c, ids = config["committee"], None
    out = {}
    for r, run in zip(c["rounds"], runs):
        table = pd.read_csv(paddy_path(run, c["members"][member]["tile"], c["paddy"]["zoom"]))
        ids = ids or table["photo_id"].tolist()
        if set(table["round"]) != {r} or table["photo_id"].tolist() != ids:
            raise ValueError(f"{run}: Paddy predictions are not round {r} on the shared photo order")
        out[r] = table
    return out


def paddy_new_heads(config: dict[str, Any], plan_path: Path) -> None:
    """Score the chosen E2 and E3 heads on Paddy, once each, with the Stage 10 head code."""
    import torch

    from src import paddy_zero_shot as zp
    from src import train_wholeimage as tw
    from src.seeding import seed_everything

    c, plan = config["committee"], read_json(plan_path)
    for member in sorted({m for e in plan["committees"].values() for m in e["members"]}):
        if "paddy" not in c["members"][member]:
            continue
        runs, pconfig, zoom = next(e["runs"][member] for e in plan["committees"].values() if member in e["runs"]), \
            load_config(c["members"][member]["paddy"]), c["paddy"]["zoom"]
        todo = zp.pending(runs, zoom)
        if not todo:
            continue
        seed_everything(pconfig["seed"], pconfig["determinism"])
        zp.verify(pconfig, pconfig["paddy"]["caches"]["tiles"][zoom])
        probs = zp.predict_tile_heads(pconfig, zoom, todo, torch.device("cuda"))
        folds = zp.truth(pconfig)
        for run in todo:
            table_path, marker = zp.paths(run, zoom)
            with marker.open("x") as out:
                json.dump({"code_version": require_clean_code(), "plan_sha256": sha256(plan_path)}, out, indent=2)
            table = tw.prediction_table(folds, np.arange(len(folds)), probs[run], zp.record(run)["round"], "paddy",
                                        config["classes"])
            partial = table_path.with_name(table_path.name + ".tmp")
            table.to_csv(partial, index=False)
            partial.replace(table_path)
        print(f"{member}: {len(todo)} runs scored on Paddy at {zoom}", flush=True)


def paddy_scores(config: dict[str, Any], rounds_a: dict[int, pd.DataFrame], rounds_b: dict[int, pd.DataFrame]) -> dict[str, Any]:
    """Round committees A and B on Paddy: mean ± SD of each metric, the five-round committees, and A against B
    (paired photo bootstrap on the five-round committees; rounds won counted per round)."""
    classes, rounds = config["classes"], config["committee"]["rounds"]
    boot, rule = config["evaluation"]["bootstrap"], config["evaluation"]["comparison"]
    scorers = fr.scorers(classes)
    own = {m: [scorers[m](t["true_class"], t["pred_class"]) for t in (rounds_a[r] for r in rounds)] for m in scorers}
    other = {m: [scorers[m](t["true_class"], t["pred_class"]) for t in (rounds_b[r] for r in rounds)] for m in scorers}
    pooled_a, pooled_b = (average([t[r] for r in rounds], None, classes) for t in (rounds_a, rounds_b))
    row: dict[str, Any] = {}
    for m, scorer in scorers.items():
        paired = stats.paired(pooled_a["true_class"], pooled_a["pred_class"], pooled_b["pred_class"], scorer, boot, config["seed"])
        wins = sum(a > b for a, b in zip(own[m], other[m]))
        row.update({f"{m}_mean": float(np.mean(own[m])), f"{m}_sd": float(np.std(own[m], ddof=1)),
                    f"{m}_five_round": scorer(pooled_a["true_class"], pooled_a["pred_class"]),
                    f"{m}_difference": paired["difference"], f"{m}_low": paired["low"], f"{m}_high": paired["high"],
                    f"{m}_round_wins": int(wins),
                    f"{m}_verdict": two_way(paired["low"], paired["high"], wins, len(rounds), rule)})
    return row


def paddy_all(config: dict[str, Any]) -> None:
    require_clean_code()
    plan_path = path_of(config, "plan")
    paddy_new_heads(config, plan_path)
    plan, c = read_json(plan_path), config["committee"]
    runs = selected_runs(config)
    single = c["paddy"]["comparison_member"]
    base = paddy_member_rounds(config, single, runs[single])
    rows = [{"committee": single, "against": single, **paddy_scores(config, base, base)}]
    rows[0].update({k: "–" if k.endswith("verdict") else np.nan for k in rows[0] if k.endswith(("_difference", "_low", "_high", "_round_wins", "_verdict"))})
    for cid, spec in plan["committees"].items():
        members = {m: paddy_member_rounds(config, m, spec["runs"][m]) for m in spec["members"]}
        mine = {r: average([members[m][r] for m in spec["members"]], spec["weights"], config["classes"])
                for r in c["rounds"]}
        rows.append({"committee": cid, "against": single, **paddy_scores(config, mine, base)})
    pd.DataFrame(rows).to_csv(path_of(config, "paddy"), index=False)
    print(f"{len(rows)} committees scored on Paddy at {c['paddy']['zoom']}")


# ----------------------------------------------------------------------------------------------- test-time augmentation
def views(x: Any) -> list[Any]:
    """The eight flip and rotation views of an image batch (n, c, h, w): four 90 degree turns, each also mirrored."""
    turned = [x.rot90(k, dims=(2, 3)) for k in range(4)]
    return turned + [v.flip(3) for v in turned]


def tta_run(run: str, device: Any) -> pd.DataFrame:
    """Probabilities averaged over the eight views for one whole-image round-model, on its own test photos."""
    import torch

    from src import train_wholeimage as tw
    from src.seeding import seed_everything
    from src.split import read_folds

    record = read_json(get_default_runs_dir() / run / "record.json")
    config = evaluate.restore_size_keys(record["config"])
    seed_everything(config["seed"], config["wholeimage"]["determinism"])
    folds, _ = read_folds(config)
    model = tw.build_model(config, config["wholeimage"]["size"])
    model.load_state_dict(torch.load(get_default_runs_dir() / run / "model.pt", map_location="cpu", weights_only=True))
    model.eval().to(device)
    mean = torch.tensor(config["normalisation"]["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(config["normalisation"]["std"], device=device).view(1, 3, 1, 1)
    rows, probs = [], []
    with torch.inference_mode():
        for images, _, row in tw.loaders(config, folds, config["round"], ("test",))["test"]:
            x = (images.to(device).float() / 255 - mean) / std
            total = 0
            for v in views(x):
                with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    total = total + torch.softmax(model(v).float(), dim=1)
            probs.append((total / 8).cpu())
            rows.append(row)
    return tw.prediction_table(folds, torch.cat(rows).numpy(), torch.cat(probs).numpy(), config["round"], "test",
                               config["classes"])


def tta_all(config: dict[str, Any]) -> None:
    import torch

    require_clean_code()
    plan = read_json(path_of(config, "plan"))
    runs = selected_runs(config)[plan["tta_member"]]
    for run in runs:
        folder = get_default_runs_dir() / run
        if (folder / "test_tta_predictions.csv").exists():
            continue
        try:
            with (folder / "test_tta_started.json").open("x") as out:
                json.dump({"code_version": require_clean_code(), "plan_sha256": sha256(path_of(config, "plan")),
                           "views": 8}, out)
        except FileExistsError:
            raise ValueError(f"{run}: TTA started but has no predictions; stop and report, do not retry") from None
        table = tta_run(run, torch.device("cuda"))
        partial = folder / "test_tta_predictions.csv.tmp"
        table.to_csv(partial, index=False)
        partial.replace(folder / "test_tta_predictions.csv")
        print(f"{run}: TTA scored", flush=True)


# ----------------------------------------------------------------------------------------------- report
def pts(x: float) -> str:
    return "–" if pd.isna(x) else f"{100 * x:+.1f}"


def pct(x: float) -> str:
    return "–" if pd.isna(x) else f"{100 * x:.1f}"


def plus_minus(mean: float, sd: float) -> str:
    return f"{pct(mean)} ± {pct(sd)}"


def rng(low: float, high: float) -> str:
    return f"[{pts(low)}, {pts(high)}]"


def committee_table(cid: str) -> pd.DataFrame:
    folder = committee_dir(cid)
    return pd.concat([pd.read_csv(folder / "predictions.csv"), pd.read_csv(folder / "test_predictions.csv")], ignore_index=True)


def readable(verdict: str, a: str, b: str) -> str:
    return verdict.replace("A beats B", f"{a} beats {b}").replace("B beats A", f"{b} beats {a}")


def paired_with_two_way(a: pd.DataFrame, b: pd.DataFrame, a_name: str, b_name: str, blocks: pd.Series,
                        config: dict[str, Any]) -> pd.DataFrame:
    """A against B on test (both ranges, both metrics) with the declared rule read in both directions."""
    rule = config["evaluation"]["comparison"]
    rows = fr.paired_rows(a, {b_name: b}, blocks, config)
    rows.insert(0, "A", a_name)
    for kind in ("photo", "block"):
        rows[f"verdict_{kind}"] = [two_way(r[f"{kind}_low"], r[f"{kind}_high"], r["round_wins"], r["rounds"], rule)
                                   for _, r in rows.iterrows()]
    return rows


def tta_tables(config: dict[str, Any], blocks: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scores of the TTA member with one view and with eight views, and the paired test of the two."""
    plan = read_json(path_of(config, "plan"))
    runs = selected_runs(config)[plan["tta_member"]]
    files = [get_default_runs_dir() / run / "test_tta_predictions.csv" for run in runs]
    if not all(f.exists() for f in files):
        return pd.DataFrame(), pd.DataFrame()
    plain = fr.read_predictions(runs)
    augmented = pd.concat([plain[plain["partition"] == "val"], *[pd.read_csv(f) for f in files]], ignore_index=True)
    scores = pd.DataFrame([{"name": f"{plan['tta_member']}, one view", **fr.method_scores(plain, blocks, config)},
                           {"name": f"{plan['tta_member']}, eight views", **fr.method_scores(augmented, blocks, config)}])
    return scores, paired_with_two_way(augmented, plain, "eight views", "one view", blocks, config)


def build_tables(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Every number of the report, from saved predictions only."""
    c, blocks = config["committee"], fr.block_of()
    selection, plan = read_json(path_of(config, "selection")), read_json(path_of(config, "plan"))
    single = c["best_single"]
    tables = {single: fr.read_predictions(selection["member_runs"][single])}
    tables |= {cid: committee_table(cid) for cid in plan["committees"]}
    scores = pd.DataFrame.from_dict({name: fr.method_scores(t, blocks, config) for name, t in tables.items()},
                                    orient="index").rename_axis("name").reset_index()
    pairs = [(cid, single) for cid in plan["committees"]]
    if {"row2_plain", "row3_plain"} <= set(plan["committees"]):
        pairs.append(("row2_plain", "row3_plain"))
    paired = pd.concat([paired_with_two_way(tables[a], tables[b], a, b, blocks, config) for a, b in pairs], ignore_index=True)
    paddy_file = path_of(config, "paddy")
    tta_scores, tta_paired = tta_tables(config, blocks)
    return {"scores": scores, "paired": paired, "tta_scores": tta_scores, "tta_paired": tta_paired,
            "paddy": pd.read_csv(paddy_file) if paddy_file.exists() else pd.DataFrame()}


ROW_NAMES = {"row2_plain": "2 committee with tile models", "row2_weighted": "2 committee with tile models, weighted",
             "row3_plain": "3 committee of whole-image models", "row3_weighted": "3 committee of whole-image models, weighted"}


def shown(name: str, single: str = "attention_e1") -> str:
    """The name of a committee row (or member) as the report prints it."""
    return "1 best single model (attention, E1)" if name == single else ROW_NAMES.get(name, name)


def label(members: list[str], weights: list[int]) -> str:
    return " + ".join(m if w == 1 else f"{m} x{w}" for m, w in zip(members, weights))


def results_rows(config: dict[str, Any], tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """committee_results.csv for src/make_tables.py: score columns are fractions (their names hold f1 or
    accuracy), every other column is text."""
    c, plan = config["committee"], read_json(path_of(config, "plan"))
    scores, paired = tables["scores"].set_index("name"), tables["paired"]
    paddy = tables["paddy"].set_index("committee") if not tables["paddy"].empty else None
    single, rows = c["best_single"], []
    for name in [single, *plan["committees"]]:
        spec = plan["committees"].get(name)
        mine = paired[(paired["A"] == name) & (paired["B"] == single) & (paired["metric"] == "macro_f1")]
        row = {"row": "1 best single model" if spec is None else ROW_NAMES[name],
               "members": single if spec is None else label(spec["members"], spec["weights"]),
               **{f"test_{m}_{k}": scores.loc[name, f"{m}_{k}"] for m in ("macro_f1", "balanced_accuracy") for k in ("mean", "sd")}}
        if paddy is not None and name in paddy.index:
            row |= {f"paddy_{m}_{k}": paddy.loc[name, f"{m}_{k}"] for m in ("macro_f1", "balanced_accuracy") for k in ("mean", "sd")}
        shown = {"difference_vs_attention_points": "–", "photo_range": "–", "block_range": "–", "rounds_won": "–",
                 "verdict_photo": "–", "verdict_block": "–"}
        if len(mine):
            m = mine.iloc[0]
            shown = {"difference_vs_attention_points": pts(m["difference"]), "photo_range": rng(m["photo_low"], m["photo_high"]),
                     "block_range": rng(m["block_low"], m["block_high"]), "rounds_won": f"{m['round_wins']}/{m['rounds']}",
                     "verdict_photo": readable(m["verdict_photo"], "committee", "attention"),
                     "verdict_block": readable(m["verdict_block"], "committee", "attention")}
        rows.append({**row, **shown, "cost": c["cost"]})
    return pd.DataFrame(rows)


def paired_view(paired: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "A": [shown(a) for a in paired["A"]], "B": [shown(b) for b in paired["B"]], "metric": paired["metric"],
        "difference (points)": [pts(v) for v in paired["difference"]],
        "photo range": [rng(a, b) for a, b in zip(paired["photo_low"], paired["photo_high"])],
        "block range": [rng(a, b) for a, b in zip(paired["block_low"], paired["block_high"])],
        "rounds won (A)": [f"{w}/{n}" for w, n in zip(paired["round_wins"], paired["rounds"])],
        "verdict, photo range": [readable(v, "A", "B") for v in paired["verdict_photo"]],
        "verdict, block range": [readable(v, "A", "B") for v in paired["verdict_block"]]})


def score_view(scores: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"model": [shown(n) for n in scores["name"]],
                         "macro-F1 (mean ± SD)": [plus_minus(m, s) for m, s in zip(scores["macro_f1_mean"], scores["macro_f1_sd"])],
                         "balanced acc. (mean ± SD)": [plus_minus(m, s) for m, s in zip(scores["balanced_accuracy_mean"], scores["balanced_accuracy_sd"])]})


def paddy_view(paddy: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "model": [shown(n) for n in paddy["committee"]],
        "macro-F1 (mean ± SD)": [plus_minus(m, s) for m, s in zip(paddy["macro_f1_mean"], paddy["macro_f1_sd"])],
        "balanced acc. (mean ± SD)": [plus_minus(m, s) for m, s in zip(paddy["balanced_accuracy_mean"], paddy["balanced_accuracy_sd"])],
        "five-round committee macro-F1": [pct(v) for v in paddy["macro_f1_five_round"]],
        "difference to attention (points)": [pts(v) for v in paddy["macro_f1_difference"]],
        "photo range": [rng(a, b) for a, b in zip(paddy["macro_f1_low"], paddy["macro_f1_high"])],
        "rounds won": [f"{w}/5" for w in paddy["macro_f1_round_wins"]],
        "verdict": [readable(v, "committee", "attention") for v in paddy["macro_f1_verdict"]]})


def report_text(config: dict[str, Any], tables: dict[str, pd.DataFrame]) -> str:
    selection, c = read_json(path_of(config, "selection")), config["committee"]
    chosen = pd.DataFrame([{"committee": shown(cid), "members (weight)": label(e["members"], e["weights"]), "S, validation": pct(e["score"]),
                            "note": "same as plain, not tested again" if e.get("same_as_plain") else ("E1 attention forced first" if e.get("forced_first") else "")}
                           for cid, e in selection["committees"].items()])
    singles = pd.DataFrame({"member": list(selection["member_scores"]),
                            "validation macro-F1, mean over rounds": [pct(v) for v in selection["member_scores"].values()]})
    disagree = pd.DataFrame(selection["disagreement"])
    disagree = disagree.assign(disagreement=[pct(v) for v in disagree["disagreement"]],
                               double_fault=[pct(v) for v in disagree["double_fault"]])
    results = score_view(tables["scores"])
    paddy = md_table(paddy_view(tables["paddy"])) if not tables["paddy"].empty else "Not scored yet."
    tta = ("TTA was not run." if tables["tta_scores"].empty else
           md_table(score_view(tables["tta_scores"])) + "\n\n" + md_table(paired_view(tables["tta_paired"])))
    return f"""# Committee: does combining models beat the best single model? (Stage 11)

Built by `src/committee.py` from `runs/`; do not edit by hand. Declaration: handbook §24, 2026-11-21 ("Stage 11
committee"), written before any committee number. All scores are percentages on the session-blocked split
(`splits/session_blocked/folds.csv`) unless the heading says Paddy Doctor.

**Which photos each number comes from.** Sections 1 and 5 are **validation** photos (used for the choice).
Sections 2, 3 and 6 are **test** photos, each committee tested once (`runs/committee_*/test_started.json`).
Section 4 is **Paddy Doctor** (8,086 photos, zero-shot, native zoom); nothing was chosen or trained on it.

**Limits to read first**

- The test scores of the already tested members were known before the choice rule was written (disclosed in the
  declaration). The rule uses validation predictions only and was fixed before any committee number was computed.
- A photo that is a validation photo in one round is a test photo in another round, so settings chosen on
  validation are not independent of the pooled test photos (the limit of the session report applies).
- Members and committees use one training seed (seed 0). "Rounds won" counts the rounds in which the committee of
  that round beats the single model of that round.
- The cost column is not measured here: a committee costs the sum of its members; Stage 13 measures it.
- Test-time augmentation was done for the best whole-image member only; it was **not** done for attention (new
  tile descriptions of every flipped and rotated test photo would be needed).

## 1. Choice on validation photos

Single members (validation macro-F1, mean over the five rounds):

{md_table(singles)}

Chosen committees (S = mean over the rounds of the validation macro-F1 of the average):

{md_table(chosen)}

## 2. Test results (session-blocked split)

{md_table(results)}

The table file `results/tables/committee_results.csv` also holds balanced accuracy, the Paddy columns and the paired
verdicts of each row against attention.

## 3. Paired tests on test photos (A against B)

Rule (the decision rule, read in both directions): A beats B if the range is above 0 and A wins at least 4 of 5 rounds;
B beats A if the range is below 0 and A wins at most 1; otherwise no clear difference.

{md_table(paired_view(tables["paired"]))}

## 4. Paddy Doctor (native zoom, zero-shot)

Committee rows: the round committees' mean ± SD, the five-round committee, and the paired test against attention
(photo range only: Paddy photos have no capture blocks).

{paddy}

## 5. Disagreement of the members on validation photos (rounds pooled)

`pair`: share of photos where two members predict different classes, and where both are wrong. `committee`: mean
pairwise disagreement of its members, and the share of photos where every member is wrong.

{md_table(disagree)}

## 6. Test-time augmentation (eight views), best whole-image member

{tta}

Cost column: {c["cost"]}.
"""


def report(config: dict[str, Any]) -> None:
    tables = build_tables(config)
    results = results_rows(config, tables)
    tables["paired"].to_csv(path_of(config, "paired"), index=False)
    if not tables["tta_scores"].empty:
        pd.concat([tables["tta_scores"], tables["tta_paired"]], ignore_index=True).to_csv(path_of(config, "tta"), index=False)
    results.to_csv(path_of(config, "results"), index=False)
    path_of(config, "report").write_text(report_text(config, tables), encoding="utf-8")
    print(f"report: {len(results)} rows, {len(tables['paired'])} paired rows")


ACTIONS = {"select": select, "plan": plan, "evaluate": evaluate_all, "paddy": paddy_all, "tta": tta_all, "report": report}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=ACTIONS)
    parser.add_argument("--config", default=CONFIG)
    args = parser.parse_args()
    ACTIONS[args.action](load_config(args.config))
    print(f"COMMITTEE {args.action} DONE", flush=True)


if __name__ == "__main__":
    main()
