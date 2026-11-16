"""The session-split tuning protocol declared on 2026-11-16 (handbook §24; manuals/HANDOFF.md).

Every method is tuned on validation photos of the session-blocked split, with the search rules of the
2026-11-14 follow-up study (edge rule D6, two studies per head D7, same-head whole-photo controls D8).

    uv run python -m src.session_study <action>

cnn             learning-rate search of the four whole-image models (declared grid, then the edge rule)
probes          L2 search of the two whole-photo probes (declared grid, then the edge rule)
study <name>    one 40-trial study and its five final fits; <name> is a head, or a head followed by _wide
select          write runs/session_selection.json from validation scores only
seeds           the stability seeds of the selected mean, max and attention studies (validation only)
evaluate        freeze runs/session_evaluation_plan.json, then test each selected configuration once
"""

from __future__ import annotations

import argparse
from typing import Any

from src import linear_probe, train_mil
from src.config import load_config
from src.followup import (AGGREGATORS, L2_LADDER, MODELS, ROUNDS, SIZES, best_value, better_study, edge_search,
                          evaluate_once, keyed, read_json, require_complete, search_cnn, study, write_json)
from src.records import get_default_runs_dir, require_clean_code
from src.train_wholeimage import run_name

HEADS = (*AGGREGATORS, "whole224", "whole512")   # the five methods that share the MIL head and its search
STUDIES = tuple(f"{head}{suffix}" for head in HEADS for suffix in ("", "_wide"))


def cnn() -> None:
    search_cnn("session", "session_search_cnn.json")   # the 20 session fits of 2026-11-15 are reused


def probes() -> None:
    for size in SIZES:
        config = load_config(f"configs/session/probe_e1_{size}.yaml")
        declared = config["probe"]["l2_strengths"]

        def mean_score(strength: float) -> float:
            # One strength per call, exactly as the session refit of 2026-11-15 did, so its runs are reused.
            one = {**config, "probe": {**config["probe"], "l2_strengths": [strength]}}
            return linear_probe.run(one)["trials"][0]["mean_val_macro_f1"]

        scores = {strength: mean_score(strength) for strength in declared}
        write_json(f"session_search_probe_{size}.json", keyed(edge_search(scores, mean_score, L2_LADDER, declared)))


def select() -> dict[str, Any]:
    """Every choice below reads validation scores only."""
    runs = get_default_runs_dir()
    methods: dict[str, Any] = {}
    searched_cnn = read_json(runs / "session_search_cnn.json")
    for model in MODELS:
        chosen = best_value({float(lr): score for lr, score in searched_cnn[model].items()})
        config = load_config(f"configs/session/{model}_224.yaml")
        methods[f"{model}_224"] = {"searched": searched_cnn[model], "selected": f"{chosen:g}",
                                   "runs": [run_name(config, chosen, r) for r in ROUNDS]}
    for size in SIZES:
        searched = read_json(runs / f"session_search_probe_{size}.json")
        chosen = best_value({float(strength): score for strength, score in searched.items()})
        methods[f"probe_{size}"] = {"searched": searched, "selected": f"{chosen:g}",
                                    "runs": [f"stage7a_probe_e1_{size}_session_l2{chosen:g}_r{r}" for r in ROUNDS]}
    for head in HEADS:
        methods[head] = better_study(f"session_{head}")
    refit = read_json(runs / "followup_selection.json")["session"]   # the locked-split settings, refitted (D11)
    for method, entry in methods.items():
        entry["changed"] = entry["runs"] != refit.get(method)   # False: these runs were tested on 2026-11-16
    require_complete([run for entry in methods.values() for run in entry["runs"]])
    selection = {"partition": "val", "split": "splits/session_blocked/folds.csv",
                 "code_version": require_clean_code(), "methods": methods}
    write_json("session_selection.json", selection)
    return selection


def seeds() -> None:
    selection = read_json(get_default_runs_dir() / "session_selection.json")
    for name in AGGREGATORS:
        config = load_config(f"configs/mil/{selection['methods'][name]['study']}.yaml")
        train_mil.final(config, [config["seed"], *config["mil"]["stability_seeds"]])


def evaluate_selected() -> None:
    selection = read_json(get_default_runs_dir() / "session_selection.json")
    evaluate_once([run for entry in selection["methods"].values() for run in entry["runs"]],
                  "session_evaluation_plan.json")


ACTIONS = {"cnn": cnn, "probes": probes, "select": select, "seeds": seeds, "evaluate": evaluate_selected}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=[*ACTIONS, "study"])
    parser.add_argument("name", nargs="?", help=f"for 'study': one of {', '.join(STUDIES)}")
    args = parser.parse_args()
    if args.action == "study":
        if args.name not in STUDIES:
            parser.error(f"study needs one of: {', '.join(STUDIES)}")
        study(f"session_{args.name}")
    else:
        ACTIONS[args.action]()
    print(f"SESSION {args.action} {args.name or ''} DONE", flush=True)


if __name__ == "__main__":
    main()
