"""Stage 5 other conditions on the session-blocked split (declared 2026-11-20, handbook §24).

Every model keeps the learning rate that was selected at 224 px on session-split validation photos
(runs/session_selection.json). It is trained again at 128 px and at 224 px with Otsu background removal. The
model with the best validation score at 224 px is also trained at 448 px. Nothing is searched here, and the
choice of the 448 px model reads validation scores only.

    uv run python -m src.session_conditions <action>

fit        the 40 runs of the two conditions and the 5 runs at 448 px; writes runs/session_conditions_selection.json
evaluate   freeze runs/session_conditions_plan.json, then test each of these 45 runs once
"""

from __future__ import annotations

import argparse
from typing import Any

from src import stage5
from src.config import load_config
from src.followup import MODELS, ROUNDS, evaluate_once, read_json, write_json
from src.records import get_default_runs_dir, require_clean_code
from src.session_study import require_intact

CONDITIONS = ("128", "224_otsu")
PLAN = "session_conditions_plan.json"
SELECTION = "session_conditions_selection.json"


def selected_at_224(selection: dict[str, Any]) -> dict[str, tuple[float, float]]:
    """Each model's selected learning rate and its mean validation macro-F1 at 224 px."""
    chosen = {}
    for model in MODELS:
        entry = selection["methods"][f"{model}_224"]
        chosen[model] = (float(entry["selected"]), entry["searched"][entry["selected"]])
    return chosen


def best_model(chosen: dict[str, tuple[float, float]]) -> str:
    """The highest validation score; the model listed first in MODELS wins a tie."""
    return max(MODELS, key=lambda model: chosen[model][1])


def planned(best: str) -> list[tuple[str, str]]:
    """The (model, condition) pairs to fit and test: two conditions for every model, 448 px for the best one."""
    return [(model, condition) for model in MODELS
            for condition in (*CONDITIONS, *(["448"] if model == best else []))]


def fit() -> None:
    runs = get_default_runs_dir()
    chosen = selected_at_224(read_json(runs / "session_selection.json"))
    best = best_model(chosen)
    configurations = {}
    for model, condition in planned(best):
        config = load_config(f"configs/session/{model}_{condition}.yaml")
        configurations[f"{model}_{condition}"] = [stage5.run_one(config, chosen[model][0], r)[0] for r in ROUNDS]
    tested = [run for names in configurations.values() for run in names]
    if (runs / PLAN).exists() and set(read_json(runs / PLAN)["runs"]) != set(tested):
        raise RuntimeError(f"runs/{PLAN} is frozen with other runs: the selection cannot change after `evaluate`")
    write_json(SELECTION, {"partition": "val", "split": "splits/session_blocked/folds.csv",
                           "code_version": require_clean_code(), "best_448_model": best,
                           "learning_rates": {model: lr for model, (lr, _) in chosen.items()},
                           "configurations": configurations})


def evaluate_selected() -> None:
    runs = get_default_runs_dir()
    selection = read_json(runs / SELECTION)
    best = best_model(selected_at_224(read_json(runs / "session_selection.json")))
    configurations = selection["configurations"]
    if (selection["best_448_model"] != best or set(configurations) != {f"{m}_{c}" for m, c in planned(best)}
            or any(len(names) != len(ROUNDS) for names in configurations.values())):
        raise RuntimeError(f"runs/{SELECTION} does not hold the declared 45 runs; stop and report")
    evaluate_once([run for names in configurations.values() for run in names], PLAN)


ACTIONS = {"fit": fit, "evaluate": evaluate_selected}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=ACTIONS)
    args = parser.parse_args()
    require_intact()
    ACTIONS[args.action]()
    print(f"SESSION CONDITIONS {args.action} DONE", flush=True)


if __name__ == "__main__":
    main()
