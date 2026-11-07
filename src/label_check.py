"""Stage 2d: a light label sanity check, without changing any label.

Each team member looks at their own random ``per_class`` photos per class (seeded by their name), with
the dataset's label shown, and answers: K the label looks right, 1-4 it looks like another class, or U
unclear. Photos anyone thinks look like another class go to the suspects list. The main results always
use the original labels; if there are many suspects, one extra analysis without them is added later.

    uv run python -m src.label_check check --checker <name>    # each team member, alone
    uv run python -m src.label_check results                   # writes data/suspect_labels.csv
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from src import review_tool
from src.config import REPO_ROOT, dataset_dir, load_config
from src.inventory import read_inventory

COLUMNS = ["photo_id", "class", "checker", "answer", "checked_at"]


def choices(classes: list[str]) -> dict[str, str]:
    """K: the label looks right; U: unclear; 1-4: the photo looks like that class."""
    return {"k": "right", "u": "unclear", **{str(i): c for i, c in enumerate(classes, start=1)}}


def instructions(classes: list[str]) -> str:
    others = "   ".join(f"{i} {c}" for i, c in enumerate(classes, start=1))
    return f"K label looks right   U unclear   Looks like: {others}   Backspace = undo   Esc = stop"


def my_photos(config: dict[str, Any], checker: str) -> pd.DataFrame:
    """This checker's own random photos (``per_class`` per class), in their own shuffled order."""
    seed = review_tool.person_seed(config["seed"], checker)
    inventory = read_inventory(config["inventory"]["files"]["primary"])
    chosen = review_tool.stratified_sample(inventory, config["label_check"]["per_class"], seed)
    return chosen.sample(frac=1, random_state=np.random.default_rng(seed)).reset_index(drop=True)


def check(config: dict[str, Any], checker: str, ask: Any = review_tool.ask) -> None:
    """Show each of this checker's photos with its label; save every answer at once."""
    def make_row(photo: dict[str, Any], answer: str, flags: set[str]) -> dict[str, Any]:
        return {"photo_id": photo["photo_id"], "class": photo["class"], "checker": checker, "answer": answer,
                "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    root = dataset_dir(config, "primary")
    done = review_tool.collect(
        my_photos(config, checker).to_dict("records"), "photo_id",
        REPO_ROOT / config["label_check"]["checks_dir"] / f"{checker}.csv", COLUMNS,
        show=lambda photo: [review_tool.load_photo(root / photo["photo_id"], 1400)],
        title=lambda photo, n, total: (f"{checker}: photo {n}/{total}   label: {photo['class'].upper()}\n"
                                       f"{instructions(config['classes'])}"),
        choices=choices(config["classes"]), make_row=make_row, ask=ask)
    if done:
        print("All photos checked.")


def results(config: dict[str, Any]) -> dict[str, Any]:
    """Answers per checker and the suspects list, once every team member has finished (reads only)."""
    settings = config["label_check"]
    people = settings["checkers"]
    data = review_tool.people_rows(REPO_ROOT / settings["checks_dir"], COLUMNS, "checker",
                                   review_tool.allowed_people(config), "photo_id", limit=len(people))
    expected = settings["per_class"] * len(config["classes"])
    done = data.groupby("checker")["photo_id"].nunique().to_dict()
    if sorted(done) != sorted(people) or any(n != expected for n in done.values()):
        return {"complete": False, "done": done, "expected": expected}
    other = data["answer"].isin(config["classes"]) & (data["answer"] != data["class"])
    suspects = data[other].rename(columns={"answer": "looks_like"})
    suspects = suspects[["photo_id", "class", "checker", "looks_like"]]
    return {"complete": True, "checked": len(data), "unique_photos": data["photo_id"].nunique(),
            "suspects": suspects.sort_values(["photo_id", "checker"], ignore_index=True)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["check", "results"])
    parser.add_argument("--checker", help="your first name in lower case (for 'check')")
    parser.add_argument("--config", default="configs/audits.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.action == "check":
        allowed = review_tool.allowed_people(config)
        check(config, review_tool.person_name(parser, args.checker, allowed, "--checker"))
    else:
        result = results(config)
        if not result["complete"]:
            print(f"Not complete: {result}")
            return
        result["suspects"].to_csv(REPO_ROOT / config["label_check"]["suspects_file"], index=False)
        print(f"{result['checked']} answers on {result['unique_photos']} photos; "
              f"{len(result['suspects'])} suspect answers -> {config['label_check']['suspects_file']}")


if __name__ == "__main__":
    main()
