"""Shared pieces for the Stage 2 human judgements (duplicate pairs, backgrounds, label checks).

- ``ask`` shows one or more photos side by side and returns the key-press answer.
- ``collect`` runs a whole session: each item not yet answered, in the person's own order, saving the
  file after every answer (resume by running again; Backspace undoes the last answer).
- ``people_rows`` reads everyone's answer files and refuses unknown people or a third person;
  ``refuse_newcomer`` does that check before a session starts.
- ``two_person`` turns two people's answers plus the settled ones into final answers and Cohen's kappa.
- ``load_photo``, ``stratified_sample`` and ``person_name`` are the small helpers every tool needs.
"""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.metrics import cohen_kappa_score

STOP, UNDO, SETTLED = "stop", "undo", "settled"
WARM_UP_S = 0.3  # keys pressed this soon after a photo appears are ignored (a held key cannot answer unseen)


def ask(images: list[Any], title: str, choices: dict[str, str],
        toggles: dict[str, str] | None = None) -> tuple[str, set[str]]:
    """Answer for one item: a ``choices`` value, ``UNDO`` (Backspace) or ``STOP`` (Esc / closing the window).

    Without ``toggles`` a choice key answers at once. With toggles (yes/no flags), the choice and flags can be
    changed freely and Enter accepts.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    if matplotlib.get_backend().lower() in {"agg", "pdf", "svg", "ps", "pgf", "cairo", "template"}:
        raise RuntimeError("reviewing needs a screen and an interactive matplotlib backend (e.g. TkAgg)")
    toggles = toggles or {}
    state: dict[str, Any] = {"choice": None, "flags": set(), "done": None}
    settings = {k: [] for k in plt.rcParams if k.startswith("keymap.")}  # only our keys do anything
    with plt.rc_context(settings):
        fig, axes = plt.subplots(1, len(images), figsize=(7 * len(images), 7.5), squeeze=False)
        fig.subplots_adjust(0.01, 0.01, 0.99, 0.88, wspace=0.03)
        for ax, image in zip(axes[0], images):
            ax.imshow(image)
            ax.set_axis_off()

        def redraw() -> None:
            status = ""
            if toggles:
                flags = ", ".join(sorted(state["flags"])) or "none"
                status = f"\nchosen: {state['choice'] or '-'}   flags: {flags}   (Enter = accept)"
            fig.suptitle(title + status, fontsize=10)
            fig.canvas.draw_idle()

        def finish(answer: str) -> None:
            if state["done"] is None:  # the first answer wins: closing the window afterwards must not undo it
                state["done"] = answer
            fig.canvas.stop_event_loop()

        def on_key(event: Any) -> None:
            key = (event.key or "").lower()
            if key == "escape":
                finish(STOP)
            elif time.monotonic() - opened < WARM_UP_S:
                return
            elif key == "backspace":
                finish(UNDO)
            elif key in choices:
                state["choice"] = choices[key]
                if toggles:
                    redraw()
                else:
                    finish(choices[key])
            elif key in toggles:
                state["flags"] ^= {toggles[key]}
                redraw()
            elif key == "enter" and toggles and state["choice"]:
                finish(state["choice"])

        fig.canvas.mpl_connect("key_press_event", on_key)
        fig.canvas.mpl_connect("close_event", lambda event: finish(STOP))
        redraw()
        plt.show(block=False)
        opened = time.monotonic()
        fig.canvas.start_event_loop(timeout=-1)
        plt.close(fig)
    answer = state["done"] or STOP
    return answer, (state["flags"] if answer not in (STOP, UNDO) else set())


def person_seed(seed: int, name: str) -> list[int]:
    """A random seed that is fixed for each person and differs between people."""
    return [seed, int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "big")]


def person_order(seed: int, name: str, n: int) -> np.ndarray:
    """A shuffled order of n items that is fixed for each person and differs between people."""
    return np.random.default_rng(person_seed(seed, name)).permutation(n)


def read_rows(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=columns)
    return pd.read_csv(path, dtype=str)


def collect(items: list[dict[str, Any]], key: str, target: Path, columns: list[str],
            show: Callable[[dict[str, Any]], list[Any]], title: Callable[[dict[str, Any], int, int], str],
            choices: dict[str, str], make_row: Callable[[dict[str, Any], str, set[str]], dict[str, Any]],
            toggles: dict[str, str] | None = None, ask: Callable[..., tuple[str, set[str]]] = ask) -> bool:
    """Ask about every item whose ``key`` is not yet in ``target``, in the given order; save after every
    answer (whole file, swapped in atomically). Backspace removes the last saved answer and asks again.
    True when every item is answered."""
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = read_rows(target, columns)
    while True:
        done = set(rows[key])
        todo = [item for item in items if item[key] not in done]
        if not todo:
            return True
        item = todo[0]
        answer, flags = ask(show(item), title(item, len(items) - len(todo) + 1, len(items)), choices, toggles)
        if answer == STOP:
            print("Stopped; progress saved.")
            return False
        if answer == UNDO:
            rows = rows.iloc[:-1]
        else:
            rows = pd.concat([rows, pd.DataFrame([make_row(item, answer, flags)], columns=columns)],
                             ignore_index=True)
        rows.to_csv(target.with_suffix(".tmp"), index=False)
        os.replace(target.with_suffix(".tmp"), target)


def people_rows(folder: Path, columns: list[str], person: str, allowed: list[str], key: str,
                limit: int = 2) -> pd.DataFrame:
    """Every person's answers (the settled file left out); refuses unknown people, more than ``limit``
    people, or an item answered twice by one person."""
    frames = [read_rows(f, columns) for f in sorted(folder.glob("*.csv")) if f.stem != SETTLED]
    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    names = sorted(data[person].unique())
    if any(n not in allowed for n in names) or len(names) > limit:
        raise ValueError(f"{person}s {names}: expected at most {limit} of {allowed}")
    if data.duplicated([key, person]).any():
        raise ValueError(f"an item was answered twice by the same {person}")
    return data


def refuse_newcomer(folder: Path, name: str, limit: int = 2) -> None:
    """Stop a person before they start if ``limit`` other people already have answer files."""
    others = [f.stem for f in folder.glob("*.csv") if f.stem not in (SETTLED, name)]
    if len(others) >= limit:
        raise SystemExit(f"{sorted(others)} are already doing this task; it needs exactly {limit} people")


def both_done(wide: pd.DataFrame) -> bool:
    """Exactly two people have answered every item (one column each, no gaps)."""
    return wide.shape[1] == 2 and not wide.isna().any().any()


def disagreements(wide: pd.DataFrame) -> list[str]:
    return wide.index[wide.iloc[:, 0] != wide.iloc[:, 1]].tolist()


def two_person(wide: pd.DataFrame, settled: pd.Series, answered: dict[str, int]) -> dict[str, Any]:
    """Final answers for a two-column table of answers (items x people): the agreed answer, or the settled
    one where they disagreed. Returns {"complete": False, ...progress} until both have finished and every
    disagreement is settled; then final, agree and Cohen's kappa (None when undefined: both people gave
    one same answer to everything)."""
    if not both_done(wide):
        return {"complete": False, "answered": answered}
    first, second = wide.columns
    agree = wide[first] == wide[second]
    final = wide[first].where(agree, settled.reindex(wide.index))
    if final.isna().any():
        return {"complete": False, "answered": answered, "unsettled": int(final.isna().sum())}
    kappa = float(cohen_kappa_score(wide[first], wide[second]))
    return {"complete": True, "final": final, "agree": agree, "kappa": None if np.isnan(kappa) else kappa,
            "people": [first, second]}


def settled_answers(folder: Path, columns: list[str], key: str, value: str) -> pd.Series:
    return read_rows(folder / f"{SETTLED}.csv", columns).set_index(key)[value]


def person_name(parser: Any, value: str | None, allowed: list[str], flag: str) -> str:
    """The lower-case name given on the command line; stops with a usage error if it is not allowed."""
    name = (value or "").strip().lower()
    if name not in allowed:
        parser.error(f"{flag} must be one of {allowed}")
    return name


def load_photo(path: str | Path, side: int, mode: str = "RGB") -> Image.Image:
    """A photo decoded at reduced scale (JPEG draft), upright, in mode, shrunk to fit a side x side box."""
    with Image.open(path) as image:
        image.draft(mode, (side, side))
        image = ImageOps.exif_transpose(image).convert(mode)
    image.thumbnail((side, side), Image.Resampling.BICUBIC)
    return image


def stratified_sample(inventory: pd.DataFrame, per_class: int, seed: Any) -> pd.DataFrame:
    """``per_class`` random photos from each class (seeded, independent of row order), sorted by photo_id."""
    rng = np.random.default_rng(seed)
    opened = inventory[inventory["opened_ok"]].sort_values("photo_id", ignore_index=True)
    picks = [group.iloc[np.sort(rng.choice(len(group), per_class, replace=False))]
             for _, group in opened.groupby("class", sort=True)]
    return pd.concat(picks)[["photo_id", "class"]].sort_values("photo_id", ignore_index=True)
