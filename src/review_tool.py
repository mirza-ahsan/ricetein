"""A keyboard-only photo viewer for the Stage 2 human judgements (duplicates, backgrounds, labels).

``ask(images, title, choices, toggles)`` shows one or more photos side by side and returns the answer:
- ``choices`` maps a key to an answer (e.g. {"s": "same", "d": "different"});
- ``toggles`` maps a key to an extra yes/no flag (e.g. {"m": "multiple_leaves"}).

Without toggles, pressing a choice key answers at once. With toggles, the choice and flags can be changed
freely and Enter accepts. Esc or closing the window returns ("stop", set()) so the caller can save and
quit; the caller keeps progress, so running again continues where it stopped.
"""

from __future__ import annotations

from typing import Any

STOP = "stop"


def ask(images: list[Any], title: str, choices: dict[str, str],
        toggles: dict[str, str] | None = None) -> tuple[str, set[str]]:
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
            elif key in choices:
                state["choice"] = choices[key]
                if not toggles:
                    finish(choices[key])
                else:
                    redraw()
            elif key in toggles:
                state["flags"] ^= {toggles[key]}
                redraw()
            elif key == "enter" and toggles and state["choice"]:
                finish(state["choice"])

        fig.canvas.mpl_connect("key_press_event", on_key)
        fig.canvas.mpl_connect("close_event", lambda event: finish(STOP))
        redraw()
        plt.show(block=False)
        fig.canvas.start_event_loop(timeout=-1)
        plt.close(fig)
    answer = state["done"] or STOP
    return answer, (state["flags"] if answer != STOP else set())
