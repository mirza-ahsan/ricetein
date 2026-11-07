"""Stage 2 report: docs/reports/audit_report.md, built only from the audit files (every number computed).

The report only reads; each section shows its progress until the human part (labelling, reviewing,
checking) is finished.

    uv run python -m src.audit_report
"""

from __future__ import annotations

import argparse
from typing import Any

import pandas as pd

from src import background_audit, duplicates, label_check
from src.config import REPO_ROOT, load_config, path_for
from src.inventory_report import md_table

COMPARISONS = {"primary": "within primary", "cross": "primary x Paddy Doctor"}


def pct(share: float) -> str:
    return f"{share:.1%}"


def pending(result: dict[str, Any], done_key: str, total: Any) -> str:
    unsettled = f"; {result['unsettled']} disagreements to settle" if "unsettled" in result else ""
    return f"**Pending.** Done so far: {result[done_key] or 'nobody yet'} of {total}{unsettled}.\n"


def background_section(config: dict[str, Any]) -> str:
    settings = config["background"]
    text = ("## 1. Backgrounds (2a)\n\n"
            f"Sample: {settings['per_class']} random photos per class (seed {config['seed']}), frozen in "
            f"`{settings['sample_file']}`. Two team members label every photo alone, in their own shuffled "
            "order, without seeing its class: plain or cluttered, plus several leaves, hand, strong shadow. "
            "Disagreements on the background are settled together; both original labels are kept.\n\n")
    result = background_audit.results(config)
    if not result["complete"]:
        return text + pending(result, "answered", result["sampled"])
    plain = result["plain"]
    rows = pd.DataFrame([{"class": c, "photos": s["n"], "plain": pct(s["share"]),
                          "95% CI": f"{pct(s['low'])}-{pct(s['high'])}"}
                         for c, s in result["plain_by_class"].items()])
    flags = pd.DataFrame([{"flag": f, **{r: pct(v) for r, v in rates.items()}}
                          for f, rates in result["flag_rates"].items()])
    first, second = result["raters"]
    kappa = "n/a" if result["kappa"] is None else f"{result['kappa']:.2f}"
    return text + (
        f"**Plain background: {pct(plain['share'])}** of {plain['n']} photos (95% CI "
        f"{pct(plain['low'])}-{pct(plain['high'])}). Raters {first} and {second} agree on "
        f"{pct(result['agreement'])} of photos (Cohen's kappa {kappa}); {result['n_settled']} "
        f"disagreements were settled together. File: `{settings['audit_file']}`.\n\n" + md_table(rows)
        + "\n\nFlags (share of photos, per rater):\n\n" + md_table(flags) + "\n")


def duplicates_section(config: dict[str, Any]) -> str:
    settings = config["duplicates"]
    check = settings["check"]
    text = ("## 2. Near-duplicates (2b) and overlap with Paddy Doctor (2c)\n\n"
            f"Flag rules (pre-declared methodology): perceptual hashes within {settings['max_hash_bits']} bits "
            f"(any of 8 rotations/flips), or whole-photo E1 class-token cosine > {settings['min_cosine']}. "
            f"Automatic check: SIFT keypoints (photos at {check['long_side']} px), ratio test "
            f"{check['ratio']}, each keypoint used once, RANSAC similarity transform (tolerance "
            f"{check['ransac_px']} px, scale {check['min_scale']}-{check['max_scale']}x, also against the "
            f"mirrored photo); a pair passes with >= {check['min_inliers']} distinct inliers. People review "
            "every "
            f"passed pair plus {settings['rejected_sample']} random rejected pairs per comparison, blind to "
            "class and check result (deviation from \"review every flagged pair\": handbook §24, "
            "2026-11-07).\n\n")
    path = REPO_ROOT / settings["candidates_file"]
    if not path.exists():
        return text + "**Pending.** Run `uv run python -m src.duplicates flag`.\n"
    candidates = pd.read_csv(path)
    queue = pd.read_csv(REPO_ROOT / settings["review_file"])
    rows = [{"comparison": COMPARISONS[name], "flagged": len(group),
             "hash only": int((group["by_hash"] & ~group["by_cosine"]).sum()),
             "cosine only": int((~group["by_hash"] & group["by_cosine"]).sum()),
             "both": int((group["by_hash"] & group["by_cosine"]).sum()),
             "pass check": int(group["passed"].sum()), "queued": int((queue["comparison"] == name).sum())}
            for name, group in candidates.groupby("comparison", sort=False)]
    text += (md_table(pd.DataFrame(rows))
             + f"\n\nFiles: `{settings['candidates_file']}`, `{settings['review_file']}`.\n\n")
    result = duplicates.outcome(config)
    if not result["complete"]:
        return text + pending(result, "answered", result["queued"])
    pairs, groups = result["pairs"], result["groups"]
    reviewed = pairs.groupby(["comparison", "round", "passed"])
    same = reviewed["final"].apply(lambda v: int((v == "same").sum()))
    outcome = pd.DataFrame({"reviewed": reviewed.size(), "confirmed same leaf": same})
    misses = [f"{COMPARISONS[c]}: {int(m['confirmed'])} of {int(m['sampled'])} sampled rejected pairs "
              f"confirmed (miss rate {pct(s['share'])}, 95% CI {pct(s['low'])}-{pct(s['high'])})"
              for c, m in result["misses"].iterrows()
              for s in [background_audit.share(int(m["confirmed"]), int(m["sampled"]))]]
    sizes = groups.drop_duplicates("group_id")["group_size"].value_counts().sort_index()
    first, second = result["reviewers"]
    kappa = "n/a" if result["kappa"] is None else f"{result['kappa']:.2f}"
    return text + (
        f"Reviewers {first} and {second}: Cohen's kappa {kappa}; "
        f"{int(pairs['settled'].sum())} disagreements settled together.\n\n"
        + md_table(outcome.reset_index())
        + "\n\nWhat the automatic check misses: " + "; ".join(misses) + ".\n\n"
        f"**Groups:** {groups['group_id'].nunique()} groups for {len(groups)} photos; group sizes "
        + ", ".join(f"{size}: {n}" for size, n in sizes.items())
        + f". **Label conflicts:** {len(result['conflicts'])} photos in confirmed pairs with different "
        "classes "
        f"(excluded from all experiments; `{settings['conflicts_file']}`). **Overlap with Paddy Doctor:** "
        f"{len(result['overlap'])} confirmed matches (`{settings['overlap_file']}`).\n")


def label_section(config: dict[str, Any]) -> str:
    settings = config["label_check"]
    text = ("## 3. Label sanity check (2d)\n\n"
            f"Each team member looks at their own {settings['per_class']} random photos per class with the "
            "label shown, and says whether it looks right, looks like another class, or is unclear. "
            "No label is changed: the main results always use the original labels.\n\n")
    result = label_check.results(config)
    if not result["complete"]:
        return text + pending(result, "done", f"{result['expected']} each")
    suspects = result["suspects"]
    counts = suspects.groupby(["class", "looks_like"]).size().rename("answers").reset_index()
    return text + (f"{result['checked']} answers on {result['unique_photos']} photos; {len(suspects)} "
                   f"answers on {suspects['photo_id'].nunique()} photos look like another class "
                   f"(`{settings['suspects_file']}`).\n\n" + md_table(counts) + "\n")


def build_report(config: dict[str, Any]) -> str:
    return "\n".join(["# Stage 2 audit report\n",
                      "Built by `src/audit_report.py` from the audit files; do not edit by hand.\n",
                      background_section(config), duplicates_section(config), label_section(config)])


def write_report(config: dict[str, Any]) -> str:
    """Build the report, write docs/reports/audit_report.md and return its text."""
    text = build_report(config)
    target = path_for(config, "reports") / "audit_report.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Write docs/reports/audit_report.md.")
    parser.add_argument("--config", default="configs/audits.yaml")
    write_report(load_config(parser.parse_args().config))
    print("wrote docs/reports/audit_report.md")


if __name__ == "__main__":
    main()
