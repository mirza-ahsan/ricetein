"""Readiness check before the session-split tuning runs and before their test evaluation.

    uv run --locked python scripts/session_preflight.py

It trains and evaluates nothing and reads no test file. It checks that:
1. the 30 session runs of 2026-11-15 that the searches reuse (20 whole-image fits, 10 probe fits) are
   accepted by ``reusable_run`` with the settings the new code builds;
2. every packed descriptor file the head studies (8 versions) and their evaluation (1 version) need is on
   disk. A missing file is built here, one descriptor kind at a time. The frozen ``BagStore`` fails with
   "class-token and patch-mean bag indices differ" when it has to build one kind while the other is already
   on disk (incidents of 2026-11-16), so the files must exist before it is used.
The packed files live in ``cache/`` (git-ignored); building them changes no code and no result.
"""

from src import train_mil
from src.config import cache_folder, load_config
from src.embed import completed_cache
from src.followup import MODELS, ROUNDS, SIZES, core_config
from src.records import reusable_run
from src.split import read_folds
from src.train_wholeimage import run_name


def require_reusable(name: str, config: dict, checksum: str, checkpoint: str) -> None:
    if not reusable_run(name, config, checksum, (checkpoint, "metrics.json", "predictions.csv")):
        raise SystemExit(f"{name}: not complete, so it would be trained again")


def reused_runs() -> int:
    """The locked-split settings that were refitted on the session split on 2026-11-15, one run per round."""
    for model in MODELS:
        config = load_config(f"configs/session/{model}_224.yaml")
        _, checksum = read_folds(config)
        lr = core_config(f"{model}_224")["lr"]
        for r in ROUNDS:
            require_reusable(run_name(config, lr, r), {**config, "lr": lr, "round": r}, checksum, "model.pt")
    for size in SIZES:
        config = load_config(f"configs/session/probe_e1_{size}.yaml")
        strength = core_config(f"probe_{size}")["l2_strength"]
        config = {**config, "probe": {**config["probe"], "l2_strengths": [strength]}}
        config["cache_spec_sha256"] = completed_cache(config, config["probe"]["cache"])["spec_sha256"]
        _, checksum = read_folds(config)
        for r in ROUNDS:
            require_reusable(f"stage7a_probe_e1_{size}_session_l2{strength:g}_r{r}",
                             {**config, "round": r, "l2_strength": strength}, checksum, "model.pkl")
    return (len(MODELS) + len(SIZES)) * len(ROUNDS)


def packed_files() -> None:
    for stem in ("session_attention", "session_whole224", "session_whole512"):
        config = load_config(f"configs/mil/{stem}.yaml")
        cache = config["mil"]["cache"]
        spec = completed_cache(config, cache)["spec_sha256"]
        folds, _ = read_folds(config)
        photo_ids = sorted(folds.photo_id)
        folder, packed = cache_folder(cache), cache_folder(f"packed/{cache}", write=True)
        packed.mkdir(parents=True, exist_ok=True)
        for versions in (config["mil"]["versions"], 1):
            for key in ("cls", "mean"):
                train_mil.pack(folder, packed, key, photo_ids, versions, spec)
            for descriptor in ("c", "m", "cm"):
                store = train_mil.BagStore(cache, descriptor, photo_ids, versions)
                assert store.bag(photo_ids[0], 0).shape[1] == store.dim
        print(f"{cache}: packed files ready for {config['mil']['versions']} versions and for 1 version")


if __name__ == "__main__":
    print(f"{reused_runs()} earlier session runs will be reused")
    packed_files()
    print("SESSION PREFLIGHT OK")
