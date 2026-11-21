"""Stage 13 cost: size, speed and memory of every main method, and the one tested small model (handbook §24,
2026-11-21).

    uv run python -m src.cost sizes                  # parameters, files, cache bytes (no timing)
    uv run python -m src.cost time --device gpu      # end-to-end seconds per photo, each method in its own process
    uv run python -m src.cost time --device cpu      # (nothing else may run on the machine)
    uv run python -m src.cost select                 # the final small model, chosen on validation by the declared rule
    uv run python -m src.cost plan                   # freeze runs/cost_plan.json and the model's validation run
    uv run python -m src.cost test                   # test it ONCE (runs/cost_final/test_started.json, never deleted)
    uv run python -m src.cost report                 # results/tables/cost_*.csv and docs/reports/cost_report.md

Compression variants and distillation have their own modules (``src.compress``, ``src.distill``). Test scores of the
main methods are read from their existing test runs; the only new test evaluation is the final small model.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import resource
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import torch

from src import compress, tiling
from src import followup_report as fr
from src.compress import CACHES, ENCODER, TILE, VARIANTS, head_runs, read_json, selected_runs
from src.config import REPO_ROOT, cache_folder, load_config, output_root
from src.encoders import describe, load_encoder
from src.inventory_report import md_table
from src.records import RunRecorder, get_default_runs_dir, require_clean_code
from src.split import read_folds, round_column

# method -> how it is run. Keys are those of src/make_tables.py, so the cost chart can name them.
METHODS: dict[str, dict[str, Any]] = {
    "attention": {"kind": "tiles", "head": "attention", "tiles": "all"},
    "mean": {"kind": "tiles", "head": "mean", "tiles": "all"},
    "max": {"kind": "tiles", "head": "max", "tiles": "all"},
    "probe_224": {"kind": "probe", "size": 224},
    "probe_512": {"kind": "probe", "size": 512},
    **{f"{m}_224": {"kind": "cnn", "model": m, "condition": "224"}
       for m in ("inceptionv3", "efficientnetv2s", "convnext_tiny", "swin_tiny")},
    "convnext_tiny_448": {"kind": "cnn", "model": "convnext_tiny", "condition": "448"},
}
# compression variants that are timed, and the two that only run on the CPU
VARIANT_TIMING = {"e1_top16": {"kind": "tiles", "head": "attention", "tiles": "top16"},
                  "e4": {"kind": "tiles", "head": "e4", "tiles": "all"},
                  "e4_top16": {"kind": "tiles", "head": "e4", "tiles": "top16"},
                  "attention_int8": {"kind": "tiles", "head": "attention", "tiles": "all", "int8": True},
                  "final_onnx": {"kind": "onnx"}}
CPU_ONLY = ("attention_int8", "final_onnx")
TIMED = {**METHODS, **VARIANT_TIMING}
# the distilled head has the architecture of the E4 head, so it takes that timing
TIMING_KEY = {"e1_top16": "e1_top16", "e4": "e4", "e4_top16": "e4_top16", "distilled_e4": "e4",
              "distilled_e4_top16": "e4_top16"}


# ---------------------------------------------------------------------------------------------- timing harness
class Laps:
    """Splits one photo's time into named parts. ``sync`` makes the GPU finish before each clock reading."""

    def __init__(self, sync: Callable[[], None], clock: Callable[[], float] = time.perf_counter) -> None:
        self.sync, self.clock, self.parts = sync, clock, {}
        sync()
        self.start = self.last = clock()

    def mark(self, name: str) -> None:
        self.sync()
        now = self.clock()
        self.parts[name], self.last = self.parts.get(name, 0.0) + now - self.last, now

    def total(self) -> float:
        return self.last - self.start


def time_photos(run: Callable[[str, Laps], Any], timed: list[str], warmup: list[str], sync: Callable[[], None],
                clock: Callable[[], float] = time.perf_counter) -> list[dict[str, Any]]:
    """Run every warm-up photo (discarded), then each timed photo once, one at a time."""
    for path in warmup:
        run(path, Laps(sync, clock))
    rows = []
    for path in timed:
        laps = Laps(sync, clock)
        run(path, laps)
        rows.append({"photo": Path(path).name, **laps.parts, "total": laps.total()})
    return rows


def summarise(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Median and quartiles of the total, and the median of each part."""
    frame = pd.DataFrame(rows)
    out = {"n": len(frame), "median": float(frame["total"].median()),
           "q1": float(frame["total"].quantile(0.25)), "q3": float(frame["total"].quantile(0.75))}
    return out | {f"median_{c}": float(frame[c].median()) for c in frame.columns if c not in ("photo", "total")}


def timing_photos(config: dict[str, Any]) -> tuple[list[str], list[str]]:
    """(timed, warm-up) photo ids: a seeded draw from the training photos of the source round. No label is read."""
    t = config["cost"]["timing"]
    folds, _ = read_folds(config)
    train = sorted(folds["photo_id"][folds[round_column(t["source_round"])] == "train"])
    chosen = np.random.default_rng(t["seed"]).choice(train, t["photos"] + t["warmup"], replace=False).tolist()
    return chosen[:t["photos"]], chosen[t["photos"]:]


def machine_state() -> dict[str, Any]:
    """What else runs on the machine: processes using the CPU over two seconds, python processes, GPU memory."""
    me = psutil.Process()
    processes = [p for p in psutil.process_iter(["pid", "name"]) if p.pid != me.pid]
    for p in processes:
        try:
            p.cpu_percent(None)
        except psutil.Error:
            pass
    psutil.cpu_percent(None)
    time.sleep(2)
    busy, python = [], []
    for p in processes:
        try:
            usage = p.cpu_percent(None)
            if usage > 1:
                busy.append({"pid": p.pid, "name": p.info["name"], "cpu_percent": usage})
            if "python" in (p.info["name"] or ""):
                python.append({"pid": p.pid, "cmdline": " ".join(p.cmdline())[:200]})
        except psutil.Error:
            pass
    state: dict[str, Any] = {"total_cpu_percent": psutil.cpu_percent(None), "busy_processes": busy,
                             "python_processes": python, "gpu_mib_used": None}
    try:
        used = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                       text=True, timeout=20)
        state["gpu_mib_used"] = int(used.split()[0])
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return state


def require_quiet(state: dict[str, Any], quiet: dict[str, float], device: str) -> None:
    """Refuse to time on a busy machine (thresholds from configs/cost.yaml)."""
    problems = [f"{p['name']} (pid {p['pid']}) uses {p['cpu_percent']:.0f}% of a core" for p in state["busy_processes"]
                if p["cpu_percent"] > quiet["max_process_cpu_percent"]]
    if state["total_cpu_percent"] > quiet["max_total_cpu_percent"]:
        problems.append(f"total CPU use is {state['total_cpu_percent']:.0f}%")
    if state["gpu_mib_used"] is not None and state["gpu_mib_used"] > quiet["max_gpu_mib"]:
        problems.append(f"{state['gpu_mib_used']} MiB of GPU memory are in use")
    if problems:
        raise RuntimeError("the machine is not quiet: " + "; ".join(problems))


# ---------------------------------------------------------------------------------------------- pipelines
def decision_of(config: dict[str, Any]) -> dict[str, Any]:
    return read_json(REPO_ROOT / config["pixel_scale"]["decision_file"])


def descriptor_of(cls: torch.Tensor, mean: torch.Tensor, descriptor: str) -> torch.Tensor:
    return torch.cat({"c": [cls], "m": [mean], "cm": [cls, mean]}[descriptor], dim=1)


def tile_pipeline(config: dict[str, Any], device: torch.device, spec: dict[str, Any]) -> Callable:
    """Open, rescale to the leaf scale, (pick and) cut tiles, describe in batches, decide with the round-1 head."""
    head = spec["head"]
    code, rule = ENCODER[head], compress.rule_of(config)
    encoder = load_encoder(config, code, torch.device("cpu"))
    encoder = compress.quantise(encoder) if spec.get("int8") else encoder.to(device)
    model, params = compress.load_head(head_runs(head)[0])
    model.to(device)
    decision, batch = decision_of(config), config["cost"]["timing"]["tile_batch"][device.type]
    encoder_spec = config["encoders"][code]

    def run(path: str, laps: Laps) -> None:
        rgb = compress.leaf_scale_photo(path, decision)
        tiles = (compress.cut_selected(rgb, compress.top_tiles(rgb, rule)) if spec["tiles"] == "top16"
                 else tiling.cut(rgb, TILE)[0])
        laps.mark("prepare")
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda" and not spec.get("int8")):
            parts = [describe(encoder, tiles[i:i + batch], encoder_spec, config, device) for i in range(0, len(tiles), batch)]
        cls, mean = torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])
        laps.mark("describe")
        bag = descriptor_of(cls, mean, params["descriptor"])[None].to(device).float()
        with torch.inference_mode():
            torch.softmax(model(bag, torch.ones(1, bag.shape[1], dtype=torch.bool, device=device))[0], 1).cpu()
        laps.mark("decide")
    return run


def probe_pipeline(config: dict[str, Any], device: torch.device, spec: dict[str, Any]) -> Callable:
    """Open, rescale, pad to square at 224 or 512 px, one E1 description, the probe's prediction."""
    import pickle

    from src.photo_dataset import prepare_square

    encoder = load_encoder(config, "E1", device)
    with (get_default_runs_dir() / selected_runs(f"probe_{spec['size']}")[0] / "model.pkl").open("rb") as f:
        model = pickle.load(f)   # our own local checkpoint
    decision = decision_of(config)

    def run(path: str, laps: Laps) -> None:
        square = prepare_square(compress.leaf_scale_photo(path, decision), spec["size"])
        laps.mark("prepare")
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            cls, _ = describe(encoder, square[None], config["encoders"]["E1"], config, device)
        laps.mark("describe")
        model.predict_proba(cls.float().numpy())
        laps.mark("decide")
    return run


def cnn_config(spec: dict[str, Any]) -> dict[str, Any]:
    from src.session_conditions import condition_config

    if spec["condition"] == "448":
        return condition_config(spec["model"], "448")
    return load_config(f"configs/session/{spec['model']}_224.yaml")


def cnn_pipeline(config: dict[str, Any], device: torch.device, spec: dict[str, Any], method: str) -> Callable:
    """Open (reduced-scale decode), pad to square, resize, predict: how the whole-image models were trained."""
    from src import augment, photo_dataset
    from src import train_wholeimage as tw

    own = cnn_config(spec)
    w = own["wholeimage"]
    model = tw.build_model(own, w["size"])
    model.load_state_dict(torch.load(get_default_runs_dir() / selected_runs(method)[0] / "model.pt",
                                     map_location="cpu", weights_only=True))
    model = model.to(device).eval()
    transform, side = augment.evaluation(w["size"]), photo_dataset.cache_size(w["size"])
    mean = torch.tensor(config["normalisation"]["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(config["normalisation"]["std"], device=device).view(1, 3, 1, 1)

    def run(path: str, laps: Laps) -> None:
        image = transform(torch.from_numpy(photo_dataset.prepare(path, side, w["otsu"])).permute(2, 0, 1))
        laps.mark("prepare")
        x = (image[None].to(device).float() / 255 - mean) / std
        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            torch.softmax(model(x).float(), 1).cpu()
        laps.mark("decide")
    return run


def onnx_pipeline(config: dict[str, Any]) -> Callable:
    """The final small model from its two ONNX files on onnxruntime (CPU)."""
    selection = read_json(get_default_runs_dir() / "cost_selection.json")["selected"]
    folder, threads = cache_folder("compress/onnx"), config["cost"]["timing"]["cpu_threads"]
    encoder = compress.onnx_session(folder / f"{selection}_encoder.onnx", threads)
    head = compress.onnx_session(folder / f"{selection}_head.onnx", threads)
    decision, rule, batch = decision_of(config), compress.rule_of(config), config["cost"]["timing"]["tile_batch"]["cpu"]
    top16 = VARIANTS[selection][1] == "top16"

    def run(path: str, laps: Laps) -> None:
        rgb = compress.leaf_scale_photo(path, decision)
        tiles = compress.cut_selected(rgb, compress.top_tiles(rgb, rule)) if top16 else tiling.cut(rgb, TILE)[0]
        laps.mark("prepare")
        pixels = tiles.transpose(0, 3, 1, 2).astype(np.float32) / 255
        described = np.concatenate([encoder.run(None, {"pixels": pixels[i:i + batch]})[0] for i in range(0, len(pixels), batch)])
        laps.mark("describe")
        head.run(None, {"tiles": described[None], "mask": np.ones((1, len(described)), bool)})
        laps.mark("decide")
    return run


def build_run(method: str, config: dict[str, Any], device: torch.device) -> Callable:
    spec = TIMED[method]
    if spec["kind"] == "tiles":
        return tile_pipeline(config, device, spec)
    if spec["kind"] == "probe":
        return probe_pipeline(config, device, spec)
    if spec["kind"] == "cnn":
        return cnn_pipeline(config, device, spec, method)
    return onnx_pipeline(config)


def rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024    # Linux: kilobytes


def time_one(method: str, device_name: str, out: Path) -> None:
    """One method in this fresh process: so the memory peaks belong to it alone."""
    config = load_config("configs/cost.yaml")
    device = torch.device("cuda" if device_name == "gpu" else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(config["cost"]["timing"]["cpu_threads"])
    else:
        torch.cuda.reset_peak_memory_stats()
    baseline = rss_mib()
    run = build_run(method, config, device)
    timed, warmup = timing_photos(config)
    root = _photo_root(config)
    for photo in [*warmup, *timed]:           # photo files into the OS cache, untimed: only decoding is timed
        (root / photo).read_bytes()
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    rows = time_photos(run, [str(root / p) for p in timed], [str(root / p) for p in warmup], sync)
    out.write_text(json.dumps({"method": method, "device": device_name, "rows": rows, "summary": summarise(rows),
                               "peak_gpu_mib": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None,
                               "peak_rss_mib": rss_mib(), "rss_before_model_mib": baseline}, indent=2))


def _photo_root(config: dict[str, Any]) -> Path:
    from src.config import dataset_dir

    return dataset_dir(config, "primary")


def time_all(config: dict[str, Any], device_name: str) -> None:
    require_clean_code()
    start = machine_state()
    require_quiet(start, config["cost"]["quiet"], device_name)
    _, checksum = read_folds(config)
    recorder = RunRecorder(f"cost_time_{device_name}", config=config, split_checksum=checksum, overwrite=True,
                           data_version={"primary": config["datasets"]["primary"]["version"]})
    methods = [m for m in TIMED if not (device_name == "gpu" and m in CPU_ONLY)]
    if "final_onnx" in methods and not (get_default_runs_dir() / "cost_selection.json").exists():
        methods.remove("final_onnx")          # the final model is not chosen yet: time it after `select` and `export`
    results = {}
    for method in methods:
        out = recorder.run_dir / f"{method}.json"
        subprocess.run([sys.executable, "-m", "src.cost", "time-one", "--method", method, "--device", device_name,
                        "--out", str(out)], check=True)
        results[method] = read_json(out)
        print(f"{method}: median {results[method]['summary']['median']:.3f} s per photo ({device_name})", flush=True)
    pd.DataFrame([{"method": m, "device": device_name, **row} for m, res in results.items() for row in res["rows"]]
                 ).to_csv(recorder.run_dir / "timing.csv", index=False)
    recorder.save_metrics({"partition": "none", "device": device_name,
                           "median_seconds_per_photo": {m: r["summary"]["median"] for m, r in results.items()},
                           "summary": {m: r["summary"] for m, r in results.items()},
                           "peak_gpu_mib": {m: r["peak_gpu_mib"] for m, r in results.items()},
                           "peak_rss_mib": {m: r["peak_rss_mib"] for m, r in results.items()},
                           "machine_state_start": start, "machine_state_end": machine_state()})
    recorder.finish()


# ---------------------------------------------------------------------------------------------- sizes
def dir_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def encoder_file_bytes(config: dict[str, Any], code: str) -> int:
    from huggingface_hub import snapshot_download

    spec = config["encoders"][code]
    folder = Path(snapshot_download(spec["name"], revision=spec["revision"], local_files_only=True))
    return sum(f.stat().st_size for f in folder.glob("*.safetensors"))


def count(parameters) -> int:
    return int(sum(p.numel() for p in parameters))


def size_row(method: str, config: dict[str, Any]) -> dict[str, Any]:
    """Parameters (encoder, decision-maker), bytes on disk and cache bytes of one method."""
    runs = get_default_runs_dir()
    spec = TIMED.get(method) or {"kind": "tiles", "head": VARIANTS[method][0]}    # a candidate name, e.g. distilled_e4
    if spec["kind"] == "tiles":
        code = ENCODER[spec["head"]]
        encoder = load_encoder(config, code, torch.device("cpu"))
        state = torch.load(runs / head_runs(spec["head"])[0] / "model.pt", map_location="cpu", weights_only=True)
        if spec.get("int8"):
            buffer = io.BytesIO()
            torch.save(compress.quantise(encoder).state_dict(), buffer)
            encoder_bytes = buffer.getbuffer().nbytes
        else:
            encoder_bytes = encoder_file_bytes(config, code)
        return {"method": method, "params_encoder": count(encoder.parameters()), "params_decider": count(state.values()),
                "disk_bytes": encoder_bytes + (runs / head_runs(spec["head"])[0] / "model.pt").stat().st_size,
                "cache_bytes": dir_bytes(cache_folder(CACHES[code]))}
    if spec["kind"] == "probe":
        import pickle

        encoder = load_encoder(config, "E1", torch.device("cpu"))
        file = runs / selected_runs(method)[0] / "model.pkl"
        with file.open("rb") as f:
            model = pickle.load(f)   # our own local checkpoint
        scaler, logistic = model.steps[0][1], model.steps[1][1]
        stored = [scaler.mean_, scaler.scale_, logistic.coef_, logistic.intercept_]
        return {"method": method, "params_encoder": count(encoder.parameters()), "params_decider": int(sum(a.size for a in stored)),
                "disk_bytes": encoder_file_bytes(config, "E1") + file.stat().st_size,
                "cache_bytes": dir_bytes(cache_folder("e1_wholephoto"))}
    if spec["kind"] == "cnn":
        from src import photo_dataset
        from src import train_wholeimage as tw

        own = cnn_config(spec)
        w = own["wholeimage"]
        model = tw.build_model(own, w["size"])
        decider = count(model.get_classifier().parameters())
        photos = output_root() / w["cache_dir"] / f"photos_{photo_dataset.cache_size(w['size'])}{'_otsu' if w['otsu'] else ''}.u8"
        return {"method": method, "params_encoder": count(model.parameters()) - decider, "params_decider": decider,
                "disk_bytes": (runs / selected_runs(method)[0] / "model.pt").stat().st_size, "cache_bytes": photos.stat().st_size}
    exported = read_json(runs / "cost_export" / "metrics.json")
    return {**size_row(exported["candidate"], config), "method": method, "disk_bytes": sum(exported["bytes"].values())}


def sizes(config: dict[str, Any]) -> None:
    require_clean_code()
    names = [*METHODS, "e1_top16", "e4", "e4_top16", "distilled_e4", "attention_int8"]
    if (get_default_runs_dir() / "cost_export" / "metrics.json").exists():
        names.append("final_onnx")
    rows = [size_row(m, config) for m in names]
    by = {r["method"]: r for r in rows}
    if by["distilled_e4"]["params_decider"] != by["e4"]["params_decider"]:
        raise RuntimeError("the distilled head is not the architecture of the E4 head: it cannot take its timing")
    recorder = RunRecorder("cost_sizes", config=config, overwrite=True,
                           data_version={"primary": config["datasets"]["primary"]["version"]})
    pd.DataFrame(rows).to_csv(recorder.run_dir / "sizes.csv", index=False)
    recorder.save_metrics({"partition": "none", "rows": rows})
    recorder.finish()


# ---------------------------------------------------------------------------------------------- the final model
def choose_final(val_f1: dict[str, float], cpu_seconds: dict[str, float], reference_f1: float, reference_seconds: float,
                 margin_points: float, order: list[str]) -> tuple[str, str]:
    """The declared rule. Eligible: validation macro-F1 within ``margin_points`` of the reference; the one with the
    lowest CPU time wins. If none is eligible: the best validation macro-F1 among those faster than the reference.
    Ties go to the higher validation macro-F1, then to ``order``."""
    def best(names: list[str], key: Callable[[str], tuple]) -> str:
        return min(names, key=lambda n: (*key(n), -val_f1[n], order.index(n)))

    eligible = [n for n in order if val_f1[n] >= reference_f1 - margin_points / 100]
    if eligible:
        return best(eligible, lambda n: (cpu_seconds[n],)), "fastest on the CPU among those within the margin"
    faster = [n for n in order if cpu_seconds[n] < reference_seconds]
    if not faster:
        raise RuntimeError("no candidate is faster than E1 + attention on the CPU")
    return best(faster, lambda n: (-val_f1[n],)), "none within the margin: best validation macro-F1 among the faster ones"


def select(config: dict[str, Any]) -> None:
    version = require_clean_code()
    root = get_default_runs_dir()
    variants = read_json(root / "cost_fewer_tiles" / "metrics.json")["variants"]
    cpu = read_json(root / "cost_time_cpu" / "metrics.json")["median_seconds_per_photo"]
    val = {n: variants[n]["val_macro_f1_mean"] for n in compress.CANDIDATES}
    seconds = {n: cpu[TIMING_KEY[n]] for n in compress.CANDIDATES}
    chosen, why = choose_final(val, seconds, variants["attention"]["val_macro_f1_mean"], cpu["attention"],
                               config["compress"]["final_margin_points"], compress.CANDIDATES)
    (root / "cost_selection.json").write_text(json.dumps(
        {"partition": "val", "selected": chosen, "reason": why, "code_version": version,
         "reference": {"val_macro_f1": variants["attention"]["val_macro_f1_mean"], "cpu_seconds": cpu["attention"]},
         "candidates": {n: {"val_macro_f1": val[n], "cpu_seconds": seconds[n]} for n in compress.CANDIDATES}}, indent=2) + "\n")
    print(f"final small model: {chosen} ({why})", flush=True)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan_entries(selected: str) -> dict[str, dict[str, str]]:
    root = get_default_runs_dir()
    return {run: {"record_sha256": digest(root / run / "record.json"), "checkpoint_sha256": digest(root / run / "model.pt")}
            for run in head_runs(VARIANTS[selected][0])}


def plan(config: dict[str, Any]) -> None:
    """Freeze the choice (validation only) with the hashes of the five round-models, and make the final run's
    validation record. Safe to repeat: the plan cannot change, and the run folder is made once."""
    version = require_clean_code()
    root = get_default_runs_dir()
    selection = read_json(root / "cost_selection.json")
    _, checksum = read_folds(config)
    frozen = {"selection_partition": "val", "code_version": version, "selected": selection["selected"],
              "selection_sha256": digest(root / "cost_selection.json"), "rule": compress.rule_of(config),
              "split_checksum": checksum, "runs": plan_entries(selection["selected"])}
    path = root / "cost_plan.json"
    if path.exists():
        if {k: v for k, v in read_json(path).items() if k != "code_version"} != {k: v for k, v in frozen.items() if k != "code_version"}:
            raise ValueError("the frozen test plan cannot be changed")
    else:
        with path.open("x") as out:
            json.dump(frozen, out, indent=2)
    if (root / "cost_final" / "record.json").exists():
        return
    recorder = RunRecorder("cost_final", config={**config, "selected": selection["selected"]}, split_checksum=checksum,
                           overwrite=not (root / "cost_final" / "test_started.json").exists(),
                           data_version={"primary": config["datasets"]["primary"]["version"]})
    recorder.save_predictions(pd.read_csv(root / "cost_fewer_tiles" / f"predictions_{selection['selected']}.csv"))
    recorder.save_metrics({"partition": "val", **{k: v for k, v in selection.items() if k != "partition"}})
    recorder.finish()


def evaluate_final(config: dict[str, Any]) -> None:
    """The one test evaluation of Stage 13."""
    version = require_clean_code()
    root = get_default_runs_dir()
    frozen = read_json(root / "cost_plan.json")
    if frozen["selection_sha256"] != digest(root / "cost_selection.json") or frozen["runs"] != plan_entries(frozen["selected"]):
        raise ValueError("the selection or a round-model changed after the plan was frozen")
    folder = root / "cost_final"
    try:
        marker = (folder / "test_started.json").open("x")
    except FileExistsError:
        raise RuntimeError("the final small model was already tested or its test was started; no retry") from None
    with marker:
        json.dump({"evaluation_code": version, "plan_sha256": digest(root / "cost_plan.json")}, marker)
    folds, _ = read_folds(config)
    rows = compress.role_rows(folds, "test")
    photos = sorted({folds["photo_id"].iloc[r] for rs in rows.values() for r in rs})
    name = frozen["selected"]
    selection = compress.tile_selection(config, photos) if VARIANTS[name][1] == "top16" else None
    result = compress.variant_scores(config, [name], folds, "test", selection, rows)[name]
    table = result["predictions"]
    table.to_csv(folder / "test_predictions.csv", index=False)
    from src import metrics

    scores = metrics.scores(table["true_class"], table["pred_class"], config["classes"])
    (folder / "test_metrics.json").write_text(json.dumps(
        {"partition": "test", **scores, "macro_f1_rounds": result["scores"], "macro_f1_mean": float(np.mean(result["scores"])),
         "tiles_used": result["tiles_used"], "code_version": version}, indent=2))


# ---------------------------------------------------------------------------------------------- report
def method_rows(config: dict[str, Any]) -> pd.DataFrame:
    """The cost table: every main method (and the committee, if Stage 11 gave one), test macro-F1 from its test runs."""
    root = get_default_runs_dir()
    sizes_ = {r["method"]: r for r in read_json(root / "cost_sizes" / "metrics.json")["rows"]}
    timing = {d: read_json(root / f"cost_time_{d}" / "metrics.json") for d in ("gpu", "cpu")}
    rows = []
    for method in METHODS:
        s = sizes_[method]
        test = [read_json(root / run / "test_metrics.json")["macro_f1"] for run in selected_runs(method)]
        rows.append({"method": method, "params_encoder_m": s["params_encoder"] / 1e6, "params_decider_m": s["params_decider"] / 1e6,
                     "disk_mb": s["disk_bytes"] / 2**20, "cache_gb": s["cache_bytes"] / 2**30,
                     "seconds_per_photo": timing["gpu"]["median_seconds_per_photo"][method],
                     "seconds_per_photo_cpu": timing["cpu"]["median_seconds_per_photo"][method],
                     "peak_gpu_mb": timing["gpu"]["peak_gpu_mib"][method], "peak_cpu_rss_mb": timing["cpu"]["peak_rss_mib"][method],
                     "macro_f1": float(np.mean(test)), "partition": "test"})
    members = config["cost"]["committee"]["members"]
    if members:
        parts = [r for r in rows if r["method"] in members]
        rows.append({"method": "committee", **{k: sum(p[k] for p in parts) for k in
                     ("params_encoder_m", "params_decider_m", "disk_mb", "cache_gb", "seconds_per_photo", "seconds_per_photo_cpu")},
                     "peak_gpu_mb": max(p["peak_gpu_mb"] for p in parts), "peak_cpu_rss_mb": max(p["peak_cpu_rss_mb"] for p in parts),
                     "macro_f1": config["cost"]["committee"]["test_macro_f1"], "partition": "test"})
    final = read_json(root / "cost_final" / "test_metrics.json")
    name = read_json(root / "cost_plan.json")["selected"]
    key = TIMING_KEY[name]
    s = sizes_.get(name) or sizes_["distilled_e4"]       # both distilled candidates have the one head's size
    rows.append({"method": f"final_small_model ({name})", "params_encoder_m": s["params_encoder"] / 1e6,
                 "params_decider_m": s["params_decider"] / 1e6, "disk_mb": s["disk_bytes"] / 2**20,
                 "cache_gb": s["cache_bytes"] / 2**30, "seconds_per_photo": timing["gpu"]["median_seconds_per_photo"][key],
                 "seconds_per_photo_cpu": timing["cpu"]["median_seconds_per_photo"][key],
                 "peak_gpu_mb": timing["gpu"]["peak_gpu_mib"][key], "peak_cpu_rss_mb": timing["cpu"]["peak_rss_mib"][key],
                 "macro_f1": final["macro_f1_mean"], "partition": "test"})
    return pd.DataFrame(rows)


def compression_rows(config: dict[str, Any]) -> pd.DataFrame:
    """Each variant: validation macro-F1 against seconds per photo (GPU and CPU)."""
    root = get_default_runs_dir()
    variants = read_json(root / "cost_fewer_tiles" / "metrics.json")["variants"]
    timing = {d: read_json(root / f"cost_time_{d}" / "metrics.json")["median_seconds_per_photo"] for d in ("gpu", "cpu")}
    reference = variants["attention"]["val_macro_f1_mean"]
    rows = []
    for name, v in variants.items():
        key = "attention" if name == "attention" else TIMING_KEY.get(name)
        rows.append({"variant": name, "what": {"attention": "E1 + attention, all tiles (the reference)"}.get(name, name.replace("_", " ")),
                     "val_macro_f1": v["val_macro_f1_mean"], "val_macro_f1_sd": v["val_macro_f1_sd"],
                     "difference_points": 100 * (v["val_macro_f1_mean"] - reference), "tiles_used": v["tiles_used"],
                     "seconds_per_photo": timing["gpu"].get(key), "seconds_per_photo_cpu": timing["cpu"].get(key),
                     "note": "control (not a candidate)" if "even" in name else "", "partition": "val"})
    int8 = read_json(root / "cost_int8" / "metrics.json")
    rows.append({"variant": "attention_int8", "what": f"E1 with 8-bit linear layers, on a subset of {int8['photos']} validation photos",
                 "val_macro_f1": int8["int8_macro_f1_mean"], "val_macro_f1_sd": float(np.std(int8["int8_macro_f1_rounds"], ddof=1)),
                 "difference_points": 100 * (int8["int8_macro_f1_mean"] - int8["float_macro_f1_mean"]), "tiles_used": None,
                 "seconds_per_photo": None, "seconds_per_photo_cpu": timing["cpu"].get("attention_int8"),
                 "note": (f"difference is against the float descriptions of the same subset ({100 * int8['float_macro_f1_mean']:.1f}%); "
                          f"mean cosine of float and 8-bit tile descriptions {int8['description_cosine']:.2f}"),
                 "partition": "val"})
    if "final_onnx" in timing["cpu"]:
        chosen = read_json(root / "cost_plan.json")["selected"]
        rows.append({"variant": "final_onnx", "what": f"{chosen} exported to ONNX (onnxruntime, CPU)", "val_macro_f1": variants[chosen]["val_macro_f1_mean"],
                     "val_macro_f1_sd": variants[chosen]["val_macro_f1_sd"], "difference_points": 100 * (variants[chosen]["val_macro_f1_mean"] - reference),
                     "tiles_used": variants[chosen]["tiles_used"], "seconds_per_photo": None,
                     "seconds_per_photo_cpu": timing["cpu"]["final_onnx"], "note": "same model, other runtime", "partition": "val"})
    return pd.DataFrame(rows)


def report(config: dict[str, Any]) -> None:
    root = get_default_runs_dir()
    cost, compression = method_rows(config), compression_rows(config)
    tables = REPO_ROOT / "results/tables"
    tables.mkdir(parents=True, exist_ok=True)
    cost.to_csv(tables / "cost_results.csv", index=False)
    compression.to_csv(tables / "cost_compression.csv", index=False)
    final_name = read_json(root / "cost_plan.json")["selected"]
    attention = fr.read_predictions(selected_runs("attention"))
    final = pd.concat([pd.read_csv(root / "cost_final" / f) for f in ("predictions.csv", "test_predictions.csv")], ignore_index=True)
    paired = fr.paired_rows(attention, {final_name: final}, fr.block_of(), config)
    shown = pd.DataFrame({"method": cost["method"].map(lambda m: m.replace("_", " ")),
                          "encoder params (M)": cost["params_encoder_m"].map("{:.1f}".format),
                          "decision-maker params (M)": cost["params_decider_m"].map("{:.2f}".format),
                          "files (MB)": cost["disk_mb"].map("{:.0f}".format), "training cache (GB)": cost["cache_gb"].map("{:.1f}".format),
                          "GPU s/photo": cost["seconds_per_photo"].map("{:.3f}".format),
                          "CPU s/photo": cost["seconds_per_photo_cpu"].map("{:.2f}".format),
                          "peak GPU (MB)": cost["peak_gpu_mb"].map("{:.0f}".format), "peak CPU RAM (MB)": cost["peak_cpu_rss_mb"].map("{:.0f}".format),
                          "test macro-F1 (%)": cost["macro_f1"].map(lambda x: "–" if pd.isna(x) else f"{100 * x:.1f}")})
    fmt = lambda x, f: "–" if pd.isna(x) else f.format(x)   # noqa: E731
    shown_c = pd.DataFrame({"variant": compression["variant"], "what": compression["what"],
                            "validation macro-F1 (%)": [f"{100 * m:.1f} ± {100 * s:.1f}" for m, s in zip(compression["val_macro_f1"], compression["val_macro_f1_sd"])],
                            "difference (points)": compression["difference_points"].map("{:+.1f}".format),
                            "tiles per photo": compression["tiles_used"].map(lambda x: fmt(x, "{:.0f}")),
                            "GPU s/photo": compression["seconds_per_photo"].map(lambda x: fmt(x, "{:.3f}")),
                            "CPU s/photo": compression["seconds_per_photo_cpu"].map(lambda x: fmt(x, "{:.2f}")),
                            "note": compression["note"]})
    p = paired.iloc[0]
    text = f"""# Stage 13: cost and compression

Built by `src/cost.py` from `runs/`; do not edit by hand. Declaration: handbook §24, 2026-11-21.

**Hardware dependence.** Every time and every memory figure comes from one machine, `halo` (RTX 4060 8 GB, Ryzen 7 7700,
8 threads for torch and onnxruntime), and depends on it. Times are medians over 100 training photos of round 1, one photo
at a time from the file on disk (decoding included, cold-disk reads not), after 5 discarded warm-up photos. GPU: fp16
autocast, 128 tiles per pass; CPU: fp32, 16 tiles per pass. Each method ran in its own process.

## 1. Cost of every main method

Accuracy is **test** macro-F1 (mean over 5 rounds) of the method's existing test runs; times, sizes and memory are measured here.
The tile method describes about 79 tiles per photo, so it is **slower per photo than a whole-image model**. The training
cache is the description cache (all 8 versions) or the resized-photo cache.

{md_table(shown)}

## 2. Compression variants (validation macro-F1)

Validation photos only (session split, mean ± SD over 5 rounds); no variant was tested except the final small model.
The round-models are applied as they are (nothing retrained, nothing tuned), except the distilled head.

{md_table(shown_c)}

## 3. The final small model: {final_name}

Chosen on validation by the declared rule (`runs/cost_selection.json`), frozen in `runs/cost_plan.json`, tested once
(`runs/cost_final/test_started.json`). Test, session split. Attention is A and the small model is B, so a positive
difference means attention is better.

{md_table(pd.DataFrame([{"metric": r.metric, "difference A - B (points)": f"{100 * r.difference:+.1f}",
                         "photo range": fr.interval_text(r.photo_low, r.photo_high), "block range": fr.interval_text(r.block_low, r.block_high),
                         "rounds won by attention": f"{r.round_wins}/{r.rounds}", "verdict (photo range)": r.verdict_photo,
                         "verdict (block range)": r.verdict_block} for r in paired.itertuples()]))}

**Limits.** One machine; one seed per round; the 8-bit row rests on a subset of validation photos; the distillation teacher
answers on its own training photos, so its answers are close to the labels; the tile rule is a simple colour heuristic.
"""
    (REPO_ROOT / "docs/reports/cost_report.md").write_text(text)
    print(f"report: {len(cost)} methods, {len(compression)} variants; first row of the paired test: {p.verdict_photo}")


ACTIONS = {"sizes": sizes, "select": select, "plan": plan, "test": evaluate_final, "report": report}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=[*ACTIONS, "time", "time-one"])
    parser.add_argument("--device", choices=["gpu", "cpu"])
    parser.add_argument("--method", choices=list(TIMED))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.action == "time-one":
        time_one(args.method, args.device, args.out)
        return
    config = load_config("configs/cost.yaml")
    if args.action == "time":
        if not args.device:
            parser.error("time needs --device gpu or cpu")
        time_all(config, args.device)
    else:
        ACTIONS[args.action](config)
    print(f"COST {args.action} DONE", flush=True)


if __name__ == "__main__":
    main()
