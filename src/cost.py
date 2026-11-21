"""Stage 13 cost: size, speed and memory of every main method, and the one tested small model (declared
2026-11-21).

    uv run python -m src.cost sizes                  # parameters, files, cache bytes (no timing)
    uv run python -m src.cost time --device gpu      # end-to-end seconds per photo, each method in its own process
    uv run python -m src.cost time --device cpu      # (nothing else may run on the machine)
    uv run python -m src.cost time --device cpu --methods final_onnx   # after `select` and `compress export`
    uv run python -m src.cost select                 # the final small model, chosen on validation by the declared rule
    uv run python -m src.cost plan                   # freeze runs/cost_plan.json and the model's validation run
    uv run python -m src.cost test                   # test it ONCE (runs/cost_final/test_started.json, never deleted)
    uv run python -m src.cost report                 # results/tables/cost_*.csv and docs/reports/cost_report.md

Compression variants and distillation have their own modules (``src.compress``, ``src.distill``). Test scores of the
main methods are read from their existing test runs; the only new test evaluation is the final small model.
"""

from __future__ import annotations

import argparse
import io
import json
import resource
import shutil
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

from src import compress, metrics
from src.compress import CACHES, ENCODER, PARTS, VARIANTS, head_runs, selected_runs
from src.config import REPO_ROOT, cache_folder, dataset_dir, load_config, output_root
from src.encoders import describe, load_encoder
from src.followup import read_json
from src.records import RunRecorder, get_default_runs_dir, require_clean_code
from src.split import read_folds, round_column, sha256

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
TIMING_KEY = {"distilled_e4": "e4", "distilled_e4_top16": "e4_top16"}


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


def machine_state(full: bool = False) -> dict[str, Any]:
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
                             "python_processes": python, "gpu_mib_used": None, "nvidia_smi": None}
    try:
        used = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                       text=True, timeout=20)
        state["gpu_mib_used"] = int(used.split()[0])
        if full:
            state["nvidia_smi"] = subprocess.check_output(["nvidia-smi"], text=True, timeout=20)
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return state


def require_quiet(state: dict[str, Any], quiet: dict[str, float]) -> None:
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
    parts = {"cls": cls, "mean": mean}
    return torch.cat([parts[k] for k in PARTS[descriptor]], dim=1)


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
        tiles = compress.photo_tiles(path, decision, rule, spec["tiles"] == "top16")
        laps.mark("prepare")
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda" and not spec.get("int8")):
            cls, mean = compress.describe_batches(encoder, tiles, encoder_spec, config, device, batch)
        laps.mark("describe")
        bag = descriptor_of(cls, mean, params["descriptor"])[None].to(device).float()
        with torch.inference_mode():
            torch.softmax(model(bag, torch.ones(1, bag.shape[1], dtype=torch.bool, device=device))[0], 1).cpu()
        laps.mark("decide")
    return run


def probe_pipeline(config: dict[str, Any], device: torch.device, spec: dict[str, Any]) -> Callable:
    """Open, rescale, pad to square at 224 or 512 px, one E1 description, the probe's prediction."""
    from src.photo_dataset import prepare_square

    encoder = load_encoder(config, "E1", device)
    model = compress.load_probe(f"probe_{spec['size']}", 1)
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
        tiles = compress.photo_tiles(path, decision, rule, top16)
        laps.mark("prepare")
        pixels = np.ascontiguousarray(tiles.transpose(0, 3, 1, 2), dtype=np.float32) / 255
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
    root = dataset_dir(config, "primary")
    for photo in [*warmup, *timed]:           # photo files into the OS cache, untimed: only decoding is timed
        (root / photo).read_bytes()
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    rows = time_photos(run, [str(root / p) for p in timed], [str(root / p) for p in warmup], sync)
    out.write_text(json.dumps({"method": method, "device": device_name, "rows": rows, "summary": summarise(rows),
                               "peak_gpu_mib": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None,
                               "peak_rss_mib": rss_mib(), "rss_before_model_mib": baseline}, indent=2))


def read_timing(device: str) -> dict[str, Any]:
    """The timing results of a device: ``cost_time_<device>`` and any ``cost_time_<device>_<methods>`` run (a later
    run of a few methods, e.g. the ONNX model after the export), merged by key."""
    merged: dict[str, Any] = {"median_seconds_per_photo": {}, "peak_gpu_mib": {}, "peak_rss_mib": {}, "summary": {}}
    for folder in sorted(get_default_runs_dir().glob(f"cost_time_{device}*")):
        if (folder / "metrics.json").exists():
            result = read_json(folder / "metrics.json")
            for key in merged:
                merged[key].update(result[key])
    return merged


def time_all(config: dict[str, Any], device_name: str, only: list[str] | None = None) -> None:
    """Time every method (or only ``only``, stored as its own run so that earlier timings stay untouched).
    The machine is checked before every method."""
    require_clean_code()
    runs = get_default_runs_dir()
    if only:
        methods, name = only, f"cost_time_{device_name}_{'_'.join(only)}"
    else:
        methods, name = [m for m in TIMED if m != "final_onnx" and not (device_name == "gpu" and m in CPU_ONLY)], f"cost_time_{device_name}"
    if device_name == "gpu" and (bad := [m for m in methods if m in CPU_ONLY]):
        raise ValueError(f"{', '.join(bad)}: timed on the CPU only")
    if "final_onnx" in methods and not (runs / "cost_export" / "metrics.json").exists():
        raise RuntimeError("run `python -m src.compress export` before timing the ONNX model")
    start = machine_state(full=True)
    require_quiet(start, config["cost"]["quiet"])
    _, checksum = read_folds(config)
    recorder = RunRecorder(name, config=config, split_checksum=checksum, overwrite=True,
                           data_version={"primary": config["datasets"]["primary"]["version"]})
    results: dict[str, Any] = {}

    def save(end: dict[str, Any] | None) -> None:       # after every method, so a late failure loses nothing
        recorder.save_metrics({"partition": "none", "device": device_name, "methods": methods,
                               "median_seconds_per_photo": {m: r["summary"]["median"] for m, r in results.items()},
                               "summary": {m: r["summary"] for m, r in results.items()},
                               "peak_gpu_mib": {m: r["peak_gpu_mib"] for m, r in results.items()},
                               "peak_rss_mib": {m: r["peak_rss_mib"] for m, r in results.items()},
                               "machine_state_start": start, "machine_state_end": end})
        pd.DataFrame([{"method": m, "device": device_name, **row} for m, res in results.items() for row in res["rows"]]
                     ).to_csv(recorder.run_dir / "timing.csv", index=False)

    for method in methods:
        require_quiet(machine_state(), config["cost"]["quiet"])
        out = recorder.run_dir / f"{method}.json"
        subprocess.run([sys.executable, "-m", "src.cost", "time-one", "--method", method, "--device", device_name,
                        "--out", str(out)], check=True, cwd=REPO_ROOT)
        results[method] = read_json(out)
        save(None)
        print(f"{method}: median {results[method]['summary']['median']:.3f} s per photo ({device_name})", flush=True)
    save(machine_state(full=True))
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


def size_row(method: str, config: dict[str, Any], memo: dict | None = None) -> dict[str, Any]:
    """Parameters (encoder, decision-maker), bytes on disk and cache bytes of one method. ``memo`` holds what depends
    on the encoder only (parameters, file, cache), so that ``sizes`` loads each encoder once."""
    runs, memo = get_default_runs_dir(), {} if memo is None else memo
    spec = TIMED.get(method) or {"kind": "tiles", "head": VARIANTS[method][0]}    # a candidate name, e.g. distilled_e4
    if spec["kind"] == "tiles":
        code = ENCODER[spec["head"]]
        encoder = load_encoder(config, code, torch.device("cpu"))     # the 8-bit variant needs it anyway
        facts = memo.setdefault(code, {"params": count(encoder.parameters()), "file": encoder_file_bytes(config, code),
                                       "cache": dir_bytes(cache_folder(CACHES[code]))})
        state = torch.load(runs / head_runs(spec["head"])[0] / "model.pt", map_location="cpu", weights_only=True)
        if spec.get("int8"):
            buffer = io.BytesIO()
            torch.save(compress.quantise(encoder).state_dict(), buffer)
            encoder_bytes = buffer.getbuffer().nbytes
        else:
            encoder_bytes = facts["file"]
        return {"method": method, "params_encoder": facts["params"], "params_decider": count(state.values()),
                "disk_bytes": encoder_bytes + (runs / head_runs(spec["head"])[0] / "model.pt").stat().st_size,
                "cache_bytes": facts["cache"]}
    if spec["kind"] == "probe":
        facts = memo.setdefault("E1", {"params": count(load_encoder(config, "E1", torch.device("cpu")).parameters()),
                                       "file": encoder_file_bytes(config, "E1")})
        scaler, logistic = (step for _, step in compress.load_probe(method, 1).steps)
        stored = [scaler.mean_, scaler.scale_, logistic.coef_, logistic.intercept_]
        return {"method": method, "params_encoder": facts["params"], "params_decider": int(sum(a.size for a in stored)),
                "disk_bytes": facts["file"] + (runs / selected_runs(method)[0] / "model.pkl").stat().st_size,
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
    return {**size_row(exported["candidate"], config, memo), "method": method, "disk_bytes": sum(exported["bytes"].values())}


def sizes(config: dict[str, Any]) -> None:
    require_clean_code()
    names = [*METHODS, "e1_top16", "e4", "e4_top16", "distilled_e4", "attention_int8"]
    if (get_default_runs_dir() / "cost_export" / "metrics.json").exists():
        names.append("final_onnx")
    memo: dict = {}
    rows = [size_row(m, config, memo) for m in names]
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

    eligible = [n for n in order if val_f1[n] >= reference_f1 - margin_points / 100 - 1e-9]
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
    cpu = read_timing("cpu")["median_seconds_per_photo"]
    val = {n: variants[n]["val_macro_f1_mean"] for n in compress.CANDIDATES}
    seconds = {n: cpu[TIMING_KEY.get(n, n)] for n in compress.CANDIDATES}
    reference, reference_seconds = variants["attention"]["val_macro_f1_mean"], cpu["attention"]
    chosen, why = choose_final(val, seconds, reference, reference_seconds, config["compress"]["final_margin_points"],
                               compress.CANDIDATES)
    (root / "cost_selection.json").write_text(json.dumps(
        {"partition": "val", "selected": chosen, "reason": why, "code_version": version,
         "reference": {"val_macro_f1": reference, "cpu_seconds": reference_seconds},
         "candidates": {n: {"val_macro_f1": val[n], "cpu_seconds": seconds[n]} for n in compress.CANDIDATES}}, indent=2) + "\n")
    print(f"final small model: {chosen} ({why})", flush=True)


def plan_entries(selected: str) -> dict[str, dict[str, str]]:
    root = get_default_runs_dir()
    return {run: {"record_sha256": sha256(root / run / "record.json"), "checkpoint_sha256": sha256(root / run / "model.pt")}
            for run in head_runs(VARIANTS[selected][0])}


def plan(config: dict[str, Any]) -> None:
    """Freeze the choice (validation only) with the hashes of the five round-models, and make the final run's
    validation record. Safe to repeat: the plan cannot change, and the run folder is made once."""
    version = require_clean_code()
    root = get_default_runs_dir()
    selection = read_json(root / "cost_selection.json")
    _, checksum = read_folds(config)
    frozen = {"selection_partition": "val", "code_version": version, "selected": selection["selected"],
              "selection_sha256": sha256(root / "cost_selection.json"), "rule": compress.rule_of(config),
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
    # the round-1 head as the run's checkpoint: session_study.require_intact wants one in every tested run folder
    shutil.copy2(root / head_runs(VARIANTS[selection["selected"]][0])[0] / "model.pt", recorder.run_dir / "model.pt")
    recorder.save_metrics({"partition": "val", **{k: v for k, v in selection.items() if k != "partition"}})
    recorder.finish()


def evaluate_final(config: dict[str, Any]) -> None:
    """The one test evaluation of Stage 13. Everything that needs no test score (the checks against the plan, the
    five heads, the descriptions, the tile selection of the test photos: pixels only) is done first; the marker is
    written last, just before the first test prediction, so that a crash in the preparation does not use it up."""
    version = require_clean_code()
    root = get_default_runs_dir()
    frozen = read_json(root / "cost_plan.json")
    folds, checksum = read_folds(config)
    if frozen["selection_sha256"] != sha256(root / "cost_selection.json") or frozen["runs"] != plan_entries(frozen["selected"]):
        raise ValueError("the selection or a round-model changed after the plan was frozen")
    if frozen["rule"] != compress.rule_of(config) or frozen["split_checksum"] != checksum:
        raise ValueError("the tile rule or the split changed after the plan was frozen")
    name, folder = frozen["selected"], root / "cost_final"
    rows = compress.role_rows(folds, "test")
    ids = folds["photo_id"].to_numpy()
    photos = sorted({ids[r] for rs in rows.values() for r in rs})
    selection = compress.tile_selection(config, photos) if VARIANTS[name][1] == "top16" else None
    variant = compress.Variant(config, name, folds, selection)
    try:
        marker = (folder / "test_started.json").open("x")
    except FileExistsError:
        raise RuntimeError("the final small model was already tested or its test was started; no retry") from None
    with marker:
        json.dump({"evaluation_code": version, "plan_sha256": sha256(root / "cost_plan.json")}, marker)
    result = compress.score_variant(variant, folds, rows, config["classes"], "test")
    table = result["predictions"]
    table.to_csv(folder / "test_predictions.csv", index=False)
    scores = metrics.scores(table["true_class"], table["pred_class"], config["classes"])
    (folder / "test_metrics.json").write_text(json.dumps(
        {"partition": "test", **scores, "macro_f1_rounds": result["scores"], "macro_f1_mean": float(np.mean(result["scores"])),
         "tiles_used": result["tiles_used"], "code_version": version}, indent=2))


# ---------------------------------------------------------------------------------------------- report
def cost_row(label: str, sizes_row: dict[str, Any], timing: dict[str, Any], key: str, macro_f1: float | None) -> dict[str, Any]:
    """One row of the cost table: sizes, timing under ``key`` on both devices, and the test macro-F1."""
    return {"method": label, "params_encoder_m": sizes_row["params_encoder"] / 1e6, "params_decider_m": sizes_row["params_decider"] / 1e6,
            "disk_mb": sizes_row["disk_bytes"] / 2**20, "cache_gb": sizes_row["cache_bytes"] / 2**30,
            "seconds_per_photo": timing["gpu"]["median_seconds_per_photo"][key],
            "seconds_per_photo_cpu": timing["cpu"]["median_seconds_per_photo"][key],
            "peak_gpu_mb": timing["gpu"]["peak_gpu_mib"][key], "peak_cpu_rss_mb": timing["cpu"]["peak_rss_mib"][key],
            "macro_f1": macro_f1, "partition": "test"}


def method_rows(config: dict[str, Any]) -> pd.DataFrame:
    """The cost table: every main method (and the committee, if Stage 11 gave one), test macro-F1 from its test runs."""
    root = get_default_runs_dir()
    sizes_ = {r["method"]: r for r in read_json(root / "cost_sizes" / "metrics.json")["rows"]}
    timing = {d: read_timing(d) for d in ("gpu", "cpu")}
    rows = [cost_row(m, sizes_[m], timing, m, float(np.mean([read_json(root / run / "test_metrics.json")["macro_f1"]
                                                              for run in selected_runs(m)]))) for m in METHODS]
    members = config["cost"]["committee"]["members"]
    if members:
        parts = [r for r in rows if r["method"] in members]
        rows.append({"method": "committee", **{k: sum(p[k] for p in parts) for k in
                     ("params_encoder_m", "params_decider_m", "disk_mb", "cache_gb", "seconds_per_photo", "seconds_per_photo_cpu")},
                     "peak_gpu_mb": max(p["peak_gpu_mb"] for p in parts), "peak_cpu_rss_mb": max(p["peak_cpu_rss_mb"] for p in parts),
                     "macro_f1": config["cost"]["committee"]["test_macro_f1"], "partition": "test"})
    name = read_json(root / "cost_plan.json")["selected"]
    final = read_json(root / "cost_final" / "test_metrics.json")["macro_f1_mean"]
    # both distilled candidates have the one head's size
    rows.append(cost_row(f"final_small_model ({name})", sizes_.get(name) or sizes_["distilled_e4"], timing,
                         TIMING_KEY.get(name, name), final))
    return pd.DataFrame(rows)


def compression_rows(config: dict[str, Any]) -> pd.DataFrame:
    """Each variant: validation macro-F1 against seconds per photo (GPU and CPU)."""
    root = get_default_runs_dir()
    variants = read_json(root / "cost_fewer_tiles" / "metrics.json")["variants"]
    timing = {d: read_timing(d)["median_seconds_per_photo"] for d in ("gpu", "cpu")}
    reference = variants["attention"]["val_macro_f1_mean"]
    rows = []
    for name, v in variants.items():
        key = TIMING_KEY.get(name, name)
        rows.append({"variant": name, "what": "E1 + attention, all tiles (the reference)" if name == "attention" else name.replace("_", " "),
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
        v = variants[chosen]
        rows.append({"variant": "final_onnx", "what": f"{chosen} exported to ONNX (onnxruntime, CPU)", "val_macro_f1": v["val_macro_f1_mean"],
                     "val_macro_f1_sd": v["val_macro_f1_sd"], "difference_points": 100 * (v["val_macro_f1_mean"] - reference),
                     "tiles_used": v["tiles_used"], "seconds_per_photo": None,
                     "seconds_per_photo_cpu": timing["cpu"]["final_onnx"], "note": "same model, other runtime", "partition": "val"})
    return pd.DataFrame(rows)


def report(config: dict[str, Any]) -> None:
    from src import followup_report as fr
    from src.inventory_report import md_table

    root = get_default_runs_dir()
    t = config["cost"]["timing"]
    fmt = lambda x, f: "–" if pd.isna(x) else f.format(x)   # noqa: E731
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
                          "encoder params (M, frozen)": cost["params_encoder_m"].map("{:.1f}".format),
                          "decision-maker params (M)": cost["params_decider_m"].map("{:.2f}".format),
                          "files (MB)": cost["disk_mb"].map("{:.0f}".format), "training cache (GB)": cost["cache_gb"].map("{:.1f}".format),
                          "GPU s/photo": cost["seconds_per_photo"].map("{:.3f}".format),
                          "CPU s/photo": cost["seconds_per_photo_cpu"].map("{:.2f}".format),
                          "peak GPU (MB)": cost["peak_gpu_mb"].map("{:.0f}".format), "peak CPU RAM (MB)": cost["peak_cpu_rss_mb"].map("{:.0f}".format),
                          "test macro-F1 (%)": cost["macro_f1"].map(lambda x: fmt(100 * x, "{:.1f}"))})
    shown_c = pd.DataFrame({"variant": compression["variant"], "what": compression["what"],
                            "validation macro-F1 (%)": [f"{100 * m:.1f} ± {100 * s:.1f}" for m, s in zip(compression["val_macro_f1"], compression["val_macro_f1_sd"])],
                            "difference (points)": compression["difference_points"].map("{:+.1f}".format),
                            "tiles per photo": compression["tiles_used"].map(lambda x: fmt(x, "{:.0f}")),
                            "GPU s/photo": compression["seconds_per_photo"].map(lambda x: fmt(x, "{:.3f}")),
                            "CPU s/photo": compression["seconds_per_photo_cpu"].map(lambda x: fmt(x, "{:.2f}")),
                            "note": compression["note"]})
    p = paired.iloc[0]
    committee_note = ("" if config["cost"]["committee"]["members"] else
                      "\nThere is no committee row: Stage 11 had not produced a committee when this was built.\n")
    text = f"""# Stage 13: cost and compression

Built by `src/cost.py` from `runs/`; do not edit by hand. Declared 2026-11-21.

**Hardware dependence.** Every time and every memory figure comes from one machine, `halo` (RTX 4060 8 GB, Ryzen 7 7700,
{t["cpu_threads"]} threads for torch and onnxruntime), and depends on it. Times are medians over {t["photos"]} training photos of round {t["source_round"]}, one photo
at a time from the file on disk (decoding included, cold-disk reads not), after {t["warmup"]} discarded warm-up photos. GPU: fp16
autocast, {t["tile_batch"]["cuda"]} tiles per pass; CPU: fp32, {t["tile_batch"]["cpu"]} tiles per pass. Each method ran in its own process.

## 1. Cost of every main method

Accuracy is **test** macro-F1 (mean over 5 rounds) of the method's existing test runs; times, sizes and memory are measured here.
The tile method describes about 79 tiles per photo, so it is **slower per photo than a whole-image model**. The training
cache is the description cache (all 8 versions) or the resized-photo cache.

{md_table(shown)}
{committee_note}
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
    parser.add_argument("--methods", nargs="+", choices=list(TIMED), help="for time: only these (stored as their own run)")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.action == "time-one":
        time_one(args.method, args.device, args.out)
        return
    config = load_config("configs/cost.yaml")
    if args.action == "time":
        if not args.device:
            parser.error("time needs --device gpu or cpu")
        time_all(config, args.device, args.methods)
    else:
        ACTIONS[args.action](config)
    print(f"COST {args.action} DONE", flush=True)


if __name__ == "__main__":
    main()
