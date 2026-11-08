"""Diagnostic GPU shape/memory checks on random tensors; never reads research photos or scores."""

import argparse
import json

import torch

from src.config import load_config, output_root
from src.stage5 import MODELS
from src.train_wholeimage import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pipeline.yaml")
    args = parser.parse_args()
    load_config(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run outside the sandbox or on Kaggle")
    results = []
    for name in MODELS:
        for size, batch in ((128, 32), (224, 32), (448, 16)):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            config = load_config(f"configs/wholeimage/{name}_224.yaml")
            model = build_model(config, size).cuda().train()
            x = torch.rand(batch, 3, size, size, device="cuda")
            try:
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = model(x).float().square().mean()
                loss.backward()
                torch.cuda.synchronize()
                result = {
                    "model": name,
                    "size": size,
                    "batch": batch,
                    "fits": True,
                    "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
                }
            except torch.OutOfMemoryError:
                result = {"model": name, "size": size, "batch": batch, "fits": False}
            results.append(result)
            print(json.dumps(result), flush=True)
            model.cpu()
            del model, x
            if "loss" in locals():
                del loss
            torch.cuda.empty_cache()
    folder = output_root() / "runs"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "gpu_preflight.json").write_text(
        json.dumps(
            {
                "kind": "random-tensor diagnostic, no research scores",
                "gpu": torch.cuda.get_device_name(0),
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
