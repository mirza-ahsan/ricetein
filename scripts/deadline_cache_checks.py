"""Run the declared repeatability checks for the two completed core caches."""

import json

from src import cache_report, embed
from src.config import REPO_ROOT, load_config
from src.records import require_clean_code


if __name__ == "__main__":
    checks = {}
    for name in ("e1_dinov3_vitb16", "wholephoto"):
        config = load_config(f"configs/embed/{name}.yaml")
        checks[name] = {
            "cache": embed.validate_cache(config),
            "repeatability": cache_report.check_example(config),
        }
        print(f"Core cache checks passed: {name}", flush=True)
    target = REPO_ROOT / "docs/reports/deadline_core_cache_checks.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"code_version": require_clean_code(), "checks": checks}, indent=2) + "\n")
