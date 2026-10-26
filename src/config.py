import argparse
import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
KAGGLE_WORKING = Path("/kaggle/working")


def read_yaml(filepath: str) -> dict:
    with open(filepath, "r") as file:
        data = yaml.safe_load(file)

    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="")
