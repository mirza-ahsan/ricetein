"""List every number with a decimal point in paper/sections/*.tex and say whether it appears in a result table
(results/tables/final_*.csv) or a report (docs/reports/*.md). Standard library only.

Usage: python3 scripts/check_paper_numbers.py [--all]   (default: print only the numbers that were not found)
A number x with d decimals counts as found when a number in a source, rounded to d decimals, equals x
(so 0.3095978 in a report matches 0.31 in the text). The sign is ignored.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NUMBER = re.compile(r"(?<![\w.])(\d+(?:,\d{3})*\.\d+)(?![\w])")
SOURCES = sorted((ROOT / "results/tables").glob("final_*.csv")) + sorted((ROOT / "docs/reports").glob("*.md"))


def source_numbers() -> dict:
    found: dict = {}
    for path in SOURCES:
        for token in re.findall(r"\d+\.\d+(?:e[+-]?\d+)?", path.read_text(encoding="utf-8")):
            value = float(token)
            for decimals in range(1, 7):
                found.setdefault((decimals, round(value, decimals)), path.name)
    return found


def strip_comments(text: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in text.splitlines())


def main() -> None:
    show_all = "--all" in sys.argv
    sources = source_numbers()
    missing = 0
    for path in sorted((ROOT / "paper/sections").glob("*.tex")):
        for number_line, line in enumerate(strip_comments(path.read_text(encoding="utf-8")).splitlines(), 1):
            for match in NUMBER.finditer(line):
                text = match.group(1).replace(",", "")
                decimals = len(text.split(".")[1])
                origin = sources.get((decimals, round(float(text), decimals)))
                if origin is None:
                    missing += 1
                if origin is None or show_all:
                    print(f"{path.name}:{number_line}: {text:>10}  {origin or 'NOT FOUND'}")
    print(f"{missing} numbers not found")


if __name__ == "__main__":
    main()
