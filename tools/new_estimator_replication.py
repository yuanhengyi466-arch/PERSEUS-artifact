"""Prepare a clean estimator working copy, preserving released observations."""
import argparse
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    destination = args.destination.resolve()
    if destination.exists():
        raise SystemExit("Destination already exists; choose a new directory.")
    if destination.is_relative_to(ROOT):
        raise SystemExit("Choose a destination outside the artifact directory.")
    destination.mkdir(parents=True)
    for folder in ("experiments", "research_plan", "baselines"):
        shutil.copytree(ROOT / folder, destination / folder, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    relative = Path("rebuttal_20260929/reviewer_4udk")
    (destination / relative).mkdir(parents=True)
    for source in (ROOT / relative).iterdir():
        if source.is_file() and (source.suffix == ".py" or source.name in {"PROTOCOL.md", "THROUGHPUT_PROTOCOL.md"}):
            shutil.copy2(source, destination / relative / source.name)
    relative = Path("paper/workload_drift_residual_field/results_duckdb_k1450/duckdb_execution_probe.csv")
    (destination / relative).parent.mkdir(parents=True)
    shutil.copy2(ROOT / relative, destination / relative)
    for name in ("requirements.txt", "LICENSE"):
        shutil.copy2(ROOT / name, destination / name)
    print("Created a clean estimator working copy at", destination)


if __name__ == "__main__":
    main()
