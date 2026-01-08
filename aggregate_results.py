"""
aggregate_results.py
Aggregate result CSVs into a single combined CSV.
"""

import csv
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def main():
    results_dir = Path("results")
    out_path = results_dir / "all_results.csv"
    results_dir.mkdir(parents=True, exist_ok=True)

    combined = []

    # Hypernetwork results
    hyper_rows = read_rows(results_dir / "hyper_vit_cifar100_results.csv")
    for row in hyper_rows:
        row["method"] = "hyper_vit_lora"
        combined.append(row)

    # Baseline all classes
    baseline_all_rows = read_rows(results_dir / "baseline_vit_all_results.csv")
    for row in baseline_all_rows:
        row["method"] = "baseline_vit_all"
        combined.append(row)

    # Baseline joint split
    baseline_joint_rows = read_rows(results_dir / "baseline_vit_joint_split_results.csv")
    for row in baseline_joint_rows:
        row["method"] = "baseline_vit_joint_split"
        combined.append(row)

    if not combined:
        print("No result files found to aggregate.")
        return

    fieldnames = sorted({key for row in combined for key in row.keys()})
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
