import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_BINS = 10


def parse_list(value):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def safe_bins(values, bins):
    if values.size == 0:
        return [0.0, 1.0]

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.quantile(values, quantiles)
    edges = np.unique(edges)

    if edges.size < 2 or np.any(np.diff(edges) <= 0):
        min_val = float(np.min(values))
        max_val = float(np.max(values))
        if min_val == max_val:
            return [min_val - 0.5, max_val + 0.5]
        return [min_val, max_val]

    return [float(x) for x in edges]


def build_baseline(df, bins):
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [col for col in df.columns if col not in numeric_cols]

    baseline = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(df)),
        "numeric": {},
        "categorical": {},
    }

    for col in numeric_cols:
        values = df[col].dropna().to_numpy()
        bins_list = safe_bins(values, bins)
        hist, _ = np.histogram(values, bins=bins_list)
        total = int(hist.sum())
        counts = (hist / total).tolist() if total > 0 else [0.0] * (len(bins_list) - 1)
        baseline["numeric"][col] = {
            "bins": bins_list,
            "counts": counts,
            "mean": float(np.nanmean(df[col])) if len(df[col]) else 0.0,
            "std": float(np.nanstd(df[col])) if len(df[col]) else 0.0,
        }

    for col in categorical_cols:
        series = df[col].fillna("__MISSING__").astype(str)
        counts = series.value_counts(normalize=True).to_dict()
        baseline["categorical"][col] = {"counts": counts}

    return baseline


def main():
    parser = argparse.ArgumentParser(description="Build baseline stats for drift detection")
    parser.add_argument("--data", required=True, help="CSV file with baseline data")
    parser.add_argument("--output", required=True, help="Path to write baseline JSON")
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS, help="Number of bins for numeric PSI")
    parser.add_argument("--target", default="diabetes", help="Target column to exclude")
    parser.add_argument("--exclude", default="", help="Comma-separated columns to exclude")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    exclude = set(parse_list(args.exclude))
    if args.target:
        exclude.update(parse_list(args.target))

    df = df.drop(columns=[col for col in exclude if col in df.columns], errors="ignore")
    baseline = build_baseline(df, args.bins)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_handle:
        json.dump(baseline, file_handle, indent=2, sort_keys=True)

    print(f"Baseline written to {output_path}")


if __name__ == "__main__":
    main()
