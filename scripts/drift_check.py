import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

EPSILON = 1e-6


def psi(expected, actual):
    expected_arr = np.array(expected, dtype=float)
    actual_arr = np.array(actual, dtype=float)

    expected_arr = np.clip(expected_arr, EPSILON, 1)
    actual_arr = np.clip(actual_arr, EPSILON, 1)

    expected_arr = expected_arr / expected_arr.sum()
    actual_arr = actual_arr / actual_arr.sum()

    return float(np.sum((expected_arr - actual_arr) * np.log(expected_arr / actual_arr)))


def numeric_psi(series, baseline):
    bins = baseline["bins"]
    expected = baseline["counts"]
    values = series.dropna().to_numpy()
    hist, _ = np.histogram(values, bins=bins)
    total = int(hist.sum())
    actual = (hist / total).tolist() if total > 0 else [0.0] * (len(bins) - 1)
    return psi(expected, actual)


def categorical_psi(series, baseline):
    baseline_counts = baseline["counts"]
    baseline_keys = list(baseline_counts.keys())

    series = series.fillna("__MISSING__").astype(str)
    current_counts = series.value_counts(normalize=True).to_dict()
    other_total = sum(value for key, value in current_counts.items() if key not in baseline_keys)

    expected = [baseline_counts[key] for key in baseline_keys] + [0.0]
    actual = [current_counts.get(key, 0.0) for key in baseline_keys] + [other_total]
    return psi(expected, actual)


def main():
    parser = argparse.ArgumentParser(description="Compute PSI-based drift checks")
    parser.add_argument("--baseline", required=True, help="Baseline JSON from build_baseline.py")
    parser.add_argument("--data", required=True, help="CSV file with recent data")
    parser.add_argument("--threshold", type=float, default=0.2, help="PSI threshold for drift")
    parser.add_argument("--report", default="", help="Optional JSON report output path")
    parser.add_argument("--html", default="", help="Optional HTML report output path")
    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline file not found: {baseline_path}")

    with baseline_path.open("r", encoding="utf-8") as file_handle:
        baseline = json.load(file_handle)

    df = pd.read_csv(args.data)
    results = []
    missing_columns = []

    for col, stats in baseline.get("numeric", {}).items():
        if col not in df.columns:
            missing_columns.append(col)
            continue
        psi_value = numeric_psi(df[col], stats)
        results.append({"column": col, "type": "numeric", "psi": psi_value})

    for col, stats in baseline.get("categorical", {}).items():
        if col not in df.columns:
            missing_columns.append(col)
            continue
        psi_value = categorical_psi(df[col], stats)
        results.append({"column": col, "type": "categorical", "psi": psi_value})

    if missing_columns:
        missing_list = ", ".join(sorted(set(missing_columns)))
        raise ValueError(f"Missing columns in recent data: {missing_list}")

    results = sorted(results, key=lambda item: item["psi"], reverse=True)
    max_psi = max((item["psi"] for item in results), default=0.0)
    status = "fail" if any(item["psi"] >= args.threshold for item in results) else "ok"

    print(f"PSI threshold: {args.threshold}")
    for item in results:
        label = "DRIFT" if item["psi"] >= args.threshold else "OK"
        print(f"{item['column']}: psi={item['psi']:.4f} ({label})")

    report = {
        "threshold": args.threshold,
        "max_psi": max_psi,
        "status": status,
        "results": results,
    }

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as file_handle:
            json.dump(report, file_handle, indent=2, sort_keys=True)

        if args.html:
                html_rows = "\n".join(
                        f"<tr><td>{item['column']}</td><td>{item['type']}</td><td>{item['psi']:.4f}</td><td>{'DRIFT' if item['psi'] >= args.threshold else 'OK'}</td></tr>"
                        for item in results
                )
                html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset=\"utf-8\" />
    <title>Drift Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 24px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #f4f4f4; }}
    </style>
</head>
<body>
    <h2>Drift Report</h2>
    <p>Status: <strong>{status.upper()}</strong></p>
    <p>Threshold: {args.threshold}</p>
    <p>Max PSI: {max_psi:.4f}</p>
    <table>
        <thead>
            <tr><th>Column</th><th>Type</th><th>PSI</th><th>Status</th></tr>
        </thead>
        <tbody>
            {html_rows}
        </tbody>
    </table>
</body>
</html>"""
                html_path = Path(args.html)
                html_path.parent.mkdir(parents=True, exist_ok=True)
                with html_path.open("w", encoding="utf-8") as file_handle:
                        file_handle.write(html_content)

    if status == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
