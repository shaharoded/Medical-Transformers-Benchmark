import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ["auroc", "auprc", "f1_0_5", "best_f1", "minrp"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/user_mimic_iv/multiseed")
    parser.add_argument("--models", nargs="+", default=["strats", "grud"])
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def summarize(values):
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy()
    if len(arr) == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan, "sem": np.nan, "ci95_low": np.nan, "ci95_high": np.nan}
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    sem = float(std / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    margin = 1.96 * sem
    return {
        "n": int(len(arr)),
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
    }


def load_model_runs(root, model):
    rows = []
    for metrics_path in sorted((root / model).glob("seed_*/test_per_outcome_metrics.csv")):
        seed = metrics_path.parent.name.removeprefix("seed_")
        table = pd.read_csv(metrics_path)
        table["model"] = model
        table["seed"] = seed
        rows.append(table)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main():
    args = parse_args()
    root = Path(args.root)
    all_runs = [load_model_runs(root, model) for model in args.models]
    all_runs = [df for df in all_runs if len(df)]
    if not all_runs:
        raise SystemExit(f"No seed results found under {root}")
    runs = pd.concat(all_runs, ignore_index=True)
    rows = []
    for (model, outcome), group in runs.groupby(["model", "outcome"], sort=True):
        for metric in METRICS:
            if metric not in group:
                continue
            row = {"model": model, "outcome": outcome, "metric": metric}
            row.update(summarize(group[metric]))
            rows.append(row)
    for model, group in runs.groupby("model", sort=True):
        macro = group.groupby("seed")[METRICS].mean(numeric_only=True).reset_index()
        for metric in METRICS:
            row = {"model": model, "outcome": "MACRO_AVERAGE", "metric": metric}
            row.update(summarize(macro[metric]))
            rows.append(row)
    summary = pd.DataFrame(rows)
    output = Path(args.output) if args.output else root / "summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    print(f"Wrote {output}")
    print(summary[summary["outcome"].eq("MACRO_AVERAGE")].to_string(index=False))


if __name__ == "__main__":
    main()
