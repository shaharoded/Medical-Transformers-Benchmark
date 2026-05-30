import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ["auroc", "auprc", "f1_0_5", "best_f1", "minrp"]
SUPPORT_COLUMN = "n_pos"


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


def support_weighted_mean(group, metric):
    if SUPPORT_COLUMN not in group:
        return np.nan
    values = pd.to_numeric(group[metric], errors="coerce")
    weights = pd.to_numeric(group[SUPPORT_COLUMN], errors="coerce")
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return np.nan
    return float(np.average(values.loc[mask], weights=weights.loc[mask]))


def build_overall_by_seed(runs):
    rows = []
    for (model, seed), group in runs.groupby(["model", "seed"], sort=True):
        macro_row = {"model": model, "seed": seed, "outcome": "MACRO_AVERAGE"}
        weighted_row = {"model": model, "seed": seed, "outcome": "WEIGHTED_AVERAGE"}
        for metric in METRICS:
            macro_row[metric] = pd.to_numeric(group[metric], errors="coerce").mean()
            weighted_row[metric] = support_weighted_mean(group, metric)
        if SUPPORT_COLUMN in group:
            weighted_row["support_weight_sum"] = pd.to_numeric(
                group[SUPPORT_COLUMN], errors="coerce"
            ).sum()
        rows.extend([macro_row, weighted_row])
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    root = Path(args.root)
    all_runs = [load_model_runs(root, model) for model in args.models]
    all_runs = [df for df in all_runs if len(df)]
    if not all_runs:
        raise SystemExit(f"No seed results found under {root}")
    runs = pd.concat(all_runs, ignore_index=True)
    overall_by_seed = build_overall_by_seed(runs)
    rows = []
    for (model, outcome), group in runs.groupby(["model", "outcome"], sort=True):
        for metric in METRICS:
            if metric not in group:
                continue
            row = {"model": model, "outcome": outcome, "metric": metric}
            row.update(summarize(group[metric]))
            rows.append(row)
    for model, group in overall_by_seed.groupby("model", sort=True):
        macro = group[group["outcome"].eq("MACRO_AVERAGE")]
        for metric in METRICS:
            row = {"model": model, "outcome": "MACRO_AVERAGE", "metric": metric}
            row.update(summarize(macro[metric]))
            rows.append(row)
        weighted = group[group["outcome"].eq("WEIGHTED_AVERAGE")]
        for metric in METRICS:
            row = {"model": model, "outcome": "WEIGHTED_AVERAGE", "metric": metric}
            row.update(summarize(weighted[metric]))
            rows.append(row)
    summary = pd.DataFrame(rows)
    output = Path(args.output) if args.output else root / "summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_runs_output = output.with_name(f"{output.stem}_per_outcome_runs{output.suffix}")
    overall_by_seed_output = output.with_name(f"{output.stem}_overall_by_seed{output.suffix}")
    runs.to_csv(raw_runs_output, index=False)
    overall_by_seed.to_csv(overall_by_seed_output, index=False)
    summary.to_csv(output, index=False)
    print(f"Wrote {raw_runs_output}")
    print(f"Wrote {overall_by_seed_output}")
    print(f"Wrote {output}")
    print(summary[summary["outcome"].isin(["MACRO_AVERAGE", "WEIGHTED_AVERAGE"])].to_string(index=False))


if __name__ == "__main__":
    main()
