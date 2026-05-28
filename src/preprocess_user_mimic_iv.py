import argparse
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


DEFAULT_OUTCOMES = [
    "DEATH",
    "DISGLYCEMIA_Hyperglycemia",
    "DISGLYCEMIA_Hypoglycemia",
    "KIDNEY_COMPLICATION",
    "CARDIO-VASCULAR_DISORDER",
    "NERVOUS_SYSTEM_DISORDER",
    "NEUROVASCULAR_COMPLICATION",
    "SKIN_ULCER",
    "RETINOPATHY",
    "KETOACIDOSIS",
]

EVENT_OUTCOME_REGEX = {
    "DEATH": r"^DEATH(?:_EVENT)?$",
    "KIDNEY_COMPLICATION": r"^KIDNEY_COMPLICATION(?:_EVENT)?$",
    "CARDIO-VASCULAR_DISORDER": r"^CARDIO-?VASCULAR_DISORDER(?:_EVENT)?$",
    "NERVOUS_SYSTEM_DISORDER": r"^NERVOUS_SYSTEM_DISORDER(?:_EVENT)?$",
    "NEUROVASCULAR_COMPLICATION": r"^NEUROVASCULAR_COMPLICATION(?:_EVENT)?$",
    "SKIN_ULCER": r"^SKIN_ULCER(?:_EVENT)?$",
    "RETINOPATHY": r"^RETINOPATHY(?:_EVENT)?$",
    "KETOACIDOSIS": r"^KETOACIDOSIS(?:_EVENT)?$",
}

GLUCOSE_CONCEPT_REGEX = r"^(?:BASE_)?GLUCOSE_MEASURE(?:MENT)?$"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temporal_csv", default="data/temporal_data.csv")
    parser.add_argument("--context_csv", default="data/context_data.csv")
    parser.add_argument("--output_path", default="data/processed/user_mimic_iv.pkl")
    parser.add_argument("--input_days", type=float, default=2.0)
    parser.add_argument("--horizon_days", type=float, default=12.0)
    parser.add_argument("--label_mode", choices=["anytime", "future"], default="future")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--outcomes", nargs="*", default=DEFAULT_OUTCOMES)
    parser.add_argument("--death_token", default="DEATH_EVENT")
    return parser.parse_args()


def _to_numeric_value(series):
    numeric = pd.to_numeric(series, errors="coerce")
    true_false = series.astype(str).str.lower().map({"true": 1.0, "false": 0.0})
    return numeric.fillna(true_false)


def _resolve_path(path):
    if os.path.isabs(path) or os.path.exists(path):
        return path
    script_relative = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
    if os.path.exists(script_relative):
        return script_relative
    return path


def _matching_concepts(concepts, pattern):
    return sorted([c for c in concepts if pd.Series([c]).str.contains(pattern, regex=True).iloc[0]])


def main():
    args = parse_args()
    input_minutes = args.input_days * 24.0 * 60.0
    horizon_minutes = args.horizon_days * 24.0 * 60.0

    temporal = pd.read_csv(_resolve_path(args.temporal_csv))
    context = pd.read_csv(_resolve_path(args.context_csv))
    temporal["StartDateTime"] = pd.to_datetime(temporal["StartDateTime"])

    admissions = (
        temporal.loc[temporal["ConceptName"] == "ADMISSION", ["PatientId", "StartDateTime"]]
        .groupby("PatientId", as_index=False)["StartDateTime"]
        .min()
        .rename(columns={"StartDateTime": "admission_time"})
    )
    temporal = temporal.merge(admissions, on="PatientId", how="inner")
    temporal["minute"] = (
        (temporal["StartDateTime"] - temporal["admission_time"]).dt.total_seconds() / 60.0
    )
    temporal = temporal.loc[temporal["minute"] >= 0].copy()

    context = context.loc[context["PatientId"].isin(admissions["PatientId"])].copy()
    all_ids = np.array(sorted(context["PatientId"].unique()))
    train_valid_ids, test_ids = train_test_split(
        all_ids, test_size=0.2, random_state=args.seed
    )
    train_ids, valid_ids = train_test_split(
        train_valid_ids, test_size=0.2, random_state=args.seed
    )

    available_concepts = sorted(temporal["ConceptName"].unique())
    event_alias_map = {
        canonical: _matching_concepts(available_concepts, pattern)
        for canonical, pattern in EVENT_OUTCOME_REGEX.items()
        if canonical in args.outcomes
    }
    glucose_concepts = _matching_concepts(available_concepts, GLUCOSE_CONCEPT_REGEX)
    outcome_names = [
        outcome for outcome in args.outcomes
        if outcome in event_alias_map or outcome.startswith("DISGLYCEMIA_")
    ]
    source_outcome_set = {alias for aliases in event_alias_map.values() for alias in aliases}

    label_rows = []
    outcome_source = temporal.loc[
        temporal["ConceptName"].isin(source_outcome_set | set(glucose_concepts))
    ].copy()
    outcome_source["numeric_value"] = _to_numeric_value(outcome_source["Value"])
    for pid, group in outcome_source.groupby("PatientId"):
        labels = {"PatientId": pid}
        if args.label_mode == "future":
            in_window = group.loc[
                (group["minute"] > input_minutes)
                & (group["minute"] <= input_minutes + horizon_minutes)
            ]
        else:
            in_window = group
        for outcome in outcome_names:
            if outcome == "DISGLYCEMIA_Hyperglycemia":
                rows = in_window.loc[
                    in_window["ConceptName"].isin(glucose_concepts)
                    & (in_window["numeric_value"] >= 180.0)
                ]
            elif outcome == "DISGLYCEMIA_Hypoglycemia":
                rows = in_window.loc[
                    in_window["ConceptName"].isin(glucose_concepts)
                    & (in_window["numeric_value"] <= 70.0)
                ]
            else:
                rows = in_window.loc[in_window["ConceptName"].isin(event_alias_map.get(outcome, []))]
            labels[outcome] = int(len(rows) > 0)
            labels[outcome + "__first_hour"] = (
                float(rows["minute"].min() / 60.0) if len(rows) else np.inf
            )
        label_rows.append(labels)
    labels = pd.DataFrame(label_rows)
    labels = context[["PatientId"]].merge(labels, on="PatientId", how="left")
    for outcome in outcome_names:
        labels[outcome] = labels[outcome].fillna(0).astype(int)
        labels[outcome + "__first_hour"] = labels[outcome + "__first_hour"].fillna(np.inf)
    labels = labels.rename(columns={"PatientId": "ts_id"})

    terminal_aliases = {"RELEASE", "RELEASE_EVENT", args.death_token}
    terminal_aliases.update(["ADMISSION", "RELEASE", "DEATH"])

    temporal_inputs = temporal.loc[
        (temporal["minute"] <= input_minutes)
        & (~temporal["ConceptName"].isin(source_outcome_set))
        & (~temporal["ConceptName"].isin(terminal_aliases))
    ].copy()
    temporal_inputs["value"] = _to_numeric_value(temporal_inputs["Value"])

    categorical = temporal_inputs["value"].isna()
    temporal_inputs.loc[categorical, "ConceptName"] = (
        temporal_inputs.loc[categorical, "ConceptName"].astype(str)
        + "_"
        + temporal_inputs.loc[categorical, "Value"].astype(str)
    )
    temporal_inputs.loc[categorical, "value"] = 1.0
    temporal_inputs = temporal_inputs.dropna(subset=["value"])
    temporal_inputs = temporal_inputs.rename(
        columns={"PatientId": "ts_id", "ConceptName": "variable"}
    )[["ts_id", "minute", "variable", "value"]]

    static_varis = [c for c in context.columns if c != "PatientId"]
    static = context.rename(columns={"PatientId": "ts_id"}).melt(
        id_vars="ts_id", value_vars=static_varis, var_name="variable", value_name="value"
    )
    static["minute"] = 0.0
    data = pd.concat([temporal_inputs, static[["ts_id", "minute", "variable", "value"]]])
    data = data.groupby(["ts_id", "minute", "variable"], as_index=False)["value"].mean()

    metadata = {
        "static_varis": static_varis,
        "outcome_names": outcome_names,
        "outcome_source_aliases": event_alias_map,
        "glucose_concepts": glucose_concepts,
        "glucose_hyper_threshold": 180.0,
        "glucose_hypo_threshold": 70.0,
        "death_token": args.death_token,
        "input_hours": args.input_days * 24.0,
        "horizon_hours": args.horizon_days * 24.0,
        "label_mode": args.label_mode,
    }

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    pickle.dump(
        [data, labels, train_ids, valid_ids, test_ids, metadata],
        open(args.output_path, "wb"),
    )
    print("Wrote", args.output_path)
    print("Rows:", len(data), "patients:", data["ts_id"].nunique())
    print("Outcomes:", outcome_names)


if __name__ == "__main__":
    main()
