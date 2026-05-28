import argparse
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    CONCEPT_SUPPORT_THRESHOLD,
    DEFAULT_CONTEXT_CSV,
    DEFAULT_PROCESSED_PATH,
    DEFAULT_TEMPORAL_CSV,
    EVENT_OUTCOME_REGEX,
    GLUCOSE_CONCEPT_REGEX,
    GLUCOSE_HYPER_RECURRENT_THRESHOLD,
    GLUCOSE_HYPER_SEVERE_THRESHOLD,
    GLUCOSE_HYPO_RECURRENT_THRESHOLD,
    GLUCOSE_HYPO_SEVERE_THRESHOLD,
    OUTCOME_NAMES,
    OUTCOME_SUPPORT_THRESHOLD,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temporal_csv", default=DEFAULT_TEMPORAL_CSV)
    parser.add_argument("--context_csv", default=DEFAULT_CONTEXT_CSV)
    parser.add_argument("--output_path", default=DEFAULT_PROCESSED_PATH)
    parser.add_argument("--input_days", type=float, default=2.0)
    parser.add_argument("--horizon_days", type=float, default=12.0)
    parser.add_argument("--label_mode", choices=["anytime", "future"], default="future")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--outcomes", nargs="*", default=OUTCOME_NAMES)
    parser.add_argument("--concept_support_threshold", type=float, default=CONCEPT_SUPPORT_THRESHOLD)
    parser.add_argument("--outcome_support_threshold", type=float, default=OUTCOME_SUPPORT_THRESHOLD)
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


def _patient_support_by_concept(frame, patient_count):
    support = frame.groupby("ConceptName")["PatientId"].nunique() / patient_count
    return support.sort_values(ascending=False)


def _dysglycemia_events(glucose_rows, outcome_name, severe_threshold, recurrent_threshold, direction):
    if len(glucose_rows) == 0:
        return glucose_rows.copy()
    events = []
    comparator = np.less_equal if direction == "low" else np.greater_equal
    for _, group in glucose_rows.sort_values(["PatientId", "minute"]).groupby("PatientId"):
        group = group.copy()
        severe = comparator(group["numeric_value"], severe_threshold)
        recurrent = comparator(group["numeric_value"], recurrent_threshold)
        recurrent_count = recurrent.groupby(group["PatientId"]).cumsum()
        event_mask = severe | (recurrent & (recurrent_count >= 2))
        curr = group.loc[event_mask].copy()
        if len(curr):
            curr["ConceptName"] = outcome_name
            events.append(curr)
    if not events:
        return glucose_rows.iloc[0:0].copy()
    return pd.concat(events, ignore_index=True)


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
    patient_count = len(all_ids)
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

    outcome_source = temporal.loc[
        temporal["ConceptName"].isin(source_outcome_set | set(glucose_concepts))
    ].copy()
    outcome_source["numeric_value"] = _to_numeric_value(outcome_source["Value"])

    outcome_event_frames = []
    for outcome, aliases in event_alias_map.items():
        curr = outcome_source.loc[outcome_source["ConceptName"].isin(aliases)].copy()
        if len(curr):
            curr["ConceptName"] = outcome
            outcome_event_frames.append(curr)
    glucose_rows = outcome_source.loc[
        outcome_source["ConceptName"].isin(glucose_concepts)
        & outcome_source["numeric_value"].notna()
    ].copy()
    hyper = _dysglycemia_events(
        glucose_rows,
        "DISGLYCEMIA_Hyperglycemia",
        GLUCOSE_HYPER_SEVERE_THRESHOLD,
        GLUCOSE_HYPER_RECURRENT_THRESHOLD,
        "high",
    )
    if len(hyper):
        outcome_event_frames.append(hyper)
    hypo = _dysglycemia_events(
        glucose_rows,
        "DISGLYCEMIA_Hypoglycemia",
        GLUCOSE_HYPO_SEVERE_THRESHOLD,
        GLUCOSE_HYPO_RECURRENT_THRESHOLD,
        "low",
    )
    if len(hypo):
        outcome_event_frames.append(hypo)
    outcome_events = (
        pd.concat(outcome_event_frames, ignore_index=True)
        if outcome_event_frames
        else pd.DataFrame(columns=outcome_source.columns)
    )

    label_rows = []
    for pid, group in outcome_events.groupby("PatientId"):
        labels = {"PatientId": pid}
        if args.label_mode == "future":
            in_window = group.loc[
                (group["minute"] > input_minutes)
                & (group["minute"] <= input_minutes + horizon_minutes)
            ]
        else:
            in_window = group
        for outcome in outcome_names:
            rows = in_window.loc[in_window["ConceptName"] == outcome]
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
    outcome_support = {
        outcome: float(labels[outcome].sum() / patient_count)
        for outcome in outcome_names
    }
    dropped_outcomes = [
        outcome for outcome in outcome_names
        if outcome_support[outcome] < args.outcome_support_threshold
    ]
    outcome_names = [
        outcome for outcome in outcome_names
        if outcome_support[outcome] >= args.outcome_support_threshold
    ]
    labels = labels.rename(columns={"PatientId": "ts_id"})

    terminal_aliases = {"RELEASE", "RELEASE_EVENT", args.death_token}
    terminal_aliases.update(["ADMISSION", "RELEASE", "DEATH"])

    temporal_inputs = temporal.loc[
        (temporal["minute"] <= input_minutes)
        & (~temporal["ConceptName"].isin(source_outcome_set))
        & (~temporal["ConceptName"].isin(terminal_aliases))
    ].copy()
    concept_support = _patient_support_by_concept(temporal_inputs, patient_count)
    kept_input_concepts = set(
        concept_support.loc[concept_support >= args.concept_support_threshold].index
    )
    dropped_input_concepts = sorted(
        concept_support.loc[concept_support < args.concept_support_threshold].index
    )
    temporal_inputs = temporal_inputs.loc[
        temporal_inputs["ConceptName"].isin(kept_input_concepts)
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
        "configured_outcome_names": args.outcomes,
        "outcome_support": outcome_support,
        "dropped_outcomes": dropped_outcomes,
        "outcome_source_aliases": event_alias_map,
        "glucose_concepts": glucose_concepts,
        "glucose_hyper_severe_threshold": GLUCOSE_HYPER_SEVERE_THRESHOLD,
        "glucose_hyper_recurrent_threshold": GLUCOSE_HYPER_RECURRENT_THRESHOLD,
        "glucose_hypo_severe_threshold": GLUCOSE_HYPO_SEVERE_THRESHOLD,
        "glucose_hypo_recurrent_threshold": GLUCOSE_HYPO_RECURRENT_THRESHOLD,
        "synthetic_outcomes": ["DISGLYCEMIA_Hyperglycemia", "DISGLYCEMIA_Hypoglycemia"],
        "concept_support_threshold": args.concept_support_threshold,
        "outcome_support_threshold": args.outcome_support_threshold,
        "kept_input_concepts": sorted(kept_input_concepts),
        "dropped_input_concepts": dropped_input_concepts,
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
    print("Dropped low-support outcomes:", dropped_outcomes)
    print("Dropped low-support input concepts:", dropped_input_concepts)


if __name__ == "__main__":
    main()
