import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from config import (
    COMPUTED_OUTCOMES,
    CONCEPT_SUPPORT_THRESHOLD,
    CREATININE_ABSOLUTE_THRESHOLD,
    CREATININE_CONCEPT_REGEX,
    CREATININE_RATIO_THRESHOLD,
    CREATININE_VALUE_MAX,
    CREATININE_VALUE_MIN,
    DEFAULT_CONTEXT_CSV,
    DEFAULT_PROCESSED_PATH,
    DEFAULT_TEMPORAL_CSV,
    EVENT_OUTCOME_REGEX,
    GLUCOSE_CONCEPT_REGEX,
    GLUCOSE_HYPER_HIGH_IND_MIN,
    GLUCOSE_HYPER_SEVERE_THRESHOLD,
    GLUCOSE_HYPO_LOW_IND_MAX,
    GLUCOSE_HYPO_LOW_IND_MIN,
    GLUCOSE_HYPO_SEVERE_THRESHOLD,
    GLUCOSE_VALUE_MAX,
    GLUCOSE_VALUE_MIN,
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
    # Token support filter is disabled by default: low-support input concepts stay in the
    # vocabulary so STraTS / GRU-D see the same token surface as the Mediator-driven models.
    # (Pass any positive value to re-enable the legacy drop behaviour.)
    parser.add_argument("--concept_support_threshold", type=float, default=0.0)
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


# ---------------------------------------------------------------------------
# Complication-event derivation — Mediator TAK rules, exact match.
#
# Each function takes a measurement subset (glucose or creatinine rows with
# a `numeric_value` column) and returns the subset of measurement rows that
# satisfy the corresponding Mediator event rule. The emitted rows are
# relabelled with the Mediator event name (e.g. "HYPERGLYCEMIA_EVENT") and
# Value="True", matching the `value="True"` output of the abstraction rule.
#
# This keeps the per-outcome label support directly comparable to the
# Mediator's KBTA output for the same five events.
# ---------------------------------------------------------------------------

def _emit_event(rows, event_name):
    """
    Purpose: Stamp the selected measurement rows with a Mediator-style event
    label and a True value, so they slot into the outcome-event table.
    """
    if len(rows) == 0:
        return rows
    out = rows.copy()
    out["ConceptName"] = event_name
    out["Value"] = "True"
    return out


def _hyperglycemia_events(glucose_rows):
    """
    HYPERGLYCEMIA_EVENT (Mediator HYPERGLYCEMIA.xml):
      Fires at glucose g if:
        (A1) g >= 250                               — single severe measurement, OR
        (S1) g >= 180 AND prior glucose >= 180     — STEADY_GLUCOSE_HIGH_AFTER_FIRST
    """
    out_frames = []
    for _, group in glucose_rows.sort_values(["PatientId", "minute"]).groupby("PatientId"):
        v = group["numeric_value"].to_numpy()
        high_ind = v >= GLUCOSE_HYPER_HIGH_IND_MIN
        # Prior HIGH_GLUCOSE_IND count (excluding the current row).
        prior_high = np.concatenate(([0], np.cumsum(high_ind)[:-1]))
        severe = v >= GLUCOSE_HYPER_SEVERE_THRESHOLD
        recurrent = high_ind & (prior_high >= 1)
        mask = severe | recurrent
        if mask.any():
            out_frames.append(_emit_event(group.loc[mask], "HYPERGLYCEMIA_EVENT"))
    return pd.concat(out_frames, ignore_index=True) if out_frames else glucose_rows.iloc[0:0].copy()


def _hypoglycemia_events(glucose_rows):
    """
    HYPOGLYCEMIA_EVENT (Mediator HYPOGLYCEMIA.xml):
      Fires at glucose g if:
        (A1) g <= 54                                            — single severe measurement, OR
        (S2) g <= 70 AND prior glucose in [20, 70]             — STEADY_GLUCOSE_LOW_AFTER_FIRST
              (LOW_GLUCOSE_IND requires 20 <= g <= 70)
    """
    out_frames = []
    for _, group in glucose_rows.sort_values(["PatientId", "minute"]).groupby("PatientId"):
        v = group["numeric_value"].to_numpy()
        low_ind = (v >= GLUCOSE_HYPO_LOW_IND_MIN) & (v <= GLUCOSE_HYPO_LOW_IND_MAX)
        prior_low = np.concatenate(([0], np.cumsum(low_ind)[:-1]))
        severe = v <= GLUCOSE_HYPO_SEVERE_THRESHOLD
        recurrent = (v <= GLUCOSE_HYPO_LOW_IND_MAX) & (prior_low >= 1)
        mask = severe | recurrent
        if mask.any():
            out_frames.append(_emit_event(group.loc[mask], "HYPOGLYCEMIA_EVENT"))
    return pd.concat(out_frames, ignore_index=True) if out_frames else glucose_rows.iloc[0:0].copy()


def _severe_hyperglycemia_events(glucose_rows):
    """SEVERE_HYPERGLYCEMIA_EVENT: glucose >= 250 (single severe measurement)."""
    mask = glucose_rows["numeric_value"] >= GLUCOSE_HYPER_SEVERE_THRESHOLD
    return _emit_event(glucose_rows.loc[mask], "SEVERE_HYPERGLYCEMIA_EVENT")


def _severe_hypoglycemia_events(glucose_rows):
    """SEVERE_HYPOGLYCEMIA_EVENT: glucose <= 54 (single severe measurement)."""
    mask = glucose_rows["numeric_value"] <= GLUCOSE_HYPO_SEVERE_THRESHOLD
    return _emit_event(glucose_rows.loc[mask], "SEVERE_HYPOGLYCEMIA_EVENT")


def _kidney_complication_events(creatinine_rows):
    """
    KIDNEY_COMPLICATION_EVENT (Mediator KIDNEY_COMPLICATION.xml):
      Fires at creatinine c if:
        (C1) c >= 4.0                                       — AKI Stage 3 (absolute), OR
        (R1) c / baseline >= 2.0                            — AKI Stage 2+
             baseline = first creatinine measurement in admission
             (CREATININE_REL_SERUM_MEASURE, parameter how='before' idx=0)
    """
    out_frames = []
    for _, group in creatinine_rows.sort_values(["PatientId", "minute"]).groupby("PatientId"):
        v = group["numeric_value"].to_numpy()
        baseline = v[0] if len(v) > 0 else np.nan
        severe_abs = v >= CREATININE_ABSOLUTE_THRESHOLD
        # The ratio rule needs a usable baseline (first measurement, positive).
        # Index 0 is the baseline itself; the rule applies from index 1 onwards.
        ratio_high = np.zeros_like(v, dtype=bool)
        if baseline and baseline > 0 and len(v) > 1:
            ratio = v[1:] / baseline
            ratio_high[1:] = ratio >= CREATININE_RATIO_THRESHOLD
        mask = severe_abs | ratio_high
        if mask.any():
            out_frames.append(_emit_event(group.loc[mask], "KIDNEY_COMPLICATION_EVENT"))
    return pd.concat(out_frames, ignore_index=True) if out_frames else creatinine_rows.iloc[0:0].copy()


def main():
    args = parse_args()
    input_minutes = args.input_days * 24.0 * 60.0
    horizon_minutes = args.horizon_days * 24.0 * 60.0

    temporal = pd.read_csv(_resolve_path(args.temporal_csv))
    context = pd.read_csv(_resolve_path(args.context_csv))
    temporal["StartDateTime"] = pd.to_datetime(temporal["StartDateTime"])

    # ---------------------------------------------------------------------
    # Mediator measurement value bounds — out-of-range or non-numeric rows
    # for measurements that the Mediator constrains are dropped entirely so
    # the strats ETL sees the same valid-measurement subset Mediator does.
    # Without these filters extreme/noise readings would inflate the
    # recurrent-low / severe / kidney branches relative to Mediator output.
    #   GLUCOSE_MEASURE.xml         → [20, 1200] mg/dL
    #   CREATININE_SERUM_MEASURE.xml → [0.1, 20] mg/dL
    # ---------------------------------------------------------------------
    all_concepts = sorted(temporal["ConceptName"].astype(str).unique())
    for label, concept_regex, vmin, vmax in (
        ("glucose-bounds",     GLUCOSE_CONCEPT_REGEX,     GLUCOSE_VALUE_MIN,     GLUCOSE_VALUE_MAX),
        ("creatinine-bounds",  CREATININE_CONCEPT_REGEX,  CREATININE_VALUE_MIN,  CREATININE_VALUE_MAX),
    ):
        matched_concepts = set(_matching_concepts(all_concepts, concept_regex))
        if not matched_concepts:
            continue
        row_idx = temporal.index[temporal["ConceptName"].isin(matched_concepts)]
        values  = pd.to_numeric(temporal.loc[row_idx, "Value"], errors="coerce")
        in_range = values.between(vmin, vmax, inclusive="both")
        drop_idx = row_idx[~in_range.fillna(False)]
        dropped_n = len(drop_idx)
        if dropped_n:
            temporal = temporal.drop(drop_idx)
        print(f"[{label}] dropped {dropped_n:,} rows outside [{vmin:g}, {vmax:g}] "
              f"(of {len(row_idx):,} total).")

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

    # ---------------------------------------------------------------------
    # Build the per-outcome alias map for the pass-through Complications,
    # and locate the measurement-derived concept names (glucose, creatinine)
    # that this ETL needs in order to *compute* the five Mediator-aligned
    # events: HYPER/HYPO/SEVERE_HYPER/SEVERE_HYPO and KIDNEY_COMPLICATION.
    # ---------------------------------------------------------------------
    available_concepts = sorted(temporal["ConceptName"].unique())
    event_alias_map = {
        canonical: _matching_concepts(available_concepts, pattern)
        for canonical, pattern in EVENT_OUTCOME_REGEX.items()
        if canonical in args.outcomes
    }
    glucose_concepts    = _matching_concepts(available_concepts, GLUCOSE_CONCEPT_REGEX)
    creatinine_concepts = _matching_concepts(available_concepts, CREATININE_CONCEPT_REGEX)

    # Decide which outcomes will actually be looked at downstream: any
    # pass-through outcome with at least one matching alias, plus any
    # computed outcome we know how to derive in this script.
    requested = set(args.outcomes)
    outcome_names = [
        outcome for outcome in args.outcomes
        if outcome in event_alias_map or outcome in COMPUTED_OUTCOMES
    ]
    source_outcome_set = {alias for aliases in event_alias_map.values() for alias in aliases}

    # ---------------------------------------------------------------------
    # Build the outcome-event table:
    #   (a) pass-through events copied straight from raw temporal rows, and
    #   (b) computed events derived from glucose / creatinine measurements
    #       using the exact Mediator TAK rules.
    # Note: the source measurement rows themselves (glucose, creatinine)
    # remain in temporal_inputs — they are useful predictors for STraTS /
    # GRU-D as well; only the *event* rows are routed to labels.
    # ---------------------------------------------------------------------
    outcome_source = temporal.loc[
        temporal["ConceptName"].isin(
            source_outcome_set | set(glucose_concepts) | set(creatinine_concepts)
        )
    ].copy()
    outcome_source["numeric_value"] = _to_numeric_value(outcome_source["Value"])

    outcome_event_frames = []

    # (a) pass-through events — rename matched alias to the canonical name
    for outcome, aliases in event_alias_map.items():
        curr = outcome_source.loc[outcome_source["ConceptName"].isin(aliases)].copy()
        if len(curr):
            curr["ConceptName"] = outcome
            outcome_event_frames.append(curr)

    # (b) computed events — Mediator rules over the raw measurements
    glucose_rows = outcome_source.loc[
        outcome_source["ConceptName"].isin(glucose_concepts)
        & outcome_source["numeric_value"].notna()
    ].copy()
    creatinine_rows = outcome_source.loc[
        outcome_source["ConceptName"].isin(creatinine_concepts)
        & outcome_source["numeric_value"].notna()
    ].copy()

    if "HYPERGLYCEMIA_EVENT" in requested and len(glucose_rows):
        outcome_event_frames.append(_hyperglycemia_events(glucose_rows))
    if "HYPOGLYCEMIA_EVENT" in requested and len(glucose_rows):
        outcome_event_frames.append(_hypoglycemia_events(glucose_rows))
    if "SEVERE_HYPERGLYCEMIA_EVENT" in requested and len(glucose_rows):
        outcome_event_frames.append(_severe_hyperglycemia_events(glucose_rows))
    if "SEVERE_HYPOGLYCEMIA_EVENT" in requested and len(glucose_rows):
        outcome_event_frames.append(_severe_hypoglycemia_events(glucose_rows))
    if "KIDNEY_COMPLICATION_EVENT" in requested and len(creatinine_rows):
        outcome_event_frames.append(_kidney_complication_events(creatinine_rows))

    outcome_event_frames = [f for f in outcome_event_frames if len(f) > 0]
    outcome_events = (
        pd.concat(outcome_event_frames, ignore_index=True)
        if outcome_event_frames
        else pd.DataFrame(columns=outcome_source.columns)
    )

    # ---------------------------------------------------------------------
    # Per-patient labels: for each outcome, mark a positive when an event
    # falls in the prediction window, and record the first-hour timestamp.
    # ---------------------------------------------------------------------
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

    # ---------------------------------------------------------------------
    # Length-of-stay target (continuous, hours).
    # Per-patient LoS = time of the RELEASE event from admission, in hours.
    # Patients who died (DEATH event) or have no terminal event in the
    # temporal table get NaN — death does not count toward LoS and the model
    # masks these patients out of the LoS loss and LoS MAE evaluation.
    # ---------------------------------------------------------------------
    los_release_set = {"RELEASE", "RELEASE_EVENT"}
    los_rows = temporal.loc[temporal["ConceptName"].isin(los_release_set)]
    if len(los_rows):
        los_per_patient = (
            los_rows.groupby("PatientId")["minute"].max() / 60.0
        ).rename("length_of_stay_hours").reset_index()
        labels = labels.merge(los_per_patient, on="PatientId", how="left")
    else:
        labels["length_of_stay_hours"] = np.nan

    train_los = labels.loc[
        labels["PatientId"].isin(train_ids), "length_of_stay_hours"
    ].dropna()
    los_mean = float(train_los.mean()) if len(train_los) else 0.0
    los_std  = float(train_los.std())  if len(train_los) > 1 else 1.0
    if not np.isfinite(los_std) or los_std <= 0:
        los_std = 1.0
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

    # ---------------------------------------------------------------------
    # Temporal inputs: original raw measurements within the observation
    # window, minus terminal tokens and minus the alias rows of pass-through
    # outcomes that we are *keeping as prediction targets* (those would leak
    # label information). Two important "demotion" behaviours, matching the
    # INTERVenE side:
    #
    #   1. Pass-through outcomes whose support is below the threshold are
    #      DROPPED from prediction targets but their raw event rows STAY in
    #      the input stream (their alias is no longer in the kept-alias set,
    #      so the exclusion filter does not remove them).
    #   2. Computed outcomes (HYPER / HYPO / SEVERE_* / KIDNEY) whose support
    #      is below the threshold are dropped from targets too; their event
    #      rows (computed earlier) are explicitly appended to the input
    #      stream within the observation window so they survive as regular
    #      tokens.
    #
    # The legacy `concept_support_threshold` token filter is disabled by
    # default (--concept_support_threshold 0.0); the entire vocabulary is
    # kept so the model sees the same token surface as the Mediator-driven
    # pipelines.
    # ---------------------------------------------------------------------
    terminal_aliases = {"RELEASE", "RELEASE_EVENT", args.death_token}
    terminal_aliases.update(["ADMISSION", "RELEASE", "DEATH"])

    # Aliases to *exclude* from inputs — only those for outcomes we are
    # actually using as targets. Dropped pass-through outcomes' aliases stay
    # in inputs as regular tokens.
    kept_pass_through_aliases = {
        alias
        for canonical, aliases in event_alias_map.items()
        if canonical in outcome_names
        for alias in aliases
    }

    temporal_inputs = temporal.loc[
        (temporal["minute"] <= input_minutes)
        & (~temporal["ConceptName"].isin(kept_pass_through_aliases))
        & (~temporal["ConceptName"].isin(terminal_aliases))
    ].copy()

    # Demote computed outcomes that were dropped from targets: their event
    # rows (within the observation window) re-enter the input stream as
    # regular tokens, so the vocabulary is preserved end-to-end.
    dropped_computed = [
        o for o in COMPUTED_OUTCOMES
        if o in args.outcomes and o not in outcome_names
    ]
    if dropped_computed and len(outcome_events):
        demoted = outcome_events.loc[
            outcome_events["ConceptName"].isin(dropped_computed)
            & (outcome_events["minute"] <= input_minutes)
        ].copy()
        if len(demoted):
            shared_cols = [c for c in temporal_inputs.columns if c in demoted.columns]
            temporal_inputs = pd.concat(
                [temporal_inputs, demoted[shared_cols]],
                ignore_index=True,
            )

    # Optional legacy support filter (off by default — see parser comment).
    if args.concept_support_threshold > 0:
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
    else:
        kept_input_concepts = set(temporal_inputs["ConceptName"].unique())
        dropped_input_concepts = []
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
        # Glucose-derived events (Mediator HYPER/HYPO/SEVERE_*):
        "glucose_concepts": glucose_concepts,
        "glucose_hyper_severe_threshold": GLUCOSE_HYPER_SEVERE_THRESHOLD,
        "glucose_hyper_high_ind_min":     GLUCOSE_HYPER_HIGH_IND_MIN,
        "glucose_hypo_severe_threshold":  GLUCOSE_HYPO_SEVERE_THRESHOLD,
        "glucose_hypo_low_ind_min":       GLUCOSE_HYPO_LOW_IND_MIN,
        "glucose_hypo_low_ind_max":       GLUCOSE_HYPO_LOW_IND_MAX,
        # Creatinine-derived event (Mediator KIDNEY_COMPLICATION):
        "creatinine_concepts": creatinine_concepts,
        "creatinine_absolute_threshold": CREATININE_ABSOLUTE_THRESHOLD,
        "creatinine_ratio_threshold":    CREATININE_RATIO_THRESHOLD,
        # Which outcomes were *computed* here (vs. pass-through):
        "computed_outcomes": [o for o in COMPUTED_OUTCOMES if o in outcome_names],
        # Length-of-stay regression target (hours), z-score normalised on train:
        "los_mean": los_mean,
        "los_std":  los_std,
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
    print("Outcomes kept:", outcome_names)
    print("Outcomes dropped from targets (kept in vocab as input tokens):", dropped_outcomes)
    if dropped_input_concepts:
        print("Dropped low-support input concepts:", dropped_input_concepts)
    else:
        print("Input concept filter disabled — all observed concepts kept in vocab.")
    if creatinine_concepts:
        print("Creatinine concepts matched (kept as inputs):", creatinine_concepts)
    if glucose_concepts:
        print("Glucose concepts matched (kept as inputs):", glucose_concepts)


if __name__ == "__main__":
    main()
