OUTCOME_SUPPORT_THRESHOLD = 0.01
CONCEPT_SUPPORT_THRESHOLD = 0.01

DEFAULT_TEMPORAL_CSV = "data/mimic-iv-input-data.csv"
DEFAULT_CONTEXT_CSV = "data/context_data.csv"
DEFAULT_PROCESSED_PATH = "data/processed/user_mimic_iv.pkl"

# ---------------------------------------------------------------------------
# Outcomes — Mediator-aligned Complications.
#
# The five computed events are derived in this ETL from raw measurements using
# the exact rules in Mediator/core/knowledge-base/events/*.xml. Naming and
# Value field match the Mediator (`<name>_EVENT` with Value="True"), so per-
# outcome support can be compared row-for-row between Mediator output and
# this script.
#
# The remaining Complications are pass-through: they are NOT computed here.
# The upstream MIMIC-IV-ETL either emits them as standalone events or does
# not; the regex below dynamically picks up whichever exist (and the
# OUTCOME_SUPPORT_THRESHOLD filter drops the ones that come back empty).
# ---------------------------------------------------------------------------

# Computed by this ETL (measurement-derived per Mediator TAK rules):
COMPUTED_OUTCOMES = [
    "HYPERGLYCEMIA_EVENT",
    "HYPOGLYCEMIA_EVENT",
    "SEVERE_HYPERGLYCEMIA_EVENT",
    "SEVERE_HYPOGLYCEMIA_EVENT",
    "KIDNEY_COMPLICATION_EVENT",
]

# Pass-through from upstream MIMIC-IV-ETL (regex match on event ConceptName):
PASSTHROUGH_OUTCOMES = [
    "DEATH_EVENT",
    "CARDIO-VASCULAR_DISORDER_EVENT",
    "HYPEROSMOLALITY_EVENT",
    "KETOACIDOSIS_EVENT",
    "ACIDOSIS_EVENT",
    "INFECTION_EVENT",
    "DIABETIC_COMA_EVENT",
    "ACUTE_RESPIRATORY_DISORDER_EVENT",
    "OTHER_COMPLICATION_EVENT",
]

OUTCOME_NAMES = COMPUTED_OUTCOMES + PASSTHROUGH_OUTCOMES

EVENT_OUTCOME_REGEX = {
    "DEATH_EVENT":                    r"^DEATH(?:_EVENT)?$",
    "CARDIO-VASCULAR_DISORDER_EVENT": r"^CARDIO-?VASCULAR_DISORDER(?:_EVENT)?$",
    "HYPEROSMOLALITY_EVENT":          r"^HYPEROSMOLALITY(?:_EVENT)?$",
    "KETOACIDOSIS_EVENT":             r"^KETOACIDOSIS(?:_EVENT)?$",
    "ACIDOSIS_EVENT":                 r"^ACIDOSIS(?:_EVENT)?$",
    "INFECTION_EVENT":                r"^INFECTION(?:_EVENT)?$",
    "DIABETIC_COMA_EVENT":            r"^DIABETIC_COMA(?:_EVENT)?$",
    "ACUTE_RESPIRATORY_DISORDER_EVENT": r"^ACUTE_RESPIRATORY_DISORDER(?:_EVENT)?$",
    "OTHER_COMPLICATION_EVENT":       r"^OTHER_COMPLICATION(?:_EVENT)?$",
}

# ---------------------------------------------------------------------------
# Glucose — Mediator HYPERGLYCEMIA / HYPOGLYCEMIA / SEVERE_* rules.
# References:
#   events/HYPERGLYCEMIA.xml          → A1 (glucose ≥ 250) OR S1 (≥ 180 AND prior HIGH_GLUCOSE_IND)
#   events/HYPOGLYCEMIA.xml           → A1 (glucose ≤ 54)  OR S2 (≤ 70 AND prior LOW_GLUCOSE_IND)
#   events/SEVERE_HYPERGLYCEMIA.xml   → glucose ≥ 250
#   events/SEVERE_HYPOGLYCEMIA.xml    → glucose ≤ 54
#   raw-concepts/HIGH_GLUCOSE_IND.xml → glucose ≥ 180
#   raw-concepts/LOW_GLUCOSE_IND.xml  → 20 ≤ glucose ≤ 70
# ---------------------------------------------------------------------------
GLUCOSE_CONCEPT_REGEX = r"^(?:BASE_)?GLUCOSE_MEASURE(?:MENT)?$"
# Mediator GLUCOSE_MEASURE.xml allows only readings inside [20, 1200] mg/dL.
# Any glucose row outside this range is treated as an invalid measurement
# and dropped entirely (from both the input stream and the event-derivation
# source) so the strats ETL sees the same valid-glucose subset Mediator does.
GLUCOSE_VALUE_MIN = 20.0
GLUCOSE_VALUE_MAX = 1200.0
GLUCOSE_HYPER_SEVERE_THRESHOLD = 250.0   # SEVERE_HYPER A1
GLUCOSE_HYPER_HIGH_IND_MIN     = 180.0   # HIGH_GLUCOSE_IND lower bound (recurrent gate)
GLUCOSE_HYPO_SEVERE_THRESHOLD  = 54.0    # SEVERE_HYPO A1
GLUCOSE_HYPO_LOW_IND_MIN       = 20.0    # LOW_GLUCOSE_IND lower bound
GLUCOSE_HYPO_LOW_IND_MAX       = 70.0    # LOW_GLUCOSE_IND upper bound (recurrent gate)

# ---------------------------------------------------------------------------
# Creatinine — Mediator KIDNEY_COMPLICATION rule.
# References:
#   events/KIDNEY_COMPLICATION.xml                       → R1 (ratio ≥ 2.0) OR C1 (absolute ≥ 4.0)
#   parameterized-raw-concepts/CREATININE_REL_SERUM.xml  → ratio = current / first-in-admission
# ---------------------------------------------------------------------------
CREATININE_CONCEPT_REGEX     = r"^(?:BASE_)?CREATININE(?:_SERUM)?(?:_MEASURE(?:MENT)?)?$"
# Mediator CREATININE_SERUM_MEASURE.xml allows only readings inside
# [0.1, 20] mg/dL. Out-of-range or non-numeric creatinine rows are dropped
# entirely (input stream + KIDNEY event-derivation source).
CREATININE_VALUE_MIN          = 0.1
CREATININE_VALUE_MAX          = 20.0
CREATININE_ABSOLUTE_THRESHOLD = 4.0      # AKI Stage 3 (absolute serum creatinine)
CREATININE_RATIO_THRESHOLD    = 2.0      # AKI Stage 2+ (creatinine / baseline)
