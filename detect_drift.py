
import json
import sys

import pandas as pd
from scipy.stats import ks_2samp

REFERENCE_FILE = "student_results.csv"
CURRENT_FILE = "incoming_student_data.csv"

FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score",
]

P_VALUE_THRESHOLD = 0.05

print("Starting student data drift detection...")

reference = pd.read_csv(REFERENCE_FILE)
current = pd.read_csv(CURRENT_FILE)

drift_results = {}

for feature in FEATURES:
    statistic, p_value = ks_2samp(
        reference[feature].dropna(),
        current[feature].dropna(),
    )

    drift_detected = p_value < P_VALUE_THRESHOLD

    drift_results[feature] = {
        "p_value": round(float(p_value), 6),
        "drift_detected": bool(drift_detected),
    }

    print(
        f"{feature}: p-value={p_value:.4f}, "
        f"drift={'DETECTED' if drift_detected else 'NOT DETECTED'}"
    )

any_drift = any(
    result["drift_detected"]
    for result in drift_results.values()
)

report = {
    "drift_detected": any_drift,
    "p_value_threshold": P_VALUE_THRESHOLD,
    "features": drift_results,
}

with open("drift_report.json", "w") as file:
    json.dump(report, file, indent=4)

print("\nDrift report saved to drift_report.json")

if any_drift:
    print("DATA DRIFT DETECTED: Model review or retraining is recommended.")
    sys.exit(2)

print("No significant data drift detected.")
sys.exit(0)
