
import json
import subprocess
import sys

print("Checking drift report...")

with open("drift_report.json", "r") as file:
    report = json.load(file)

if report["drift_detected"]:
    print("Data drift detected!")
    print("Starting model retraining...")

    result = subprocess.run(
        [sys.executable, "train_model.py"],
        check=False
    )

    if result.returncode == 0:
        print("Model retraining completed successfully.")
        print("Updated model: student_result_model.pkl")
    else:
        print("Model retraining failed.")
        sys.exit(result.returncode)
else:
    print("No data drift detected.")
    print("Retraining is not required.")
