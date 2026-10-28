
import pandas as pd

# Load the original student dataset
reference = pd.read_csv("student_results.csv")

# Create incoming data with a noticeable change in student attendance
incoming = reference[
    ["attendance", "internal_marks", "assignment_marks", "previous_score"]
].copy()

# Simulate a change: incoming attendance is much lower
incoming["attendance"] = (incoming["attendance"] - 25).clip(lower=0)

# Save the simulated incoming student records
incoming.to_csv("incoming_student_data.csv", index=False)

print("Incoming student data created successfully.")
print("Number of records:", len(incoming))
print(incoming.head())
