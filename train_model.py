import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib


data = pd.DataFrame({
    "internal_marks": [35, 40, 45, 50, 55, 60, 65, 70, 75, 80],
    "attendance": [60, 65, 70, 75, 78, 80, 82, 85, 88, 90],
    "assignment_score": [40, 45, 50, 55, 60, 65, 70, 75, 80, 85],
    "result": [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
})


X = data[["internal_marks", "attendance", "assignment_score"]]
y = data["result"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = LogisticRegression()
model.fit(X_train, y_train)


predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Model accuracy:", accuracy)


joblib.dump(model, "model.pkl")

print("Model saved successfully.")
