
import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flasgger import Swagger

app = Flask(__name__)

app.config["SWAGGER"] = {
    "title": "Student Result Prediction API",
    "uiversion": 3
}

swagger = Swagger(app)

MODEL_PATH = Path("student_result_model.pkl")

FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score"
]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "student_result_model.pkl was not found. "
            "Run the training pipeline first."
        )

    return joblib.load(MODEL_PATH)


@app.get("/")
def health_check():
    """
    Check API status
    ---
    tags:
      - Health
    responses:
      200:
        description: API is running successfully
    """
    return jsonify({
        "status": "ok",
        "service": "student-result-prediction"
    })


@app.post("/predict")
def predict():
    """
    Predict student result
    ---
    tags:
      - Prediction
    consumes:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - attendance
            - internal_marks
            - assignment_marks
            - previous_score
          properties:
            attendance:
              type: number
              example: 85
            internal_marks:
              type: number
              example: 80
            assignment_marks:
              type: number
              example: 90
            previous_score:
              type: number
              example: 75
    responses:
      200:
        description: Prediction returned successfully
      400:
        description: Invalid JSON or missing fields
      500:
        description: Prediction failed or model file is missing
    """
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "A valid JSON request body is required"
        }), 400

    missing_fields = [
        feature for feature in FEATURES
        if feature not in data
    ]

    if missing_fields:
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields
        }), 400

    try:
        sample = pd.DataFrame([{
            feature: data[feature]
            for feature in FEATURES
        }])

        model = load_model()
        prediction_code = int(model.predict(sample)[0])
        prediction = "PASS" if prediction_code == 1 else "FAIL"

        return jsonify({
            "prediction": prediction,
            "prediction_code": prediction_code
        })

    except FileNotFoundError as error:
        return jsonify({"error": str(error)}), 500

    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Prediction failed. Check the server logs."
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
