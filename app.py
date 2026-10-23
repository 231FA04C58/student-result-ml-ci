
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flasgger import Swagger

app = Flask(__name__)

# Swagger documentation configuration
app.config["SWAGGER"] = {
    "title": "Student Result Prediction API",
    "uiversion": 3,
    "openapi": "3.0.0"
}

swagger = Swagger(
    app,
    config={
        "headers": [],
        "specs": [
            {
                "endpoint": "apispec",
                "route": "/apispec.json",
                "rule_filter": lambda rule: True,
                "model_filter": lambda tag: True,
            }
        ],
        "static_url_path": "/flasgger_static",
        "swagger_ui": True,
        "specs_route": "/docs/"
    }
)

MODEL_PATH = Path("student_result_model.pkl")

FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score"
]


def load_model():
    """Load the trained prediction model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "student_result_model.pkl was not found. "
            "Run train_model.py before starting the application."
        )

    return joblib.load(MODEL_PATH)


@app.route("/", methods=["GET"])
def health_check():
    """
    Check service status.
    ---
    tags:
      - Health
    responses:
      200:
        description: Service is running
    """
    return jsonify({
        "status": "ok",
        "service": "student-result-prediction"
    }), 200


@app.route("/health", methods=["GET"])
def health():
    """
    Check API health.
    ---
    tags:
      - Health
    responses:
      200:
        description: API is healthy
    """
    return jsonify({
        "status": "ok",
        "service": "student-result-prediction"
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict student performance.
    ---
    tags:
      - Prediction
    consumes:
      - application/json
    produces:
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
        description: Prediction successful
      400:
        description: Invalid input
      500:
        description: Prediction failed
    """
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Request body must be valid JSON."
        }), 400

    missing = [
        feature for feature in FEATURES
        if feature not in data
    ]

    if missing:
        return jsonify({
            "error": "Missing required fields.",
            "missing_fields": missing
        }), 400

    try:
        values = {
            feature: float(data[feature])
            for feature in FEATURES
        }
    except (TypeError, ValueError):
        return jsonify({
            "error": "All feature values must be numeric."
        }), 400

    if not all(pd.notna(value) for value in values.values()):
        return jsonify({
            "error": "Feature values must be valid numbers."
        }), 400

    try:
        model = load_model()
        input_data = pd.DataFrame([values], columns=FEATURES)
        prediction = model.predict(input_data)[0]

        if hasattr(prediction, "item"):
            prediction = prediction.item()

        return jsonify({
            "service": "student-result-prediction",
            "prediction": prediction,
            "input": values
        }), 200

    except FileNotFoundError as error:
        return jsonify({"error": str(error)}), 500

    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Prediction failed. Check application logs."
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
