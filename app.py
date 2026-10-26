
from pathlib import Path
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flasgger import Swagger

app = Flask(__name__)

# -------------------- SWAGGER CONFIGURATION --------------------

app.config["SWAGGER"] = {
    "title": "Student Result Prediction API",
    "uiversion": 3,
    "description": (
        "Predict student results using attendance, internal marks, "
        "assignment marks, and previous scores."
    ),
}

swagger = Swagger(
    app,
    config={
        "headers": [],
        "specs": [
            {
                "endpoint": "apispec_1",
                "route": "/apispec_1.json",
                "rule_filter": lambda rule: True,
                "model_filter": lambda tag: True,
            }
        ],
        "static_url_path": "/flasgger_static",
        "swagger_ui": True,
        "specs_route": "/docs/",
    },
)

# -------------------- MODEL CONFIGURATION --------------------

MODEL_PATH = (
    Path(__file__).resolve().parent / "student_result_model.pkl"
)

FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score",
]

# -------------------- SWAGGER UI CUSTOMIZATION --------------------

@app.after_request
def customize_swagger(response):
    if request.path.rstrip("/") == "/docs":
        if (
            response.content_type
            and "text/html" in response.content_type
        ):
            html = response.get_data(as_text=True)

            custom_css = """
            <style>
                .swagger-ui .topbar {
                    display: none !important;
                }

                .swagger-ui .info {
                    margin: 30px 0;
                }

                .swagger-ui .info .title {
                    font-size: 32px;
                    font-weight: 700;
                }

                .swagger-ui .opblock-tag {
                    font-size: 20px;
                    font-weight: 600;
                }

                .swagger-ui .opblock {
                    border-radius: 4px;
                }
            </style>
            """

            html = html.replace(
                "</head>",
                custom_css + "</head>",
            )
            response.set_data(html)

    return response

# -------------------- LOAD MODEL --------------------

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "student_result_model.pkl is missing. "
            "Run train_model.py before starting the application."
        )

    return joblib.load(MODEL_PATH)

# -------------------- HOME / SERVICE STATUS --------------------

@app.route("/", methods=["GET"])
def health_check():
    """
    Service status.
    ---
    tags:
      - Health
    responses:
      200:
        description: Service is running
    """
    return jsonify({
        "status": "ok",
        "service": "student-result-prediction",
    }), 200

# -------------------- HEALTH CHECK --------------------

@app.route("/health", methods=["GET"])
def health():
    """
    Health check.
    ---
    tags:
      - Health
    responses:
      200:
        description: API is healthy
    """
    return jsonify({
        "status": "ok",
        "service": "student-result-prediction",
    }), 200

# -------------------- PREDICT STUDENT RESULT --------------------

@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict student result (PASS/FAIL).
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
        description: Invalid request
      500:
        description: Prediction failed
    """
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Request body must be valid JSON."
        }), 400

    missing = [
        feature
        for feature in FEATURES
        if feature not in data
    ]

    if missing:
        return jsonify({
            "error": "Missing required fields.",
            "missing_fields": missing,
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

        input_data = pd.DataFrame(
            [values],
            columns=FEATURES,
        )

        prediction = model.predict(input_data)[0]

        if hasattr(prediction, "item"):
            prediction = prediction.item()

        if prediction == 1:
            prediction = "PASS"
        elif prediction == 0:
            prediction = "FAIL"

        return jsonify({
            "service": "student-result-prediction",
            "prediction": prediction,
            "input": values,
        }), 200

    except FileNotFoundError as error:
        return jsonify({
            "error": str(error)
        }), 500

    except Exception:
        app.logger.exception("Prediction failed")

        return jsonify({
            "error": "Prediction failed. Check application logs."
        }), 500

# -------------------- RUN APPLICATION --------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
