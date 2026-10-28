
from pathlib import Path
import os
import logging
import time
import uuid

import joblib
import pandas as pd
from flask import Flask, jsonify, request, g
from flasgger import Swagger

# -------------------- APP CONFIGURATION --------------------

app = Flask(__name__)

# -------------------- LOGGING CONFIGURATION --------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

app.logger.setLevel(logging.INFO)
app.logger.info("Student Result Prediction API logging initialized")

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

MODEL_PATH = Path(__file__).resolve().parent / "student_result_model.pkl"

FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score",
]

# -------------------- REQUEST LOGGING --------------------

@app.before_request
def log_request_start():
    """Assign a request ID and record the request start time."""
    g.request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    g.start_time = time.perf_counter()

    app.logger.info(
        "Request started | id=%s method=%s path=%s",
        g.request_id,
        request.method,
        request.path,
    )


@app.after_request
def log_request_end(response):
    """Record request duration, response status, and request ID."""
    request_id = getattr(g, "request_id", "unknown")
    start_time = getattr(g, "start_time", None)

    if start_time is not None:
        duration_ms = (time.perf_counter() - start_time) * 1000

        app.logger.info(
            "Request completed | id=%s method=%s path=%s status=%s duration_ms=%.2f",
            request_id,
            request.method,
            request.path,
            response.status_code,
            duration_ms,
        )

    response.headers["X-Request-ID"] = request_id
    return response


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

            html = html.replace("</head>", custom_css + "</head>")
            response.set_data(html)

    return response


# -------------------- LOAD MODEL --------------------

def load_model():
    """Load the trained model and log the operation."""
    if not MODEL_PATH.exists():
        app.logger.error("Model file is missing: %s", MODEL_PATH.name)
        raise FileNotFoundError(
            "student_result_model.pkl is missing. "
            "Run train_model.py before starting the application."
        )

    app.logger.info("Loading prediction model")
    model = joblib.load(MODEL_PATH)
    app.logger.info("Prediction model loaded successfully")
    return model


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

    # Validate JSON body
    if not isinstance(data, dict):
        app.logger.warning(
            "Invalid request body | id=%s reason=invalid_json",
            g.request_id,
        )
        return jsonify({
            "error": "Request body must be valid JSON."
        }), 400

    # Validate required fields
    missing = [
        feature
        for feature in FEATURES
        if feature not in data
    ]

    if missing:
        app.logger.warning(
            "Missing required fields | id=%s fields=%s",
            g.request_id,
            ",".join(missing),
        )
        return jsonify({
            "error": "Missing required fields.",
            "missing_fields": missing,
        }), 400

    # Validate numeric values
    try:
        values = {
            feature: float(data[feature])
            for feature in FEATURES
        }
    except (TypeError, ValueError):
        app.logger.warning(
            "Non-numeric feature value | id=%s",
            g.request_id,
        )
        return jsonify({
            "error": "All feature values must be numeric."
        }), 400

    # Reject NaN and infinity
    if not all(
        pd.notna(value) and abs(value) != float("inf")
        for value in values.values()
    ):
        app.logger.warning(
            "Invalid numeric feature value | id=%s",
            g.request_id,
        )
        return jsonify({
            "error": "Feature values must be valid numbers."
        }), 400

    # Run prediction
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

        app.logger.info(
            "Prediction successful | id=%s",
            g.request_id,
        )

        return jsonify({
            "service": "student-result-prediction",
            "prediction": prediction,
            "input": values,
        }), 200

    except FileNotFoundError as error:
        app.logger.error(
            "Prediction unavailable | id=%s error=%s",
            g.request_id,
            str(error),
        )
        return jsonify({
            "error": str(error)
        }), 500

    except Exception:
        app.logger.exception(
            "Prediction failed | id=%s",
            g.request_id,
        )
        return jsonify({
            "error": "Prediction failed. Check application logs."
        }), 500


# -------------------- RUN APPLICATION --------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.logger.info("Starting API on port %s", port)
    app.run(host="0.0.0.0", port=port)
