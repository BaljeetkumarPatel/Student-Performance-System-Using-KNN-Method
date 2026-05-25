from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_salary_model.pkl"

app = Flask(__name__, static_folder=str(BASE_DIR), static_url_path="")


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


model = load_model()


FEATURE_COLUMNS = [
    "StudentID",
    "Age",
    "StudyTimeWeekly",
    "Absences",
    "Tutoring",
    "ParentalSupport",
    "Extracurricular",
    "Sports",
    "Music",
    "GradeClass",
]


def _parse_payload(payload: Dict[str, Any]) -> List[float]:
    values = []
    for key in FEATURE_COLUMNS:
        if key not in payload:
            raise ValueError(f"Missing required field: {key}")
        try:
            values.append(float(payload[key]))
        except (TypeError, ValueError):
            raise ValueError(f"Invalid numeric value for field: {key}")
    return values


@app.route("/")
def serve_index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)
        if payload is None:
            return jsonify({"error": "Invalid JSON payload"}), 400

        # Accept frontend snake_case keys and map them to training columns.
        payload = {
            "StudentID": payload.get("student_id"),
            "Age": payload.get("age"),
            "StudyTimeWeekly": payload.get("study_time_weekly"),
            "Absences": payload.get("absences"),
            "Tutoring": payload.get("tutoring"),
            "ParentalSupport": payload.get("parental_support"),
            "Extracurricular": payload.get("extracurricular"),
            "Sports": payload.get("sports"),
            "Music": payload.get("music"),
            "GradeClass": payload.get("grade_class"),
        }

        input_values = _parse_payload(payload)

        # Try DataFrame first (works well if model was trained with column names),
        # then fallback to ndarray for generic sklearn models.
        try:
            model_input = pd.DataFrame([input_values], columns=FEATURE_COLUMNS)
            y_pred = model.predict(model_input)
        except Exception:
            model_input = np.array([input_values], dtype=float)
            y_pred = model.predict(model_input)

        prediction = float(y_pred[0])
        return jsonify({"prediction": round(prediction, 2)})
    except ValueError as err:
        return jsonify({"error": str(err)}), 400
    except Exception as err:
        return jsonify({"error": f"Prediction failed: {err}"}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
