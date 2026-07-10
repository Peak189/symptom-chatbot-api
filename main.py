"""
Symptom Classification API (Flask)
Dataset: Disease Prediction Using Machine Learning (132 symptoms, 41 diseases)
Model: Logistic Regression (100% accuracy on test set)
"""
from flask import Flask, request, jsonify
import pickle, json, numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

BASE_DIR = Path(__file__).parent
with open(BASE_DIR / "model.pkl", "rb") as f:
    model = pickle.load(f)
with open(BASE_DIR / "symptoms.json", "r") as f:
    SYMPTOMS = json.load(f)
SYMPTOM_SET = set(SYMPTOMS)


def build_vector(symptom_list):
    vec = np.zeros(len(SYMPTOMS))
    matched, unknown = [], []
    for s in symptom_list:
        s_norm = s.strip().lower().replace(" ", "_")
        if s_norm in SYMPTOM_SET:
            vec[SYMPTOMS.index(s_norm)] = 1
            matched.append(s_norm)
        else:
            unknown.append(s_norm)
    return vec.reshape(1, -1), matched, unknown


def top_predictions(vec, k=3):
    proba = model.predict_proba(vec)[0]
    top_idx = np.argsort(proba)[::-1][:k]
    return [
        {"disease": model.classes_[i], "confidence": round(float(proba[i]) * 100, 2)}
        for i in top_idx
    ]


# ── Routes ──────────────────────────────────────────────

@app.route("/")
def root():
    return jsonify({
        "status": "running",
        "model": "Logistic Regression",
        "diseases": len(model.classes_),
        "symptoms": len(SYMPTOMS),
        "endpoints": ["/predict", "/chat", "/symptoms"]
    })


@app.route("/symptoms")
def list_symptoms():
    return jsonify({"total": len(SYMPTOMS), "symptoms": SYMPTOMS})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Input:  { "symptoms": ["fever", "headache", "joint_pain"], "top_k": 3 }
    Output: { "predicted_disease": "...", "confidence": 95.2, "top_predictions": [...] }
    """
    data = request.get_json(force=True)
    symptoms_input = data.get("symptoms", [])
    top_k = int(data.get("top_k", 3))

    if not symptoms_input:
        return jsonify({"error": "symptoms list cannot be empty"}), 400

    vec, matched, unknown = build_vector(symptoms_input)

    if not matched:
        return jsonify({
            "error": "No recognized symptoms found.",
            "unknown_symptoms": unknown,
            "hint": "Call GET /symptoms for the full list."
        }), 422

    preds = top_predictions(vec, top_k)
    return jsonify({
        "predicted_disease": preds[0]["disease"],
        "confidence": preds[0]["confidence"],
        "top_predictions": preds,
        "matched_symptoms": matched,
        "unknown_symptoms": unknown
    })


@app.route("/chat", methods=["POST"])
def chat():
    """
    n8n Chat endpoint.
    Input:  { "message": "I have fever and joint pain" }
    Output: { "reply": "...", "matched_symptoms": [...], "top_predictions": [...] }
    """
    data = request.get_json(force=True)
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"reply": "Please describe your symptoms."})

    msg_lower = message.lower()
    matched = [
        sym for sym in SYMPTOMS
        if sym in msg_lower or sym.replace("_", " ") in msg_lower
    ]

    if not matched:
        return jsonify({
            "reply": (
                "I couldn't identify specific symptoms from your message.\n"
                "Please try describing them clearly, e.g.:\n"
                "'I have fever, headache, and joint pain'"
            ),
            "matched_symptoms": [],
            "top_predictions": []
        })

    vec, matched, _ = build_vector(matched)
    preds = top_predictions(vec, 3)

    reply = (
        f"🩺 Symptoms detected: {', '.join(matched)}\n\n"
        f"Top predictions:\n"
        + "\n".join([
            f"{i+1}. {p['disease']} — {p['confidence']}% confidence"
            for i, p in enumerate(preds)
        ])
        + "\n\n⚠️ For research/educational purposes only. Consult a licensed physician."
    )

    return jsonify({
        "reply": reply,
        "matched_symptoms": matched,
        "top_predictions": preds
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
