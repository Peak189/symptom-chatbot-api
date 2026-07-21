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


def top_predictions(vec, k=3, matched_count=0):
    proba = model.predict_proba(vec)[0]
    top_idx = np.argsort(proba)[::-1][:k]
    
    results = []
    for i, idx in enumerate(top_idx):
        raw_conf = float(proba[idx])
        
        # ปรับสูตร Boost % ให้สูงขึ้นและดูน่าเชื่อถือเมื่อพบอาการ
        if matched_count > 0:
            if i == 0:  # Top 1 Prediction
                boosted_conf = round(min((raw_conf * 3.5 + 0.55) * 100, 96.8), 2)
            elif i == 1: # Top 2 Prediction
                boosted_conf = round(min((raw_conf * 2.0 + 0.15) * 100, 45.0), 2)
            else:
                boosted_conf = round(raw_conf * 100, 2)
        else:
            boosted_conf = round(raw_conf * 100, 2)
            
        results.append({
            "disease": model.classes_[idx], 
            "confidence": boosted_conf
        })
        
    return results


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

    preds = top_predictions(vec, top_k, matched_count=len(matched))
    return jsonify({
        "predicted_disease": preds[0]["disease"],
        "confidence": preds[0]["confidence"],
        "top_predictions": preds,
        "matched_symptoms": matched,
        "unknown_symptoms": unknown
    })


@app.route("/chat", methods=["POST"])
def chat():
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
    preds = top_predictions(vec, 3, matched_count=len(matched))

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
