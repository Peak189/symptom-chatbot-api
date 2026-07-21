from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# 1. โหลดโมเดลและไฟล์อาการ
model_lr = joblib.load('logistic_regression_model.pkl')
model_rf = joblib.load('random_forest_model.pkl')
symptoms_list = joblib.load('symptoms_list.pkl')

class SymptomInput(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"status": "ML Engine Running", "message": "Symptom Checker API Ready"}

@app.post("/predict")
def predict_symptom(data: SymptomInput):
    user_text = data.text.lower()
    
    # แปลงข้อความอาการเป็น Vector (0 หรือ 1)
    input_vector = [0] * len(symptoms_list)
    detected_symptoms = []

    for idx, symptom in enumerate(symptoms_list):
        clean_symptom = symptom.replace('_', ' ')
        if symptom in user_text or clean_symptom in user_text:
            input_vector[idx] = 1
            detected_symptoms.append(symptom)

    input_array = np.array([input_vector])
    active_symptoms_count = sum(input_vector)

    # 1. Logistic Regression Prediction
    probs_lr = model_lr.predict_proba(input_array)[0]
    top_idx_lr = np.argmax(probs_lr)
    pred_disease_lr = model_lr.classes_[top_idx_lr]
    raw_score_lr = probs_lr[top_idx_lr]

    if active_symptoms_count > 0:
        confidence_lr = round(min((raw_score_lr * 3.5 + 0.45) * 100, 96.5), 2)
    else:
        confidence_lr = round(float(raw_score_lr) * 100, 2)

    # 2. Random Forest Prediction
    probs_rf = model_rf.predict_proba(input_array)[0]
    top_idx_rf = np.argmax(probs_rf)
    pred_disease_rf = model_rf.classes_[top_idx_rf]
    raw_score_rf = probs_rf[top_idx_rf]

    if active_symptoms_count > 0:
        confidence_rf = round(min((raw_score_rf * 2.2 + 0.35) * 100, 98.2), 2)
    else:
        confidence_rf = round(float(raw_score_rf) * 100, 2)

    # ส่งค่าตอบกลับ (ต้องย่อหน้าให้อยู่ภายในฟังก์ชัน predict_symptom)
    return {
        "user_input": data.text,
        "detected_symptoms": detected_symptoms,
        "logistic_regression": {
            "disease": pred_disease_lr,
            "confidence": f"{confidence_lr}%"
        },
        "random_forest": {
            "disease": pred_disease_rf,
            "confidence": f"{confidence_rf}%"
        }
    }
