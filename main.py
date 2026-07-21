from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

app = FastAPI(title="Symptom Classification ML Engine")

# =========================================================
# 1. LOAD DATASET & TRAIN MODELS AUTOMATICALLY
# =========================================================
# โหลด dataset.csv ที่อยู่ใน GitHub Repository เดียวกัน
try:
    df = pd.read_csv("dataset.csv")
except Exception as e:
    # เผื่อไฟล์ชื่อ Training.csv
    try:
        df = pd.read_csv("Training.csv")
    except:
        df = pd.read_csv("dataset.csv")

# ดึงชื่อ Feature (132 อาการ) และ Target (Prognosis/Disease)
X = df.iloc[:, :-1]  # ทุก column ยกเว้น column สุดท้าย
y = df.iloc[:, -1]   # column สุดท้าย (ชื่อโรค)

SYMPTOMS_LIST = list(X.columns)

# เทรน 2 โมเดลพร้อมกันสำหรับ Paper IEEE
model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X, y)

model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X, y)

# =========================================================
# 2. REQUEST/RESPONSE SCHEMA
# =========================================================
class SymptomInput(BaseModel):
    text: str

# =========================================================
# 3. PREDICTION ENDPOINT
# =========================================================
@app.post("/predict")
def predict_symptom(data: SymptomInput):
    user_text = data.text.lower()
    
    # Text Processing & Mapping -> Binary Vector
    input_vector = [0] * len(SYMPTOMS_LIST)
    matched_symptoms = []
    
    for idx, symptom in enumerate(SYMPTOMS_LIST):
        # แปลง underscore เป็น space เพื่อแมตช์คำ
        clean_symptom = symptom.replace("_", " ").lower()
        if clean_symptom in user_text:
            input_vector[idx] = 1
            matched_symptoms.append(symptom)
            
    if not matched_symptoms:
        # Fallback กรณีสแกนไม่เจอคำตรงๆ
        for idx, symptom in enumerate(SYMPTOMS_LIST):
            words = symptom.replace("_", " ").lower().split()
            for word in words:
                if len(word) > 3 and word in user_text:
                    input_vector[idx] = 1
                    matched_symptoms.append(symptom)
                    break

    input_array = np.array([input_vector])
    
    # --- Model A: Logistic Regression ---
    probs_lr = model_lr.predict_proba(input_array)[0]
    top_idx_lr = np.argmax(probs_lr)
    pred_disease_lr = model_lr.classes_[top_idx_lr]
    confidence_lr = round(float(probs_lr[top_idx_lr]) * 100, 2)
    
    # --- Model B: Random Forest ---
    probs_rf = model_rf.predict_proba(input_array)[0]
    top_idx_rf = np.argmax(probs_rf)
    pred_disease_rf = model_rf.classes_[top_idx_rf]
    confidence_rf = round(float(probs_rf[top_idx_rf]) * 100, 2)

    return {
        "user_input": data.text,
        "detected_symptoms": list(set(matched_symptoms)),
        "logistic_regression": {
            "disease": pred_disease_lr,
            "confidence": f"{confidence_lr}%"
        },
        "random_forest": {
            "disease": pred_disease_rf,
            "confidence": f"{confidence_rf}%"
        }
    }

@app.get("/")
def root():
    return {"status": "ML Engine Running", "symptoms_count": len(SYMPTOMS_LIST)}
