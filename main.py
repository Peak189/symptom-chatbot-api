import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression
import numpy as np

app = FastAPI(title="Symptom Classification API for n8n")

# บังคับหาตำแหน่งโฟลเดอร์ปัจจุบันที่โค้ดนี้รันอยู่ ป้องกันปัญหาหาไฟล์ไม่เจอ บน Render
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, 'Training.csv')

# 1. โหลดข้อมูลและเทรนโมเดลทันทีเมื่อเปิดเซิร์ฟเวอร์
try:
    if not os.path.exists(csv_path):
        print(f"❌ File not found at {csv_path}. Trying local fallback...")
        csv_path = 'Training.csv'

    train_data = pd.read_csv(csv_path)
    if 'Unnamed: 133' in train_data.columns:
        train_data = train_data.drop('Unnamed: 133', axis=1)
        
    X_train = train_data.drop('prognosis', axis=1)
    y_train = train_data['prognosis']
    
    FEATURE_COLUMNS = list(X_train.columns)
    
    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X_train, y_train)
    print("🧠 Model trained successfully with 132 features!")
except Exception as e:
    print(f"❌ Error training model: {str(e)}")
    model = None

class SymptomInput(BaseModel):
    extracted_symptoms: list[str]

@app.get("/")
def read_root():
    return {"status": "Symptom API is running successfully"}

@app.post("/predict")
def predict_disease(payload: SymptomInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not trained or initialized properly.")
    
    input_vector = np.zeros(len(FEATURE_COLUMNS))
    matched_count = 0
    
    for symptom in payload.extracted_symptoms:
        formatted_symptom = symptom.lower().strip().replace(" ", "_")
        if formatted_symptom in FEATURE_COLUMNS:
            idx = FEATURE_COLUMNS.index(formatted_symptom)
            input_vector[idx] = 1
            matched_count += 1
            
    if matched_count == 0 and len(payload.extracted_symptoms) > 0:
        return {
            "predicted_disease": "Unknown / Insufficient specific symptoms",
            "confidence_percentage": 0.0,
            "matched_symptoms": []
        }

    input_vector = input_vector.reshape(1, -1)
    prediction = model.predict(input_vector)[0]
    
    probabilities = model.predict_proba(input_vector)[0]
    class_idx = list(model.classes_).index(prediction)
    confidence = probabilities[class_idx] * 100

    return {
        "predicted_disease": str(prediction).strip(),
        "confidence_percentage": round(confidence, 2),
        "matched_symptoms": [f for f in payload.extracted_symptoms if f.lower().strip().replace(" ", "_") in FEATURE_COLUMNS]
    }
