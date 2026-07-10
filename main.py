import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression
import numpy as np

app = FastAPI(title="Symptom Classification API for n8n")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, 'Training.csv')

try:
    if not os.path.exists(csv_path):
        csv_path = 'Training.csv'

    train_data = pd.read_csv(csv_path)
    if 'Unnamed: 133' in train_data.columns:
        train_data = train_data.drop('Unnamed: 133', axis=1)
        
    X_train = train_data.drop('prognosis', axis=1)
    y_train = train_data['prognosis']
    
    FEATURE_COLUMNS = list(X_train.columns)
    
    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X_train, y_train)
    print("🧠 Model trained successfully!")
except Exception as e:
    print(f"❌ Error training model: {str(e)}")
    model = None

# ปรับให้รับข้อความดิบยาว ๆ (Raw Text) จากแชทได้โดยตรง ไม่ต้องสับคำมาจาก n8n
class SymptomInput(BaseModel):
    extracted_symptoms: list[str] = []
    text: str = "" # รองรับการส่งข้อความตรง ๆ

@app.get("/")
def read_root():
    return {"status": "Symptom API is running successfully"}

@app.post("/predict")
def predict_disease(payload: SymptomInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not initialized.")
    
    # รวมข้อความทั้งหมดที่อาจจะส่งมา ไม่ว่าจะส่งมาเป็น list หรือเป็นข้อความยาว
    user_text = payload.text.lower().strip()
    if not user_text and payload.extracted_symptoms:
        user_text = " ".join(payload.extracted_symptoms).lower().strip()
    
    input_vector = np.zeros(len(FEATURE_COLUMNS))
    matched_symptoms = []
    
    # ไล่เช็กทีละอาการจาก 132 อาการในตาราง ว่าคำไหนโผล่มาในประโยคที่ผู้ใช้พิมพ์บ้าง
    for symptom in FEATURE_COLUMNS:
        # เปลี่ยนตัว _ เป็นช่องว่างเพื่อให้ค้นหาในประโยคภาษาอังกฤษธรรมชาติได้ง่ายขึ้น
        readable_symptom = symptom.replace("_", " ")
        
        if readable_symptom in user_text or symptom in user_text:
            idx = FEATURE_COLUMNS.index(symptom)
            input_vector[idx] = 1
            matched_symptoms.append(symptom)
            
    if len(matched_symptoms) == 0:
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
        "matched_symptoms": matched_symptoms
    }
