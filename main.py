from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

app = FastAPI(title="Symptom Classification ML Engine")

# =========================================================
# 1. LOAD DATASET & CLEAN COLUMNS AUTOMATICALLY
# =========================================================
try:
    df = pd.read_csv("Training.csv")
except:
    df = pd.read_csv("dataset.csv")

# ลบคอลัมน์ขยะประเภท Unnamed ที่อาจแถมมา
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# หาชื่อคอลัมน์ที่เป็น Target (ชื่อโรค)
target_col = None
for col in ['prognosis', 'disease', 'Disease', 'Prognosis']:
    if col in df.columns:
        target_col = col
        break

if not target_col:
    # ถ้าหาไม่เจอ ให้เอาคอลัมน์สุดท้ายเป็น target
    target_col = df.columns[-1]

# แยก Features (X) และ Target (y) ให้ชัวร์ 100%
X = df.drop(columns=[target_col])
y = df[target_col]

# แปลงค่าใน X ให้เป็นตัวเลขทั้งหมด (เผื่อมีค่าว่างใส่ 0 แทน)
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

SYMPTOMS_LIST = list(X.columns)

# เทรน 2 โมเดลสำหรับ IEEE Paper
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
    
    # Matching Logic
    input_vector = [0] * len(SYMPTOMS_LIST)
    matched_symptoms = []
    
    for idx, symptom in enumerate(SYMPTOMS_LIST):
        clean_symptom = symptom.replace("_", " ").lower()
        if clean_symptom in user_text:
            input_vector[idx] = 1
            matched_symptoms.append(symptom)
            
    if not matched_symptoms:
        for idx, symptom in enumerate(SYMPTOMS_LIST):
            words = symptom.replace("_", " ").lower().split()
            for word in words:
                if len(word) > 3 and word in user_text:
                    input_vector[idx] = 1
                    matched_symptoms.append(symptom)
                    break

    input_array = np.array([input_vector])
    
    # Prediction Model A: Logistic Regression
    probs_lr = model_lr.predict_proba(input_array)[0]
    top_idx_lr = np.argmax(probs_lr)
    pred_disease_lr = model_lr.classes_[top_idx_lr]
    confidence_lr = round(float(probs_lr[top_idx_lr]) * 100, 2)
    
    # Prediction Model B: Random Forest
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
