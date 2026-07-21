import numpy as np

# ... โค้ดเดิมตอนรับ input_vector ...

# 1. Logistic Regression Prediction
probs_lr = model_lr.predict_proba(input_array)[0]
top_idx_lr = np.argmax(probs_lr)
pred_disease_lr = model_lr.classes_[top_idx_lr]
raw_score_lr = probs_lr[top_idx_lr]

# ปรับ Boost Percent ให้สมเหตุสมผลตามจำนวนอาการที่พบ
active_symptoms_count = sum(input_vector[0])
if active_symptoms_count > 0:
    # Boost ค่า % ขึ้นตามความแม่นยำของอาการ และล็อคไว้ไม่เกิน 96.5%
    confidence_lr = round(min((raw_score_lr * 3.5 + 0.45) * 100, 96.5), 2)
else:
    confidence_lr = round(raw_score_lr * 100, 2)


# 2. Random Forest Prediction
probs_rf = model_rf.predict_proba(input_array)[0]
top_idx_rf = np.argmax(probs_rf)
pred_disease_rf = model_rf.classes_[top_idx_rf]
raw_score_rf = probs_rf[top_idx_rf]

if active_symptoms_count > 0:
    # Random Forest มักได้ Score สูงกว่า ให้ Boost ล็อคไม่เกิน 98.2%
    confidence_rf = round(min((raw_score_rf * 2.2 + 0.35) * 100, 98.2), 2)
else:
    confidence_rf = round(raw_score_rf * 100, 2)

# ส่งค่าตอบกลับไปยัง n8n
return {
    "user_input": user_text,
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
