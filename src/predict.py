from pathlib import Path
import joblib
import pandas as pd

bundle = joblib.load(Path(__file__).resolve().parents[1] / "models" / "stroke_model.joblib")
patient = pd.DataFrame([{
    "gender": "Male", "age": 67, "hypertension": 0, "heart_disease": 1,
    "ever_married": "Yes", "work_type": "Private", "Residence_type": "Urban",
    "avg_glucose_level": 228.69, "bmi": 36.6, "smoking_status": "formerly smoked",
}])
p = bundle["pipeline"].predict_proba(patient)[0, 1]
print(f"Stroke risk score: {p:.2%}  ->  {'HIGH RISK' if p >= bundle['threshold'] else 'low risk'}")
