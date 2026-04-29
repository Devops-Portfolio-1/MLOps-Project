# train.py
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
	accuracy_score,
	precision_score,
	recall_score,
	f1_score,
	roc_auc_score,
	confusion_matrix,
)
import joblib

# Load dataset from a working source (Kaggle/hosted)
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(url)

print("✅ Columns:", df.columns.tolist())  # Debug print

# Prepare data
X = df[["Pregnancies", "Glucose", "BloodPressure", "BMI", "Age"]]
y = df["Outcome"]

# Model and cross-validation setup
model = RandomForestClassifier(random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Get fold-wise metrics
scoring = {
	"accuracy": "accuracy",
	"precision": "precision",
	"recall": "recall",
	"f1": "f1",
	"roc_auc": "roc_auc",
}
cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring)

# Get out-of-fold predictions for aggregate metrics
y_pred = cross_val_predict(model, X, y, cv=cv, method="predict")
y_proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]

print("\n=== Cross-Validation Performance (5-Fold) ===")
for metric_name in scoring:
	scores = cv_results[f"test_{metric_name}"]
	print(f"{metric_name.upper():>9}: mean={scores.mean():.4f}, std={scores.std():.4f}")

print("\n=== Aggregate Out-of-Fold Metrics ===")
print(f"Accuracy : {accuracy_score(y, y_pred):.4f}")
print(f"Precision: {precision_score(y, y_pred):.4f}")
print(f"Recall   : {recall_score(y, y_pred):.4f}")
print(f"F1-score : {f1_score(y, y_pred):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y, y_proba):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y, y_pred))

# Train final model on full dataset for deployment
model.fit(X, y)

# Save
joblib.dump(model, "diabetes_model.pkl")
print("✅ Model saved as diabetes_model.pkl")
