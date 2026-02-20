# ============================================
# PORTBRIDGE CREDIT RISK MODEL
# XGBoost + SHAP (Google Colab)
# ============================================

# Install required libraries
!pip install xgboost shap imbalanced-learn --quiet

# Import libraries
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# --------------------------------------------
# 1. Load Dataset
# --------------------------------------------
df = pd.read_csv("portbridge_credit_risk_simulation.csv")

print("Dataset Preview:")
print(df.head())

# --------------------------------------------
# 2. Preprocessing
# --------------------------------------------

# One-hot encode Industry_Type
df = pd.get_dummies(df, columns=["Industry_Type"], drop_first=True)

# Define features and target
X = df.drop(["Client_ID", "Default_Flag"], axis=1)
y = df["Default_Flag"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Handle Class Imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------
# 3. Train XGBoost Model
# --------------------------------------------

model = xgb.XGBClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=4,
    scale_pos_weight=1,
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train_scaled, y_train_resampled)

# --------------------------------------------
# 4. Model Evaluation
# --------------------------------------------

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:,1]

roc_score = roc_auc_score(y_test, y_prob)

print("\nROC-AUC Score:", round(roc_score, 4))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --------------------------------------------
# 5. SHAP Explainability
# --------------------------------------------

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_scaled)

print("\nGenerating SHAP Summary Plot...")
shap.summary_plot(shap_values, X_test, show=True)

# --------------------------------------------
# 6. Feature Importance Plot
# --------------------------------------------

xgb.plot_importance(model, max_num_features=10)
plt.title("Top 10 Feature Importance")
plt.show()
