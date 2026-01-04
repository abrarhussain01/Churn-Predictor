import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_excel("Telco_customer_churn.xlsx")
#print(df.head(5))
# Input columns (YOUR list)
input_cols = [
    'City','Gender','Senior Citizen','Partner','Dependents',
    'Tenure Months','Phone Service','Multiple Lines','Online Security',
    'Online Backup','Device Protection','Tech Support','Streaming TV',
    'Payment Method','Monthly Charges','Total Charges'
]

# Target column
target = 'Churn Label'

# =========================
# Encode target
# =========================
df[target] = df[target].map({'Yes': 1, 'No': 0})

# =========================
# Encode categorical inputs
# =========================
X = df[input_cols].copy()
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = X[col].astype('category').cat.codes

# =========================
# Correlation matrix
# =========================
X[target] = df[target]
corr = X.corr()

# =========================
# Heatmap
# =========================
plt.figure(figsize=(10, 8))
plt.imshow(corr, aspect='auto')
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Heatmap (with Churn Label)")
plt.tight_layout()
plt.show()

# =========================
# REMOVE weakly correlated features
# =========================
threshold = 0.05  # weak correlation threshold

churn_corr = corr[target].drop(target)
weak_cols = churn_corr[abs(churn_corr) < threshold].index.tolist()

# Final selected features
final_features = [col for col in input_cols if col not in weak_cols]

print("Weakly correlated columns REMOVED:")
print(weak_cols)

print("\nFinal input columns USED:")
print(final_features)

X=X[final_features+['Churn Label']]
print(X.head())

#OUTLIERS REMOVAL
X_features = X.drop(columns=['Churn Label'])
y = X['Churn Label']

# Apply IQR outlier removal ONLY on numerical columns
num_cols = X_features.select_dtypes(include=['int64', 'float64']).columns

Q1 = X_features[num_cols].quantile(0.25)
Q3 = X_features[num_cols].quantile(0.75)
IQR = Q3 - Q1


mask = ~((X_features[num_cols] < (Q1 - 1.5 * IQR)) |
         (X_features[num_cols] > (Q3 + 1.5 * IQR))).any(axis=1)


X_clean = X_features[mask]
y_clean = y[mask]


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

X = X_clean
y = y_clean

# Encode categorical features
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = X[col].astype('category').cat.codes

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# XGBoost model (FAST + STABLE)
xgb = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)

xgb.fit(X_train, y_train)

# Predictions
y_pred = xgb.predict(X_test)
print(X_test)
# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("XGBoost Accuracy:", accuracy)

#Model File Creation
import joblib
joblib.dump(xgb,"churn_pred")

import pickle

# Save model
with open("churn_model.pkl", "wb") as f:
    pickle.dump(xgb, f)

# Save feature columns
with open("feature_columns.pkl", "wb") as f:
    pickle.dump(X_train.columns.tolist(), f)
