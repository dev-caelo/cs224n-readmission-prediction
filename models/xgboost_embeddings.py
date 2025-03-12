import numpy as np
import pandas as pd
from collections import defaultdict

from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("../../processed_data/master_df.csv")

# Drop irrelevant columns
columns_to_drop = ["hadm_id", "subject_id", "admittime", "dischtime", "edregtime", "edouttime", "chartdate"]
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# Convert sequence number columns to numeric
df["diagnoses_seq_num"] = pd.to_numeric(df["diagnoses_seq_num"], errors='coerce')
df["procedures_seq_num"] = pd.to_numeric(df["procedures_seq_num"], errors='coerce')

# Handle categorical variables
categorical_columns = ["admission_type", "admission_location", "discharge_location", "insurance", "language", "marital_status", "race", "diagnoses_long_title", "diagnoses_icd", "procedures_long_title", "procedures_icd"]
label_encoders = {}

for col in categorical_columns:
    if col in df.columns:
        df[col] = df[col].astype(str)  # Convert to string before encoding
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Handle missing values
df.fillna(-1, inplace=True)  # Fill missing values with -1

# Define features and target
X = df.drop(columns=["time_until_next_admission"])
y = df["time_until_next_admission"]

# Encode target variable
le_target = LabelEncoder()
y = le_target.fit_transform(y.astype(str))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train XGBoost model
model = xgb.XGBClassifier(objective="multi:softmax", num_class=len(np.unique(y)), eval_metric="mlogloss", use_label_encoder=False)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")