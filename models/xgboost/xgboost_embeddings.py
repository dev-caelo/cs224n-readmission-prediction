import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score

# Model file path
MODEL_PATH = "models/xgboost/save/xgboost_clinical_bert_embeddings_6_class.json"

# Load dataset
df = pd.read_csv("dataset/new_discharge_master.csv")

# Create binary bins (0: <= 30 days, 1: > 30 days)
bins = [-1000, 0, 30, 90, 180, 365, 10000]
labels = [-1, 0, 1, 2, 3, 4]
    
# Apply binning
df["time_until_next_admission_binned"] = pd.cut(
    df["time_until_next_admission"], 
    bins=bins, 
    labels=labels,
    include_lowest=True,
    right=True
)


# Print unique values to debug
print("Unique values in time_until_next_admission_binned:", df['time_until_next_admission_binned'].unique())

# # Handle missing values
df.fillna(-1, inplace=True)  # Fill missing values with -1


# Check for invalid values and fix them
if -1 in df['time_until_next_admission_binned'].unique():
    print("Found -1 values in target variable, fixing...")
    # Option 1: Remove rows with -1
    # df = df[df['time_until_next_admission_binned'] >= 0]
    
    # Option 2: Map -1 to appropriate class (choosing class 1 - no readmission within 30 days)
    df['time_until_next_admission_binned'] = df['time_until_next_admission_binned'].replace(-1, 1)
    
    print("After fixing, unique values:", df['time_until_next_admission_binned'].unique())


df['time_until_next_admission_binned'] = pd.to_numeric(df['time_until_next_admission_binned'])

# Drop irrelevant columns
columns_to_drop = ["hadm_id", "subject_id", "admittime", "dischtime", "edregtime", "edouttime", "chartdate", "note_id",
                  "storetime", 'anchor_year', 'anchor_year_group', "time_until_next_admission", "text", "charttime", "dod",
                  "admission_type", "admission_location", "discharge_location", "insurance", "language", "marital_status", "race", 
                     "diagnoses_long_title", "diagnoses_icd", "procedures_long_title", "procedures_icd", "note_type", "note_seq", 'gender',
                      'anchor_age', "diagnoses_seq_num", "procedures_seq_num", "length_of_stay"]
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# # Convert sequence number columns to numeric
# df["diagnoses_seq_num"] = pd.to_numeric(df["diagnoses_seq_num"], errors='coerce')
# df["procedures_seq_num"] = pd.to_numeric(df["procedures_seq_num"], errors='coerce')

# Extract embeddings as a numpy array
embeddings_array = np.load("ehr_output/clinical_bert_embeddings.npy")

# Reduce dimensions to a manageable number
pca = PCA(n_components=20)
reduced_embeddings = pca.fit_transform(embeddings_array)

# Create dataframe with reduced dimensions
embedding_df = pd.DataFrame(
    reduced_embeddings,
    columns=[f'embedding_pca_{i}' for i in range(reduced_embeddings.shape[1])]
)

# Combine with original dataframe
df = pd.concat([df, embedding_df], axis=1)

# # Handle categorical variables
# categorical_columns = ["admission_type", "admission_location", "discharge_location", "insurance", "language", "marital_status", "race", 
#                      "diagnoses_long_title", "diagnoses_icd", "procedures_long_title", "procedures_icd", "note_type", "note_seq", 'gender',
#                       'anchor_age']

# label_encoders = {}

# for col in categorical_columns:
#     if col in df.columns:
#         df[col] = df[col].astype(str)  # Convert to string before encoding
#         le = LabelEncoder()
#         df[col] = le.fit_transform(df[col])
#         label_encoders[col] = le

# Define features and target
X = df.drop(columns=["time_until_next_admission_binned"])
y = df["time_until_next_admission_binned"]

# For binary classification, we don't need to encode the target variable further as it's already 0 and 1
# But we make sure it's in the right format
y = y.astype(int)

# Class distribution analysis
class_counts = y.value_counts()
print("\nClass Distribution:")
print(class_counts)
print(f"Class 0 (<=30 days): {class_counts.get(0, 0)} ({class_counts.get(0, 0)/len(y)*100:.2f}%)")
print(f"Class 1 (>30 days): {class_counts.get(1, 0)} ({class_counts.get(1, 0)/len(y)*100:.2f}%)")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Check if model exists and load it, otherwise train a new one
if os.path.exists(MODEL_PATH):
    print(f"Loading existing model from {MODEL_PATH}")
    
    best_params = {
            'learning_rate': 0.1,
            'max_depth': 100,
            'min_child_weight': 1,
            'subsample': 1.0,
            'colsample_bytree': 0.8,
            'gamma': 1,
            'n_estimators': 300
        }

    # For binary classification, use binary:logistic objective
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        **best_params
    )
    
    model.load_model(MODEL_PATH)
    
    # Verify the loaded model works with our data
    try:
        # Try to make predictions with a small subset to verify compatibility
        model.predict(X_test[:5])
        print("Model successfully loaded and verified.")
    except Exception as e:
        print(f"Error using loaded model: {e}")
        print("Training a new model instead.")
        train_new_model = True
    else:
        train_new_model = False
else:
    print(f"No existing model found at {MODEL_PATH}. Training a new model.")
    train_new_model = True

# Train a new model if needed
if train_new_model:
    print("Training new XGBoost binary classification model...")
    model = xgb.XGBClassifier(
        objective="binary:logistic", 
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Save the newly trained model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print(f"Saving model to {MODEL_PATH}")
    model.save_model(MODEL_PATH)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

# For multiclass, we need to use a different approach for ROC AUC
# We'll use one-vs-rest approach
roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')

# Print metrics
print(f"Model Accuracy: {accuracy:.4f}")
# print(f"ROC AUC (OVR): {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Create visualizations
plt.figure(figsize=(15, 15))

# 1. Confusion Matrix - show a subset if too large
num_classes = min(10, len(np.unique(y_test)))
plt.subplot(2, 2, 1)
conf_matrix_subset = conf_matrix[:num_classes, :num_classes]
sns.heatmap(conf_matrix_subset, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix (First {num_classes} classes)')

# 2. Feature importance
plt.subplot(2, 2, 2)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False).head(15)
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Top 15 Feature Importance')

# 3. Prediction Distribution
plt.subplot(2, 2, 3)
pred_dist = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})
sns.histplot(data=pred_dist, x='Actual', kde=True, color='blue', label='Actual', alpha=0.5)
sns.histplot(data=pred_dist, x='Predicted', kde=True, color='red', label='Predicted', alpha=0.5)
plt.legend()
plt.title('Distribution of Actual vs Predicted Classes')

# 4. Classification Metrics
plt.subplot(2, 2, 4)
# Get the average metrics
avg_metrics = report['weighted avg']
metrics = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Value': [accuracy, avg_metrics['precision'], avg_metrics['recall'], avg_metrics['f1-score']]
})
sns.barplot(x='Metric', y='Value', data=metrics)
plt.ylim(0, 1)
plt.title('Classification Metrics (Weighted Average)')

plt.tight_layout()
plt.savefig("xgboost_clinical_bert_embeddings_6_bin.png", dpi=600)
plt.show()