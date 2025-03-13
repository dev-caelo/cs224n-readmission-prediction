# import numpy as np
# import pandas as pd
# import os
# from sklearn.model_selection import train_test_split
# import xgboost as xgb
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import roc_auc_score

# # Model file path
# MODEL_PATH = "models/xgboost_clinical_bert_all.json"

# # Load dataset
# df = pd.read_csv("dataset/new_discharge_master.csv")

# # Create time bins
# bins = [0, 30, 10000]
# labels = ["Admitted", "Not Readmitted"]

# # bins = [0] + list(range(7, 366, 7)) + [np.inf]
# # labels = [0] + list(range(7, 366, 7))
    
# # Apply binning
# df["time_until_next_admission_binned"] = pd.cut(
#     df["time_until_next_admission"], 
#     bins=bins, 
#     labels=labels,
#     include_lowest=True,
#     right=True
# )

# df['time_until_next_admission_binned'] = pd.to_numeric(df['time_until_next_admission_binned'])

# # Drop irrelevant columns
# columns_to_drop = ["hadm_id", "subject_id", "admittime", "dischtime", "edregtime", "edouttime", "chartdate", "note_id",
#                   "storetime", 'anchor_year', 'anchor_year_group', "time_until_next_admission", "text", "charttime", "dod"]
# df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# # Convert sequence number columns to numeric
# df["diagnoses_seq_num"] = pd.to_numeric(df["diagnoses_seq_num"], errors='coerce')
# df["procedures_seq_num"] = pd.to_numeric(df["procedures_seq_num"], errors='coerce')

# # Extract embeddings as a numpy array
# embeddings_array = np.load("ehr_output/clinical_bert_embeddings.npy")

# # Reduce dimensions to a manageable number
# pca = PCA(n_components=20)
# reduced_embeddings = pca.fit_transform(embeddings_array)

# # Create dataframe with reduced dimensions
# embedding_df = pd.DataFrame(
#     reduced_embeddings,
#     columns=[f'embedding_pca_{i}' for i in range(reduced_embeddings.shape[1])]
# )

# # Combine with original dataframe
# df = pd.concat([df, embedding_df], axis=1)

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

# # Handle missing values
# df.fillna(-1, inplace=True)  # Fill missing values with -1

# # Define features and target
# X = df.drop(columns=["time_until_next_admission_binned"])
# y = df["time_until_next_admission_binned"]

# # Encode target variable
# le_target = LabelEncoder()
# y = le_target.fit_transform(y.astype(str))

# # Get class names for later use
# class_names = le_target.classes_

# # Split dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Check if model exists and load it, otherwise train a new one
# if os.path.exists(MODEL_PATH):
#     print(f"Loading existing model from {MODEL_PATH}")
    
#     best_params = {
#             'learning_rate': 0.01,
#             'max_depth': 3,
#             'min_child_weight': 1,
#             'subsample': 1.0,
#             'colsample_bytree': 0.8,
#             'gamma': 0.2,
#             'n_estimators': 100
#         }

#     model = xgb.XGBClassifier(
#         objective="multi:softmax",
#         num_class=len(np.unique(y_train)),
#         eval_metric="mlogloss",
#         use_label_encoder=False,
#         random_state=42,
#         **best_params
#     )
    
#     model.load_model(MODEL_PATH)
    
#     # Verify the loaded model works with our data
#     try:
#         # Try to make predictions with a small subset to verify compatibility
#         model.predict(X_test[:5])
#         print("Model successfully loaded and verified.")
#     except Exception as e:
#         print(f"Error using loaded model: {e}")
#         print("Training a new model instead.")
#         train_new_model = True
#     else:
#         train_new_model = False
# else:
#     print(f"No existing model found at {MODEL_PATH}. Training a new model.")
#     train_new_model = True

# # Train a new model if needed
# if train_new_model:
#     print("Training new XGBoost model...")
#     model = xgb.XGBClassifier(
#         objective="multi:softmax", 
#         num_class=len(np.unique(y)), 
#         eval_metric="mlogloss",
#         use_label_encoder=False
#     )
#     model.fit(X_train, y_train)
    
#     # Save the newly trained model
#     print(f"Saving model to {MODEL_PATH}")
#     model.save_model(MODEL_PATH)

# # Make predictions
# y_pred = model.predict(X_test)
# y_pred_proba = model.predict_proba(X_test)

# # Evaluate model
# accuracy = accuracy_score(y_test, y_pred)
# conf_matrix = confusion_matrix(y_test, y_pred)
# report = classification_report(y_test, y_pred, output_dict=True)

# # For multiclass, we need to use a different approach for ROC AUC
# # We'll use one-vs-rest approach
# roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')

# # Print metrics
# print(f"Model Accuracy: {accuracy:.4f}")
# # print(f"ROC AUC (OVR): {roc_auc:.4f}")
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

# # Create visualizations
# plt.figure(figsize=(15, 15))

# # 1. Confusion Matrix - show a subset if too large
# num_classes = min(10, len(np.unique(y_test)))
# plt.subplot(2, 2, 1)
# conf_matrix_subset = conf_matrix[:num_classes, :num_classes]
# sns.heatmap(conf_matrix_subset, annot=True, fmt='d', cmap='Blues')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title(f'Confusion Matrix (First {num_classes} classes)')

# # 2. Feature importance
# plt.subplot(2, 2, 2)
# feature_importance = pd.DataFrame({
#     'Feature': X.columns,
#     'Importance': model.feature_importances_
# }).sort_values('Importance', ascending=False).head(15)
# sns.barplot(x='Importance', y='Feature', data=feature_importance)
# plt.title('Top 15 Feature Importance')

# # 3. Prediction Distribution
# plt.subplot(2, 2, 3)
# pred_dist = pd.DataFrame({
#     'Actual': y_test,
#     'Predicted': y_pred
# })
# sns.histplot(data=pred_dist, x='Actual', kde=True, color='blue', label='Actual', alpha=0.5)
# sns.histplot(data=pred_dist, x='Predicted', kde=True, color='red', label='Predicted', alpha=0.5)
# plt.legend()
# plt.title('Distribution of Actual vs Predicted Classes')

# # 4. Classification Metrics
# plt.subplot(2, 2, 4)
# # Get the average metrics
# avg_metrics = report['weighted avg']
# metrics = pd.DataFrame({
#     'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
#     'Value': [accuracy, avg_metrics['precision'], avg_metrics['recall'], avg_metrics['f1-score']]
# })
# sns.barplot(x='Metric', y='Value', data=metrics)
# plt.ylim(0, 1)
# plt.title('Classification Metrics (Weighted Average)')

# plt.tight_layout()
# plt.savefig("xgboost_clinical_bert_all.png", dpi=600)
# plt.show()

# import numpy as np
# import pandas as pd
# import os
# from sklearn.model_selection import train_test_split, GridSearchCV
# import xgboost as xgb
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import roc_auc_score
# import joblib

# # Model file paths
# MODEL_DIR = "models"
# MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_readmission_embeddings.json")
# PARAMS_PATH = os.path.join(MODEL_DIR, "xgboost_best_params.joblib")

# # Create model directory if it doesn't exist
# os.makedirs(MODEL_DIR, exist_ok=True)

# # Load dataset
# df = pd.read_csv("dataset/new_discharge_master.csv")

# # Create time bins
# bins = [0] + list(range(7, 366, 7)) + [np.inf]
# labels = [0] + list(range(7, 366, 7))
    
# # Apply binning
# df["time_until_next_admission_binned"] = pd.cut(
#     df["time_until_next_admission"], 
#     bins=bins, 
#     labels=labels,
#     include_lowest=True,
#     right=True
# )

# df['time_until_next_admission_binned'] = pd.to_numeric(df['time_until_next_admission_binned'])

# # Drop irrelevant columns
# columns_to_drop = ["hadm_id", "subject_id", "admittime", "dischtime", "edregtime", "edouttime", "chartdate", "note_id",
#                   "storetime", 'anchor_year', 'anchor_year_group', "time_until_next_admission", "text", "charttime", "dod",
#                   "admission_type", "admission_location", "discharge_location", "insurance", "language", "marital_status", "race", 
#                       "diagnoses_long_title", "diagnoses_icd", "procedures_long_title", "procedures_icd", "note_type", "note_seq", 'gender',
#                       'anchor_age', "diagnoses_seq_num", "procedures_seq_num", "length_of_stay"]
# df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# # Extract embeddings as a numpy array
# embeddings_array = np.load("ehr_output/discharge_note_embeddings.npy")

# # Reduce dimensions to a manageable number
# pca = PCA(n_components=25)
# reduced_embeddings = pca.fit_transform(embeddings_array)

# # Create dataframe with reduced dimensions
# embedding_df = pd.DataFrame(
#     reduced_embeddings,
#     columns=[f'embedding_pca_{i}' for i in range(reduced_embeddings.shape[1])]
# )

# # Combine with original dataframe
# df = pd.concat([df, embedding_df], axis=1)

# # # Handle categorical variables
# # categorical_columns = ["admission_type", "admission_location", "discharge_location", "insurance", "language", "marital_status", "race", 
# #                      "diagnoses_long_title", "diagnoses_icd", "procedures_long_title", "procedures_icd", "note_type", "note_seq", 'gender',
# #                       'anchor_age']

# # label_encoders = {}

# # for col in categorical_columns:
# #     if col in df.columns:
# #         df[col] = df[col].astype(str)  # Convert to string before encoding
# #         le = LabelEncoder()
# #         df[col] = le.fit_transform(df[col])
# #         label_encoders[col] = le

# # Handle missing values
# df.fillna(-1, inplace=True)  # Fill missing values with -1

# # Define features and target
# X = df.drop(columns=["time_until_next_admission_binned"])
# y = df["time_until_next_admission_binned"]

# # Encode target variable
# le_target = LabelEncoder()
# y = le_target.fit_transform(y.astype(str))

# # Get class names for later use
# class_names = le_target.classes_

# # Split dataset - using a validation set for hyperparameter tuning
# X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
# X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# # Check if we should perform hyperparameter tuning
# perform_hyperparameter_search = True  # Set to False to skip hyperparameter search if needed

# # Define plotting function for final evaluation
# def plot_model_evaluation(y_test, y_pred, y_pred_proba, model, X, conf_matrix, accuracy, roc_auc, report):
#     plt.figure(figsize=(15, 15))

#     # 1. Confusion Matrix - show a subset if too large
#     num_classes = min(10, len(np.unique(y_test)))
#     plt.subplot(2, 2, 1)
#     conf_matrix_subset = conf_matrix[:num_classes, :num_classes]
#     sns.heatmap(conf_matrix_subset, annot=True, fmt='d', cmap='Blues')
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.title(f'Confusion Matrix (First {num_classes} classes)')

#     # 2. Feature importance
#     plt.subplot(2, 2, 2)
#     feature_importance = pd.DataFrame({
#         'Feature': X.columns,
#         'Importance': model.feature_importances_
#     }).sort_values('Importance', ascending=False).head(15)
#     sns.barplot(x='Importance', y='Feature', data=feature_importance)
#     plt.title('Top 15 Feature Importance')

#     # 3. Prediction Distribution
#     plt.subplot(2, 2, 3)
#     pred_dist = pd.DataFrame({
#         'Actual': y_test,
#         'Predicted': y_pred
#     })
#     sns.histplot(data=pred_dist, x='Actual', kde=True, color='blue', label='Actual', alpha=0.5)
#     sns.histplot(data=pred_dist, x='Predicted', kde=True, color='red', label='Predicted', alpha=0.5)
#     plt.legend()
#     plt.title('Distribution of Actual vs Predicted Classes')

#     # 4. Classification Metrics
#     plt.subplot(2, 2, 4)
#     # Get the average metrics
#     avg_metrics = report['weighted avg']
#     metrics = pd.DataFrame({
#         'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
#         'Value': [accuracy, avg_metrics['precision'], avg_metrics['recall'], avg_metrics['f1-score']]
#     })
#     sns.barplot(x='Metric', y='Value', data=metrics)
#     plt.ylim(0, 1)
#     plt.title('Classification Metrics (Weighted Average)')

#     plt.tight_layout()
#     plt.savefig("xgboost_embeddings_metrics.png", dpi=600)
#     plt.show()

# # Hyperparameter tuning
# if perform_hyperparameter_search and (not os.path.exists(PARAMS_PATH) or not os.path.exists(MODEL_PATH)):
#     print("Performing hyperparameter search...")
    
#     # Define the parameter grid to search
#     param_grid = {
#         'learning_rate': [0.01, 0.05, 0.1],
#         'max_depth': [3, 5, 7],
#         'min_child_weight': [1, 3, 5],
#         'subsample': [0.6, 0.8, 1.0],
#         'colsample_bytree': [0.6, 0.8, 1.0],
#         'gamma': [0, 0.1, 0.2],
#         'n_estimators': [100, 200]
#     }
    
#     # Create a base model
#     xgb_model = xgb.XGBClassifier(
#         objective="multi:softmax",
#         num_class=len(np.unique(y_train)),
#         eval_metric="mlogloss",
#         use_label_encoder=False,
#         random_state=42
#     )
    
#     # Set up GridSearchCV
#     # Using a smaller subset for faster hyperparameter tuning if the dataset is large
#     sample_size = min(10000, len(X_train))
#     X_train_sample = X_train.iloc[:sample_size]
#     y_train_sample = y_train[:sample_size]
    
#     grid_search = GridSearchCV(
#         estimator=xgb_model,
#         param_grid=param_grid,
#         scoring='accuracy',
#         cv=3,
#         verbose=2,
#         n_jobs=-1
#     )
    
#     # Fit the grid search to the data
#     grid_search.fit(X_train_sample, y_train_sample)
    
#     # Get the best parameters
#     best_params = grid_search.best_params_
#     print(f"Best parameters found: {best_params}")
    
#     # Save the best parameters
#     joblib.dump(best_params, PARAMS_PATH)
    
#     # Train the final model with the best parameters
#     print("Training final model with best parameters...")
#     final_model = xgb.XGBClassifier(
#         objective="multi:softmax",
#         num_class=len(np.unique(y_train)),
#         eval_metric="mlogloss",
#         use_label_encoder=False,
#         random_state=42,
#         **best_params
#     )
    
#     # Add validation set for early stopping
#     eval_set = [(X_train, y_train), (X_val, y_val)]
#     final_model.fit(
#         X_train, 
#         y_train, 
#         eval_set=eval_set,
#         eval_metric="mlogloss",
#         early_stopping_rounds=10,
#         verbose=True
#     )
    
#     # Save the model
#     final_model.save_model(MODEL_PATH)
    
#     print(f"Final model saved to {MODEL_PATH}")
    
# else:
#     # Load existing best parameters if available
#     if os.path.exists(PARAMS_PATH) and perform_hyperparameter_search:
#         print(f"Loading best parameters from {PARAMS_PATH}")
#         best_params = joblib.load(PARAMS_PATH)
#         print(f"Best parameters: {best_params}")
#     else:
#         # Default parameters if no hyperparameter search was done
#         best_params = {
#             'learning_rate': 0.1,
#             'max_depth': 5,
#             'min_child_weight': 1,
#             'subsample': 0.8,
#             'colsample_bytree': 0.8,
#             'gamma': 0,
#             'n_estimators': 100
#         }
    
#     # Check if model exists and load it, otherwise train a new one
#     if os.path.exists(MODEL_PATH):
#         print(f"Loading existing model from {MODEL_PATH}")
#         final_model = xgb.XGBClassifier()
#         final_model.load_model(MODEL_PATH)
        
#         # Verify the loaded model works with our data
#         try:
#             # Try to make predictions with a small subset to verify compatibility
#             final_model.predict(X_test[:5])
#             print("Model successfully loaded and verified.")
#         except Exception as e:
#             print(f"Error using loaded model: {e}")
#             print("Training a new model instead.")
#             train_new_model = True
#         else:
#             train_new_model = False
#     else:
#         print(f"No existing model found at {MODEL_PATH}. Training a new model.")
#         train_new_model = True

#     # Train a new model if needed
#     if train_new_model:
#         print("Training new XGBoost model with best parameters...")
#         final_model = xgb.XGBClassifier(
#             objective="multi:softmax",
#             num_class=len(np.unique(y_train)),
#             eval_metric="mlogloss",
#             use_label_encoder=False,
#             random_state=42,
#             **best_params
#         )
        
#         # Add validation set for early stopping
#         eval_set = [(X_train, y_train), (X_val, y_val)]
#         final_model.fit(
#             X_train, 
#             y_train, 
#             eval_set=eval_set,
#             eval_metric="mlogloss",
#             early_stopping_rounds=10,
#             verbose=True
#         )
        
#         # Save the newly trained model
#         print(f"Saving model to {MODEL_PATH}")
#         final_model.save_model(MODEL_PATH)

# # Make predictions on the test set
# y_pred = final_model.predict(X_test)
# y_pred_proba = final_model.predict_proba(X_test)

# # Evaluate model
# accuracy = accuracy_score(y_test, y_pred)
# conf_matrix = confusion_matrix(y_test, y_pred)
# report = classification_report(y_test, y_pred, output_dict=True)

# # For multiclass, we need to use a different approach for ROC AUC
# # We'll use one-vs-rest approach
# roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')

# # Print metrics
# print(f"Model Accuracy: {accuracy:.4f}")
# print(f"ROC AUC (OVR): {roc_auc:.4f}")
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

# # Create visualizations
# plot_model_evaluation(y_test, y_pred, y_pred_proba, final_model, X, conf_matrix, accuracy, roc_auc, report)

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
MODEL_PATH = "models/xgboost_clinical_bert_embeddings_6_class.json"

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
            'learning_rate': 0.01,
            'max_depth': 3,
            'min_child_weight': 1,
            'subsample': 1.0,
            'colsample_bytree': 0.8,
            'gamma': 0.2,
            'n_estimators': 100
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


# # Make predictions
# y_pred = model.predict(X_test)
# y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of class 1

# # Evaluate model
# accuracy = accuracy_score(y_test, y_pred)
# conf_matrix = confusion_matrix(y_test, y_pred)
# report = classification_report(y_test, y_pred, output_dict=True)

# # For binary classification, standard ROC AUC
# roc_auc = roc_auc_score(y_test, y_pred_proba)
# precision = report['1']['precision']
# recall = report['1']['recall']
# f1 = report['1']['f1-score']
# avg_precision = average_precision_score(y_test, y_pred_proba)

# # Print metrics
# print(f"\nModel Accuracy: {accuracy:.4f}")
# print(f"ROC AUC: {roc_auc:.4f}")
# print(f"Average Precision Score: {avg_precision:.4f}")
# print("\nBinary Classification Report:")
# print(classification_report(y_test, y_pred))
# print("\nConfusion Matrix:")
# print(conf_matrix)

# # Calculate metrics for ROC curve
# fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
# # Calculate metrics for PR curve
# precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)

# # Create visualizations for binary classification
# plt.figure(figsize=(20, 15))

# # 1. Confusion Matrix
# plt.subplot(2, 3, 1)
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
#             xticklabels=['≤30 days', '>30 days'], 
#             yticklabels=['≤30 days', '>30 days'])
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')

# # 2. Feature importance
# plt.subplot(2, 3, 2)
# feature_importance = pd.DataFrame({
#     'Feature': X.columns,
#     'Importance': model.feature_importances_
# }).sort_values('Importance', ascending=False).head(15)
# sns.barplot(x='Importance', y='Feature', data=feature_importance)
# plt.title('Top 15 Feature Importance')

# # 3. ROC Curve
# plt.subplot(2, 3, 3)
# plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')
# plt.legend()

# # 4. Precision-Recall Curve
# plt.subplot(2, 3, 4)
# plt.plot(recall_curve, precision_curve, label=f'PR Curve (AP = {avg_precision:.3f})')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall Curve')
# plt.legend()

# # 5. Classification Metrics Bar Chart
# plt.subplot(2, 3, 5)
# metrics = pd.DataFrame({
#     'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC'],
#     'Value': [accuracy, precision, recall, f1, roc_auc]
# })
# sns.barplot(x='Metric', y='Value', data=metrics)
# plt.ylim(0, 1)
# plt.title('Binary Classification Metrics')

# # 6. Probability Distribution
# plt.subplot(2, 3, 6)
# for i, label in enumerate(['≤30 days', '>30 days']):
#     sns.kdeplot(y_pred_proba[y_test == i], label=f'True {label}')
# plt.xlabel('Predicted Probability of >30 days')
# plt.ylabel('Density')
# plt.title('Prediction Probability Distribution')
# plt.legend()

# plt.tight_layout()
# plt.savefig("xgboost_clinical_bert_all_six_class.png", dpi=600)
# plt.show()

# # Calculate additional metrics to understand model performance
# tn, fp, fn, tp = conf_matrix.ravel()
# sensitivity = tp / (tp + fn)
# specificity = tn / (tn + fp)
# ppv = tp / (tp + fp)  # Positive Predictive Value
# npv = tn / (tn + fn)  # Negative Predictive Value

# print("\nAdditional Binary Classification Metrics:")
# print(f"Sensitivity/Recall: {sensitivity:.4f}")
# print(f"Specificity: {specificity:.4f}")
# print(f"Positive Predictive Value (Precision): {ppv:.4f}")
# print(f"Negative Predictive Value: {npv:.4f}")