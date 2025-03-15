"""
parse.py

This file contains code that we created to manually score TP/FP/TN/FN metrics on our 6-bin models, due to complications
of non-binary classification with our existing packages.
"""

import re
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def parse_prediction_file(file_path):
    """
    Parse the prediction file and extract predictions and labels.
    
    Args:
        file_path (str): Path to the prediction file
        
    Returns:
        tuple: (predictions, labels) as numpy arrays
    """
    predictions = []
    labels = []
    
    with open(file_path, 'r') as f:
        for line in f:
            # Skip separator lines and timestamp lines
            if line.startswith('--') or line.startswith('Fri'):
                continue
                
            # Extract prediction and label using regex
            match = re.search(r'Prediction: (\d+) \|\| Label: (\d+)', line)
            if match:
                pred = int(match.group(1))
                label = int(match.group(2))
                
                predictions.append(pred)
                labels.append(label)
    
    return np.array(predictions), np.array(labels)

def calculate_metrics(predictions, labels):
    """
    Calculate evaluation metrics.
    
    Args:
        predictions (numpy.ndarray): Array of predicted labels
        labels (numpy.ndarray): Array of true labels
        
    Returns:
        dict: Dictionary containing calculated metrics
    """
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    
    # For weighted metrics
    precision_weighted = precision_score(labels, predictions, average='weighted', zero_division=0)
    recall_weighted = recall_score(labels, predictions, average='weighted', zero_division=0)
    f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)
    
    # Return metrics as dictionary
    return {
        'Accuracy': accuracy,
        'Precision (Weighted)': precision_weighted,
        'Recall (Weighted)': recall_weighted,
        'F1 Score (Weighted)': f1_weighted
    }

def main():
    # File path
    file_path = 'log/clinical_bert_train_pred_vs_label.txt'
    
    try:
        # Parse file
        predictions, labels = parse_prediction_file(file_path)
        
        # Check if we have data
        if len(predictions) == 0 or len(labels) == 0:
            print("No prediction-label pairs found in the file.")
            return
            
        print(f"Found {len(predictions)} prediction-label pairs.")
        
        # Calculate class distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        print("\nClass Distribution:")
        for label, count in zip(unique_labels, counts):
            print(f"Class {label}: {count} samples ({count/len(labels)*100:.2f}%)")
        
        # Calculate metrics
        metrics = calculate_metrics(predictions, labels)
        
        # Print results
        print("\n===== Evaluation Metrics =====")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")
            
        # Print confusion information
        print("\nPrediction Distribution:")
        unique_preds, pred_counts = np.unique(predictions, return_counts=True)
        for pred, count in zip(unique_preds, pred_counts):
            print(f"Predicted as class {pred}: {count} samples ({count/len(predictions)*100:.2f}%)")
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()