import torch
from torch import nn
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score, accuracy_score, classification_report, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# evaluation function
def Evaluate(model, test_loader, loss_func, cls, device, epoch, path,
             language_model, log):
    model.eval()
    step = 0
    report_loss = 0
    all_predictions = []
    all_probabilities = []
    all_labels = []
    top_2_correct = 0
    top_3_correct = 0
    total_samples = 0

    with torch.no_grad():
        for text, mask, age, label in tqdm(test_loader):
            text = text.to(device)
            mask = mask.to(device)
            age = age.to(device)
            #others = others.to(device)
            label = label.to(device)

            # Forward pass
            logits = model(text, mask, age)
            loss = loss_func(logits, label)
            report_loss += loss.item()
            step += 1

            # Get probabilities and predictions
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            
            # Calculate top-k accuracies
            batch_size = label.size(0)
            total_samples += batch_size
            
            # Top-2 accuracy
            _, top2_indices = torch.topk(logits, 2, dim=1)
            top2_correct = torch.sum(top2_indices == label.view(-1, 1)).item()
            top_2_correct += top2_correct
            
            # Top-3 accuracy

            # _, top3_indices = torch.topk(logits, 3, dim=1)
            # top3_correct = torch.sum(top3_indices == label.view(-1, 1)).item()
            # top_3_correct += top3_correct

            # Store results for later analysis
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

        print('Val Loss: {:.6f}'.format(report_loss / step))

    # Convert lists to numpy arrays for evaluation
    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)
    y_pred_proba = np.vstack(all_probabilities)
    
    # For binary classification, we need the probability of the positive class
    if y_pred_proba.shape[1] == 2:  # Binary classification
        y_pred_proba_positive = y_pred_proba[:, 1]
    else:  # Multi-class - use one-vs-rest approach
        # For metrics like ROC, we need to convert to one-hot encoding
        y_true_onehot = np.eye(cls)[y_true]
        
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Calculate F1 scores
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    f1_unweighted = f1_score(y_true, y_pred, average='macro')
    
    # Detailed classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Print metrics
    print(f"\nModel Accuracy: {accuracy:.4f}")
    print(f"F1 Score (weighted): {f1_weighted:.4f}")
    print(f"F1 Score (unweighted): {f1_unweighted:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    # Create visualizations
    plt.figure(figsize=(20, 15))
    
    # 1. Confusion Matrix
    plt.subplot(2, 2, 1)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=cls, yticklabels=cls)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    # For binary classification, add ROC and PR curves
    if cls == 2:
        # Calculate binary metrics
        roc_auc = roc_auc_score(y_true, y_pred_proba_positive)
        avg_precision = average_precision_score(y_true, y_pred_proba_positive)
        
        # Calculate ROC curve data
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba_positive)
        
        # Calculate PR curve data
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba_positive)
        
        # 2. ROC Curve
        plt.subplot(2, 2, 2)
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        
        # 3. Precision-Recall Curve
        plt.subplot(2, 2, 3)
        plt.plot(recall_curve, precision_curve, label=f'PR Curve (AP = {avg_precision:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        
        # 4. Classification Metrics Bar Chart
        plt.subplot(2, 2, 4)
        
        # Get binary classification metrics
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        metrics = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC'],
            'Value': [accuracy, precision, recall, f1, roc_auc]
        })
        sns.barplot(x='Metric', y='Value', data=metrics)
        plt.ylim(0, 1)
        plt.title('Binary Classification Metrics')
        
        # Calculate additional metrics for binary classification
        tn, fp, fn, tp = conf_matrix.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        
        print("\nAdditional Binary Classification Metrics:")
        print(f"Sensitivity/Recall: {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"Positive Predictive Value (Precision): {ppv:.4f}")
        print(f"Negative Predictive Value: {npv:.4f}")
    else:
        # For multiclass, show different visualizations
        
        # 2. Class Distribution
        plt.subplot(2, 2, 2)
        class_counts = np.bincount(y_true)
        plt.bar(range(len(cls)), class_counts)
        plt.xticks(range(len(cls)), cls, rotation=45)
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Class Distribution')
        
        # 3. Per-Class Metrics
        plt.subplot(2, 2, 3)
        class_metrics = []
        for i, class_name in enumerate(cls):
            if str(i) in report:
                class_metrics.append({
                    'Class': class_name,
                    'Precision': report[str(i)]['precision'],
                    'Recall': report[str(i)]['recall'],
                    'F1-Score': report[str(i)]['f1-score']
                })
        
        class_metrics_df = pd.DataFrame(class_metrics)
        class_metrics_melted = pd.melt(
            class_metrics_df, 
            id_vars=['Class'], 
            value_vars=['Precision', 'Recall', 'F1-Score'],
            var_name='Metric', 
            value_name='Value'
        )
        
        sns.barplot(x='Class', y='Value', hue='Metric', data=class_metrics_melted)
        plt.ylim(0, 1)
        plt.title('Per-Class Performance Metrics')
        plt.legend(loc='lower right')
        
        # 4. Prediction Probabilities Heatmap
        plt.subplot(2, 2, 4)
        
        # Select a subset of samples for visualization if there are too many
        max_samples = min(50, len(y_true))
        sample_indices = np.random.choice(len(y_true), max_samples, replace=False)
        
        sample_probs = y_pred_proba[sample_indices]
        sns.heatmap(sample_probs, cmap='Blues', 
                    xticklabels=cls, 
                    yticklabels=False)
        plt.xlabel('Class')
        plt.ylabel('Sample')
        plt.title('Prediction Probabilities')
    
    plt.tight_layout()
    plt.savefig(f"{path}/{language_model}_{cls}_model_evaluation_epoch_{epoch}.png", dpi=600)
    
    # Save results to log file
    with open(f'log/{language_model}_{log}', 'a') as f:
        f.write('\n')
        f.write(f'Epoch: {epoch + 1}\n')
        f.write(f'  Acc: {accuracy * 100:.2f} %\n')
        f.write(f'  Top2 Acc: {top_2_correct / total_samples * 100:.2f} %\n')
        f.write(f'  Top3 Acc: {top_3_correct / total_samples * 100:.2f} %\n')
        f.write(f'  F1 Weighted: {f1_weighted * 100:.2f} %\n')
        f.write(f'  F1 Unweighted: {f1_unweighted * 100:.2f} %\n')
        f.write(f'  Precision: {precision:.2f} %\n')
        f.write(f'  Recall: {recall:.2f} %\n')
        f.write(f'  roc_auc: {roc_auc * 100:.2f} %\n')
        
        if cls == 2:
            f.write(f'  Sensitivity: {sensitivity:.4f}\n')
            f.write(f'  Specificity: {specificity:.4f}\n')
            f.write(f'  PPV: {ppv:.4f}\n')
            f.write(f'  NPV: {npv:.4f}\n')
    
    return report_loss / step, accuracy