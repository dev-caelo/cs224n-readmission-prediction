import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import json
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from models.bert_finetune import ClinicalBERTClassifier
from dataset.torch_diag import DiagTorch

def train_clinical_bert(model, train_dataloader, val_dataloader, epochs=3, lr=2e-5, device="cuda"):
    """
    Train the ClinicalBERTClassifier model.
    
    Args:
        model: The ClinicalBERTClassifier model
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        epochs (int): Number of training epochs
        lr (float): Learning rate
        device (str): Device to use for training ('cuda' or 'cpu')
        
    Returns:
        model: Trained model
        history: Training history
    """
    # Move model to device
    model = model.to(device)
    
    # Initialize optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': []
    }
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        # Progress bar for training
        train_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in train_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Compute loss
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            
            # Backward pass
            loss.backward()
            
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            scheduler.step()
            
            # Update progress bar
            train_bar.set_postfix({'loss': loss.item()})
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_dataloader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation
        val_loss, val_accuracy, val_f1 = evaluate(model, val_dataloader, criterion, device)
        
        # Update history
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['val_f1'].append(val_f1)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Accuracy: {val_accuracy:.4f}, "
              f"Val F1: {val_f1:.4f}")
    
    return model, history

def evaluate(model, dataloader, criterion, device):
    """
    Evaluate the model on the given dataloader.
    
    Args:
        model: The ClinicalBERTClassifier model
        dataloader: DataLoader for evaluation
        criterion: Loss function
        device: Device to use for evaluation
        
    Returns:
        avg_loss (float): Average loss
        accuracy (float): Accuracy score
        f1 (float): Macro F1 score
    """
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Compute loss
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            # Get predictions
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            # Append to lists
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate average validation loss
    avg_loss = val_loss / len(dataloader)
    
    # Calculate accuracy and F1 score
    accuracy = accuracy_score(all_labels, all_preds)
    _, _, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    
    return avg_loss, accuracy, f1

def predict(model, texts, tokenizer, device="cuda", max_length=512):
    """
    Make predictions on new texts.
    
    Args:
        model: Trained ClinicalBERTClassifier model
        texts (list): List of clinical texts to classify
        tokenizer: HuggingFace tokenizer
        device (str): Device to use for inference
        max_length (int): Maximum sequence length
        
    Returns:
        predictions (list): List of predicted class indices
        probabilities (np.ndarray): Array of class probabilities
    """
    model.eval()
    model = model.to(device)
    predictions = []
    probabilities = []
    
    with torch.no_grad():
        for text in texts:
            # Tokenize
            encoding = tokenizer(
                text,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Get probabilities
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            
            # Get prediction
            pred = torch.argmax(outputs, dim=1).cpu().numpy()[0]
            
            predictions.append(pred)
            probabilities.append(probs[0])
    
    return predictions, np.array(probabilities)

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
    
    # Define hyperparameters
    num_classes = 2   # Follows our LACE bucketing of <=30 days or not.
    max_length = 512  
    batch_size = 16
    epochs = 10
    learning_rate = 2e-5
    
    # Create model
    model = ClinicalBERTClassifier(num_classes=num_classes, dropout_rate=0.1)
    
    # Data path for training, validation, and testing
    train_data_path = "dataset/new_discharge_master.csv"
    
    # Define which columns to include
    columns_to_include = ['text', 'anchor_age', 'gender', 'race', 'hadm_id',
                          'admission_location', 'discharge_location', 'insurance', 'admission_type',
                          'note_type', 'language', 'marital_status', 'diagnoses_long_title',
                          'diagnoses_icd', 'procedures_long_title', 'procedures_icd', 'note_seq']
    
    # Create datasets
    train_dataset = DiagTorch(
        data_path=train_data_path,
        label='time_until_next_admission',
        subset='train',
        tokenizer=tokenizer,
        max_length=max_length,
        text_column='text',
        total_bins=num_classes,
        columns_to_include=columns_to_include,  # Use either this or columns_to_exclude
        # columns_to_exclude=columns_to_exclude,
        separator=" | "  # Use a clear separator between fields
    )
    
    val_dataset = DiagTorch(
        data_path=train_data_path,  # Same file but different subset
        label='time_until_next_admission',
        subset='val',
        tokenizer=tokenizer,
        max_length=max_length,
        text_column='text',
        total_bins=num_classes,
        columns_to_include=columns_to_include,  # Use either this or columns_to_exclude
        # columns_to_exclude=columns_to_exclude,
        separator=" | "  # Use a clear separator between fields
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True
    )
    
    # Train model
    trained_model, history = train_clinical_bert(
        model, 
        train_dataloader, 
        val_dataloader,
        epochs=epochs,
        lr=learning_rate,
        device=device
    )
    
    # Save the trained model
    torch.save(trained_model.state_dict(), "clinical_bert_readmission_classifier.pt")
    
    # Save training history
    with open('training_history.json', 'w') as f:
        json.dump({k: [float(val) for val in v] for k, v in history.items()}, f)
    
    # Optional: Make predictions on test set
    test_dataset = DiagTorch(
        data_path=train_data_path,  # Same file but test subset
        label='time_until_next_admission',
        subset='test',
        tokenizer=tokenizer,
        max_length=max_length,
        text_column='text',
        total_bins=num_classes,
        columns_to_include=columns_to_include,  # Use either this or columns_to_exclude
        # columns_to_exclude=columns_to_exclude,
        separator=" | "  # Use a clear separator between fields
    )
    
    # Test dataloader
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True
    )
    
    # Evaluate on test set
    test_criterion = torch.nn.FocalLoss()
    test_loss, test_accuracy, test_f1 = evaluate(
        trained_model, 
        test_dataloader, 
        test_criterion, 
        device
    )
    
    print(f"\nTest Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    
    # Demonstration of inference with model
    demonstrate_inference(trained_model, tokenizer, device, max_length)

def demonstrate_inference(model, tokenizer, device, max_length):
    # Readmission category mapping for interpretation
    readmission_categories = {
        0: "0-30 days",
        1: "No readmission",
    }
    
    # Sample prediction with single example - using format that matches concatenated columns
    sample_text = "text: Patient presents with shortness of breath and chest pain. History of COPD. | anchor_age: 65 | gender: Male | race: White | hospital_id: 123 | diagnosis: COPD exacerbation"
    
    preds, probs = predict(
        model, 
        [sample_text], 
        tokenizer, 
        device,
        max_length=max_length
    )
    
    print(f"\nSample Prediction:")
    print(f"Text: {sample_text}")
    print(f"Predicted class: {preds[0]} ({readmission_categories[preds[0]]})")
    print(f"Class probabilities: {probs[0]}")
    
    # Simple example with just a few columns
    simple_text = "text: Patient shows signs of pneumonia | anchor_age: 45 | gender: Female"
    simple_preds, simple_probs = predict(model, [simple_text], tokenizer, device, max_length=max_length)
    print(f"\nSimple Example Prediction:")
    print(f"Text: {simple_text}")
    print(f"Predicted class: {simple_preds[0]} ({readmission_categories[simple_preds[0]]})")
    print(f"Class probabilities: {simple_probs[0]}")

if __name__ == "__main__":
    main()