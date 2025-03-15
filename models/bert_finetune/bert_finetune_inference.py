import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from models.bert_finetune import ClinicalBERTClassifier
from utils.utils import Evaluate
import tqdm
import os

def load_model(model_path, num_classes=2, dropout_rate=0.1, device="cuda"):
    """
    Load the trained ClinicalBERTClassifier model from a saved state dict.
    
    Args:
        model_path (str): Path to the saved model state dict
        num_classes (int): Number of classification classes
        dropout_rate (float): Dropout rate used in the model
        device (str): Device to load the model on ('cuda' or 'cpu')
        
    Returns:
        model: Loaded ClinicalBERTClassifier model
    """
    # Create model with the same architecture as the trained model
    model = ClinicalBERTClassifier(num_classes=num_classes, dropout_rate=dropout_rate)
    
    # Load the saved state dict
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Move model to specified device
    model = model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    return model

def format_input_text(text, additional_fields=None, separator=" | "):
    """
    Format input text with additional fields, following the same format used during training.
    
    Args:
        text (str): The main clinical text
        additional_fields (dict, optional): Additional fields to include (e.g., age, gender)
        separator (str): Separator between fields
        
    Returns:
        str: Formatted input text
    """
    parts = [f"text: {text}"]
    
    if additional_fields:
        for field, value in additional_fields.items():
            parts.append(f"{field}: {value}")
    
    return separator.join(parts)

def predict_single(model, text, tokenizer, device="cuda", max_length=512, 
                   additional_fields=None, separator=" | "):
    """
    Make a prediction for a single clinical text.
    
    Args:
        model: The trained ClinicalBERTClassifier model
        text (str): Clinical text to classify
        tokenizer: HuggingFace tokenizer
        device (str): Device to use for inference
        max_length (int): Maximum sequence length
        additional_fields (dict, optional): Additional fields to include with the text
        separator (str): Separator between fields
        
    Returns:
        int: Predicted class index
        np.ndarray: Class probabilities
    """
    # Format the input text
    formatted_text = format_input_text(text, additional_fields, separator)
    
    # Tokenize
    encoding = tokenizer(
        formatted_text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Move to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred = torch.argmax(outputs, dim=1).cpu().numpy()[0]
    
    return pred, probs

def predict_batch(model, texts, tokenizer, device="cuda", max_length=512, 
                 additional_fields_list=None, separator=" | ", batch_size=16):
    """
    Make predictions for a batch of clinical texts.
    
    Args:
        model: The trained ClinicalBERTClassifier model
        texts (list): List of clinical texts to classify
        tokenizer: HuggingFace tokenizer
        device (str): Device to use for inference
        max_length (int): Maximum sequence length
        additional_fields_list (list, optional): List of additional fields dicts (one per text)
        separator (str): Separator between fields
        batch_size (int): Batch size for processing
        
    Returns:
        list: List of predicted class indices
        np.ndarray: Array of class probabilities
    """
    model.eval()
    
    # Format the input texts
    formatted_texts = []
    for i, text in enumerate(texts):
        add_fields = None if additional_fields_list is None else additional_fields_list[i]
        formatted_texts.append(format_input_text(text, add_fields, separator))
    
    all_predictions = []
    all_probabilities = []
    
    # Process in batches
    for i in range(0, len(formatted_texts), batch_size):
        batch_texts = formatted_texts[i:i+batch_size]
        
        # Tokenize
        encodings = tokenizer(
            batch_texts,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        
        # Make predictions
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
        
        all_predictions.extend(preds)
        all_probabilities.append(probs)
    
    return all_predictions, np.vstack(all_probabilities)

def predict_from_csv(model, csv_path, tokenizer, device="cuda", max_length=512, 
                    text_column='text', columns_to_include=None, separator=" | ", batch_size=16):
    """
    Make predictions for all texts in a CSV file.
    
    Args:
        model: The trained ClinicalBERTClassifier model
        csv_path (str): Path to the CSV file
        tokenizer: HuggingFace tokenizer
        device (str): Device to use for inference
        max_length (int): Maximum sequence length
        text_column (str): Name of the column containing the clinical text
        columns_to_include (list, optional): Columns to include as additional fields
        separator (str): Separator between fields
        batch_size (int): Batch size for processing
        
    Returns:
        pd.DataFrame: DataFrame with original data and predictions
    """
    # Load the CSV
    df = pd.read_csv(csv_path)
    
    # Check if text_column exists
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found in CSV")
    
    # Extract texts and additional fields
    texts = df[text_column].tolist()
    
    additional_fields_list = None
    if columns_to_include:
        # Filter out columns not in the CSV
        valid_columns = [col for col in columns_to_include if col in df.columns and col != text_column]
        
        # Create a list of additional fields dictionaries
        additional_fields_list = []
        for _, row in df.iterrows():
            fields = {col: row[col] for col in valid_columns}
            additional_fields_list.append(fields)
    
    # Get predictions
    predictions, probabilities = predict_batch(
        model, texts, tokenizer, device, max_length, 
        additional_fields_list, separator, batch_size
    )
    
    # Add predictions to dataframe
    df['prediction'] = predictions
    
    # Add prediction probabilities
    for i in range(probabilities.shape[1]):
        df[f'prob_class_{i}'] = probabilities[:, i]
    
    return df

def interpret_prediction(prediction, probability=None, readmission_threshold=30):
    """
    Interpret the numerical prediction into a meaningful readmission risk category.
    
    Args:
        prediction (int): The model's numerical prediction
        probability (float, optional): The confidence of the prediction
        readmission_threshold (int): Days threshold for readmission
        
    Returns:
        str: Interpretation of the prediction
    """
    # For binary classification (30-day readmission or not)
    if prediction == 0:
        category = f"High risk of readmission within {readmission_threshold} days"
    else:
        category = f"Low risk of readmission within {readmission_threshold} days"
    
    if probability is not None:
        confidence = f" (confidence: {probability:.2f})"
        return category + confidence
    return category

def main():
    """
    Main function for demonstration purposes
    """
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
    
    # Load the trained model
    model_path = "clinical_bert_readmission_classifier.pt"
    num_classes = 2  # Binary classification for readmission
    model = load_model(model_path, num_classes=num_classes, device=device)
    
    # Example of single prediction
    example_text = "Patient presents with shortness of breath and chest pain. History of COPD."
    additional_fields = {
        "anchor_age": 65,
        "gender": "Male",
        "race": "White",
        "admission_type": "EMERGENCY"
    }
    
    prediction, probabilities = predict_single(
        model, example_text, tokenizer, device=device, additional_fields=additional_fields
    )
    
    # Interpret the prediction
    interpretation = interpret_prediction(prediction, probabilities[prediction])
    
    print(f"\nSingle Prediction Example:")
    print(f"Text: {example_text}")
    print(f"Additional Fields: {additional_fields}")
    print(f"Prediction: {prediction}")
    print(f"Probabilities: {probabilities}")
    print(f"Interpretation: {interpretation}") 
    
    print("\nRunning full evaluation on test dataset...")
    try:
        from dataset.torch_diag import DiagTorch
        from torch.utils.data import DataLoader
        
        # Create test dataset
        test_data_path = "dataset/new_discharge_master.csv"
        columns_to_include = ['text', 'anchor_age', 'gender', 'race']
        
        test_dataset = DiagTorch(
            data_path=test_data_path,
            label='time_until_next_admission',
            subset='test',
            tokenizer=tokenizer,
            max_length=512,
            text_column='text',
            total_bins=num_classes,
            columns_to_include=columns_to_include,
            separator=" | "
        )
        
        # Create test dataloader
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=16,
            num_workers=2,
            pin_memory=True
        )
        
        # Define loss function
        loss_func = nn.CrossEntropyLoss()
        
        # Run evaluation
        epoch = 0
        output_path = "evaluation_results"
        os.makedirs(output_path, exist_ok=True)
        os.makedirs("log", exist_ok=True)
        
        # Update the model's forward method to match the expected input format from the Evaluate function
        original_forward = model.forward
        
        # Create a wrapper forward method
        def forward_wrapper(text, mask, age):
            return original_forward(input_ids=text, attention_mask=mask)
            
        # Temporarily replace the forward method
        model.forward = forward_wrapper
        
        # Run evaluation
        val_loss, accuracy = Evaluate(
            model=model,
            test_loader=test_dataloader,
            loss_func=loss_func,
            cls=num_classes,
            device=device,
            epoch=epoch,
            path=output_path,
            language_model="ClinicalBERT",
            log="evaluation_results.txt"
        )
        
        # Restore original forward method
        model.forward = original_forward
        
        print(f"\nEvaluation complete. Results saved to {output_path}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
    
    except Exception as e:
        print(f"Could not run evaluation: {str(e)}")
        print("Please ensure you have all required files and dependencies for evaluation.")


if __name__ == "__main__":
    main()