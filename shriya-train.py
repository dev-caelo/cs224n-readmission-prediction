import os
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union
import json
from datetime import datetime
import logging

# Import our modules
from data_processor import PatientData, DataProcessor
from model import create_model, NeuralNetworkModel, XGBoostWrapper, ModelInterpreter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PatientDataset(Dataset):
    """PyTorch Dataset for patient data"""
    def __init__(self, patient_data_list: List[PatientData]):
        """
        Initialize the dataset
        
        Args:
            patient_data_list: List of PatientData objects
        """
        self.patient_data = patient_data_list
        
    def __len__(self):
        return len(self.patient_data)
    
    def __getitem__(self, idx):
        patient = self.patient_data[idx]
        
        return {
            'text_embedding': torch.tensor(patient.text_embedding, dtype=torch.float32),
            'demographics': torch.tensor(patient.demographics, dtype=torch.float32),
            'diagnoses': torch.tensor(patient.diagnoses, dtype=torch.float32),
            'target': torch.tensor(patient.time_until_next_admission, dtype=torch.float32)
        }


def prepare_data_loaders(
    train_data: List[PatientData],
    val_data: List[PatientData],
    test_data: List[PatientData],
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Prepare PyTorch DataLoaders from the data
    
    Args:
        train_data: List of training PatientData objects
        val_data: List of validation PatientData objects
        test_data: List of test PatientData objects
        batch_size: Batch size for the DataLoaders
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = PatientDataset(train_data)
    val_dataset = PatientDataset(val_data)
    test_dataset = PatientDataset(test_data)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def train_neural_network(
    model: NeuralNetworkModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    num_epochs: int = 50,
    patience: int = 5,
    output_dir: str = "models"
) -> Tuple[NeuralNetworkModel, Dict]:
    """
    Train a neural network model
    
    Args:
        model: The neural network model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to train on (CPU or GPU)
        learning_rate: Learning rate for the optimizer
        weight_decay: Weight decay for regularization
        num_epochs: Maximum number of epochs to train for
        patience: Number of epochs to wait for improvement before early stopping
        output_dir: Directory to save model checkpoints
        
    Returns:
        Tuple of (trained_model, training_history)
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize variables for early stopping
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None
    
    # Initialize training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_rmse': [],
        'val_rmse': []
    }
    
    logger.info(f"Starting neural network training for {num_epochs} epochs")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        for batch in train_loader:
            # Move data to device
            text_embeddings = batch['text_embedding'].to(device)
            demographics = batch['demographics'].to(device)
            diagnoses = batch['diagnoses'].to(device)
            targets = batch['target'].to(device).view(-1, 1)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(text_embeddings, demographics, diagnoses)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track loss and predictions
            train_loss += loss.item() * text_embeddings.size(0)
            train_preds.extend(outputs.detach().cpu().numpy())
            train_targets.extend(targets.detach().cpu().numpy())
        
        # Calculate average training loss and metrics
        train_loss /= len(train_loader.dataset)
        train_rmse = np.sqrt(mean_squared_error(train_targets, train_preds))
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Move data to device
                text_embeddings = batch['text_embedding'].to(device)
                demographics = batch['demographics'].to(device)
                diagnoses = batch['diagnoses'].to(device)
                targets = batch['target'].to(device).view(-1, 1)
                
                # Forward pass
                outputs = model(text_embeddings, demographics, diagnoses)
                
                # Calculate loss
                loss = criterion(outputs, targets)
                
                # Track loss and predictions
                val_loss += loss.item() * text_embeddings.size(0)
                val_preds.extend(outputs.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        
        # Calculate average validation loss and metrics
        val_loss /= len(val_loader.dataset)
        val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
        
        # Update training history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_rmse'].append(train_rmse)
        history['val_rmse'].append(val_rmse)
        
        # Print progress
        logger.info(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train RMSE: {train_rmse:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val RMSE: {val_rmse:.4f}")
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
            
            # Save the best model
            model_path = os.path.join(output_dir, "best_model.pt")
            model.save(model_path)
            logger.info(f"Saved best model to {model_path}")
        else:
            epochs_without_improvement += 1
            
        # Early stopping
        if epochs_without_improvement >= patience:
            logger.info(f"Early stopping after {epoch+1} epochs")
            break
    
    # Load the best model state
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model, history


def train_xgboost(
    model: XGBoostWrapper,
    train_data: List[PatientData],
    val_data: List[PatientData],
    early_stopping_rounds: int = 10,
    output_dir: str = "models"
) -> Tuple[XGBoostWrapper, Dict]:
    """
    Train an XGBoost model
    
    Args:
        model: The XGBoost model wrapper to train
        train_data: List of training PatientData objects
        val_data: List of validation PatientData objects
        early_stopping_rounds: Number of rounds to wait for improvement before early stopping
        output_dir: Directory to save model checkpoints
        
    Returns:
        Tuple of (trained_model, training_history)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare input features and targets
    X_train = {
        'text_embeddings': np.array([p.text_embedding for p in train_data]),
        'demographics': np.array([p.demographics for p in train_data]),
        'diagnoses': np.array([p.diagnoses for p in train_data])
    }
    
    y_train = np.array([p.time_until_next_admission for p in train_data])
    
    X_val = {
        'text_embeddings': np.array([p.text_embedding for p in val_data]),
        'demographics': np.array([p.demographics for p in val_data]),
        'diagnoses': np.array([p.diagnoses for p in val_data])
    }
    
    y_val = np.array([p.time_until_next_admission for p in val_data])
    
    logger.info("Starting XGBoost training")
    
    # Train the model
    model.fit(
        X_train, 
        y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=early_stopping_rounds,
        verbose=True
    )
    
    # Save the model
    model_path = os.path.join(output_dir, "xgboost_model.pkl")
    model.save(model_path)
    logger.info(f"Saved XGBoost model to {model_path}")
    
    # Get training history (from evaluation results)
    history = {
        'train_rmse': model.model.evals_result_.get('validation_0', {}).get('rmse', []),
        'val_rmse': model.model.evals_result_.get('validation_1', {}).get('rmse', [])
    }
    
    return model, history


def evaluate_model(
    model: Union[NeuralNetworkModel, XGBoostWrapper],
    test_data: List[PatientData],
    device: Optional[torch.device] = None,
    output_dir: str = "results"
) -> Dict:
    """
    Evaluate a trained model on test data
    
    Args:
        model: The trained model to evaluate
        test_data: List of test PatientData objects
        device: Device to evaluate on (for neural networks)
        output_dir: Directory to save evaluation results
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Evaluating model on test data")
    
    if isinstance(model, NeuralNetworkModel):
        # Move model to device
        model = model.to(device)
        model.eval()
        
        # Create DataLoader
        test_loader = DataLoader(PatientDataset(test_data), batch_size=32, shuffle=False)
        
        # Make predictions
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                # Move data to device
                text_embeddings = batch['text_embedding'].to(device)
                demographics = batch['demographics'].to(device)
                diagnoses = batch['diagnoses'].to(device)
                target = batch['target'].numpy()
                
                # Forward pass
                outputs = model(text_embeddings, demographics, diagnoses)
                pred = outputs.cpu().numpy()
                
                # Store predictions and targets
                predictions.extend(pred)
                targets.extend(target)
                
    elif isinstance(model, XGBoostWrapper):
        # Prepare input features
        X_test = {
            'text_embeddings': np.array([p.text_embedding for p in test_data]),
            'demographics': np.array([p.demographics for p in test_data]),
            'diagnoses': np.array([p.diagnoses for p in test_data])
        }
        
        targets = [p.time_until_next_admission for p in test_data]
        
        # Make predictions
        predictions = model.predict(X_test)
    
    # Calculate metrics
    predictions = np.array(predictions).flatten()
    targets = np.array(targets).flatten()
    
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    
    # Store metrics
    metrics = {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2)
    }
    
    # Save metrics
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Plot predictions vs targets
    plt.figure(figsize=(10, 6))
    plt.scatter(targets, predictions, alpha=0.3)
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Predicted vs Actual Time Until Next Admission')
    plt.savefig(os.path.join(output_dir, "predictions_vs_actual.png"))
    
    logger.info(f"Evaluation metrics: MSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
    logger.info(f"Saved evaluation results to {output_dir}")
    
    return metrics


def plot_training_history(history: Dict, output_dir: str = "results"):
    """
    Plot training history
    
    Args:
        history: Dictionary of training history
        output_dir: Directory to save plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot training and validation loss
    if 'train_loss' in history and 'val_loss' in history:
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig(os.path.join(output_dir, "loss_history.png"))
    
    # Plot training and validation RMSE
    if 'train_rmse' in history and 'val_rmse' in history:
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_rmse'], label='Training RMSE')
        plt.plot(history['val_rmse'], label='Validation RMSE')
        plt.xlabel('Epoch' if isinstance(model, NeuralNetworkModel) else 'Boosting Round')
        plt.ylabel('RMSE')
        plt.title('Training and Validation RMSE')
        plt.legend()
        plt.savefig(os.path.join(output_dir, "rmse_history.png"))


def main():
    """Main function to train and evaluate models"""
    parser = argparse.ArgumentParser(description="Train a model to predict time until next admission")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the CSV data file")
    parser.add_argument("--processed_data_dir", type=str, default="processed_data", help="Directory to save/load processed data")
    parser.add_argument("--model_type", type=str, choices=["neural_network", "xgboost"], default="neural_network", help="Type of model to train")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory to save trained models")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save evaluation results")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for neural network")
    parser.add_argument("--num_epochs", type=int, default=50, help="Maximum number of epochs for neural network")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
    parser.add_argument("--embedding_model", type=str, default="emilyalsentzer/Bio_ClinicalBERT", help="Pre-trained model for text embeddings")
    parser.add_argument("--force_preprocess", action="store_true", help="Force preprocessing even if processed data exists")
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.processed_data_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Check if processed data exists
    processed_data_exists = (
        os.path.exists(os.path.join(args.processed_data_dir, "train_data.pkl")) and
        os.path.exists(os.path.join(args.processed_data_dir, "val_data.pkl")) and
        os.path.exists(os.path.join(args.processed_data_dir, "test_data.pkl")) and
        os.path.exists(os.path.join(args.processed_data_dir, "processor_data.pkl"))
    )
    
    # Process data or load from disk
    if not processed_data_exists or args.force_preprocess:
        logger.info("Processing data from CSV")
        processor = DataProcessor(embedding_model_name=args.embedding_model)
        train_data, val_data, test_data = processor.preprocess_data(args.data_path)
        processor.save_preprocessed_data(
            train_data, val_data, test_data, 
            output_dir=args.processed_data_dir
        )
    else:
        logger.info("Loading preprocessed data from disk")
        # Load processor
        processor = DataProcessor.load_processor(
            os.path.join(args.processed_data_dir, "processor_data.pkl")
        )
        
        # Load data
        with open(os.path.join(args.processed_data_dir, "train_data.pkl"), 'rb') as f:
            train_data = pickle.load(f)
        
        with open(os.path.join(args.processed_data_dir, "val_data.pkl"), 'rb') as f:
            val_data = pickle.load(f)
            
        with open(os.path.join(args.processed_data_dir, "test_data.pkl"), 'rb') as f:
            test_data = pickle.load(f)
    
    # Determine model dimensions
    embedding_dim = train_data[0].text_embedding.shape[0]
    demographics_dim = train_data[0].demographics.shape[0]
    diagnoses_dim = train_data[0].diagnoses.shape[0]
    
    # Create model
    logger.info(f"Creating {args.model_type} model")
    model = create_model(
        model_type=args.model_type,
        embedding_dim=embedding_dim,
        demographics_dim=demographics_dim,
        diagnoses_dim=diagnoses_dim
    )
    
    # Train model
    if args.model_type == "neural_network":
        # Prepare data loaders
        train_loader, val_loader, test_loader = prepare_data_loaders(
            train_data, val_data, test_data, 
            batch_size=args.batch_size
        )
        
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Train model
        model, history = train_neural_network(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            patience=args.patience,
            output_dir=args.model_dir
        )
        
        # Evaluate model
        metrics = evaluate_model(
            model=model,
            test_data=test_data,
            device=device,
            output_dir=args.results_dir
        )
        
    elif args.model_type == "xgboost":
        # Train model
        model, history = train_xgboost(
            model=model,
            train_data=train_data,
            val_data=val_data,
            early_stopping_rounds=args.patience,
            output_dir=args.model_dir
        )
        
        # Evaluate model
        metrics = evaluate_model(
            model=model,
            test_data=test_data,
            output_dir=args.results_dir
        )
    
    # Plot training history
    plot_training_history(history, output_dir=args.results_dir)
    
    logger.info("Training and evaluation complete")
    
    # Save summary
    summary = {
        'model_type': args.model_type,
        'embedding_model': args.embedding_model,
        'train_data_size': len(train_data),
        'val_data_size': len(val_data),
        'test_data_size': len(test_data),
        'metrics': metrics,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(args.results_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=4)


if __name__ == "__main__":
    main()
