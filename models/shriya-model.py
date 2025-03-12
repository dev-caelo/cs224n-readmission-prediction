import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Union, Optional, Tuple, List
import pickle
import os
import h5py
import json
import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin
from dataclasses import dataclass

class NeuralNetworkModel(nn.Module):
    """Neural network model for predicting time until next admission"""
    def __init__(
        self, 
        embedding_dim: int = 768,
        demographics_dim: int = None,
        diagnoses_dim: int = None,
        hidden_dims: List[int] = [512, 256, 128],
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True
    ):
        """
        Initialize the neural network model
        
        Args:
            embedding_dim: Dimension of the text embeddings
            demographics_dim: Dimension of the demographics features
            diagnoses_dim: Dimension of the diagnoses features
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate for regularization
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.demographics_dim = demographics_dim
        self.diagnoses_dim = diagnoses_dim
        
        # Calculate total input dimension
        total_input_dim = embedding_dim
        if demographics_dim:
            total_input_dim += demographics_dim
        if diagnoses_dim:
            total_input_dim += diagnoses_dim
            
        # Create layers
        layers = []
        prev_dim = total_input_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(dim))
                
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim
            
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers)
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
    def forward(
        self, 
        text_embedding: torch.Tensor,
        demographics: Optional[torch.Tensor] = None,
        diagnoses: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the model
        
        Args:
            text_embedding: Tensor of shape (batch_size, embedding_dim)
            demographics: Optional tensor of shape (batch_size, demographics_dim)
            diagnoses: Optional tensor of shape (batch_size, diagnoses_dim)
            
        Returns:
            Tensor of shape (batch_size, 1) with predictions
        """
        # Concatenate inputs
        inputs = [text_embedding]
        
        if demographics is not None and self.demographics_dim:
            inputs.append(demographics)
            
        if diagnoses is not None and self.diagnoses_dim:
            inputs.append(diagnoses)
            
        combined_input = torch.cat(inputs, dim=1)
        
        # Forward pass
        return self.model(combined_input)
    
    def save(self, path: str):
        """Save the model to a file"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'embedding_dim': self.embedding_dim,
            'demographics_dim': self.demographics_dim,
            'diagnoses_dim': self.diagnoses_dim,
            'model_config': {
                'hidden_dims': self.hidden_dims,
                'dropout_rate': self.dropout_rate,
                'use_batch_norm': self.use_batch_norm
            }
        }, path)
        
    @classmethod
    def load(cls, path: str) -> 'NeuralNetworkModel':
        """Load a model from a file"""
        checkpoint = torch.load(path)
        
        model = cls(
            embedding_dim=checkpoint['embedding_dim'],
            demographics_dim=checkpoint['demographics_dim'],
            diagnoses_dim=checkpoint['diagnoses_dim'],
            hidden_dims=checkpoint['model_config']['hidden_dims'],
            dropout_rate=checkpoint['model_config']['dropout_rate'],
            use_batch_norm=checkpoint['model_config']['use_batch_norm']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        return model


class XGBoostWrapper(BaseEstimator, RegressorMixin):
    """Wrapper for XGBoost model to make it compatible with our data format"""
    def __init__(
        self,
        embedding_dim: int = 768,
        demographics_dim: Optional[int] = None,
        diagnoses_dim: Optional[int] = None,
        xgb_params: Optional[Dict] = None
    ):
        """
        Initialize the XGBoost wrapper
        
        Args:
            embedding_dim: Dimension of the text embeddings
            demographics_dim: Dimension of the demographics features
            diagnoses_dim: Dimension of the diagnoses features
            xgb_params: Parameters to pass to XGBoost
        """
        self.embedding_dim = embedding_dim
        self.demographics_dim = demographics_dim
        self.diagnoses_dim = diagnoses_dim
        
        # Default XGBoost parameters if not provided
        if xgb_params is None:
            xgb_params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'eta': 0.05,
                'max_depth': 6,
                'min_child_weight': 1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'n_estimators': 500
            }
        
        self.xgb_params = xgb_params
        
        # Initialize the model
        self.model = xgb.XGBRegressor(**self.xgb_params)
        
    def fit(
        self, 
        X: Dict[str, np.ndarray], 
        y: np.ndarray,
        eval_set=None,
        **kwargs
    ):
        """
        Fit the model to the data
        
        Args:
            X: Dictionary with keys 'text_embedding', 'demographics', 'diagnoses'
            y: Target values
            eval_set: Evaluation set for early stopping
            **kwargs: Additional arguments to pass to XGBoost fit method
        """
        # Combine features
        combined_features = self._combine_features(X)
        
        # Convert eval_set if provided
        if eval_set:
            processed_eval_set = [(self._combine_features(es[0]), es[1]) for es in eval_set]
        else:
            processed_eval_set = None
        
        # Fit the model
        self.model.fit(
            combined_features, 
            y, 
            eval_set=processed_eval_set,
            **kwargs
        )
        
        return self
    
    def predict(self, X: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Make predictions with the model
        
        Args:
            X: Dictionary with keys 'text_embedding', 'demographics', 'diagnoses'
            
        Returns:
            Array of predictions
        """
        combined_features = self._combine_features(X)
        return self.model.predict(combined_features)
    
    def _combine_features(self, X: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine different feature types into a single array"""
        features_to_combine = [X['text_embedding']]
        
        if 'demographics' in X and self.demographics_dim:
            features_to_combine.append(X['demographics'])
            
        if 'diagnoses' in X and self.diagnoses_dim:
            features_to_combine.append(X['diagnoses'])
            
        return np.hstack(features_to_combine)
    
    def save(self, path: str):
        """Save the model to a file"""
        model_data = {
            'model': self.model,
            'embedding_dim': self.embedding_dim,
            'demographics_dim': self.demographics_dim,
            'diagnoses_dim': self.diagnoses_dim,
            'xgb_params': self.xgb_params
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
            
    @classmethod
    def load(cls, path: str) -> 'XGBoostWrapper':
        """Load a model from a file"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
            
        wrapper = cls(
            embedding_dim=model_data['embedding_dim'],
            demographics_dim=model_data['demographics_dim'],
            diagnoses_dim=model_data['diagnoses_dim'],
            xgb_params=model_data['xgb_params']
        )
        
        wrapper.model = model_data['model']
        return wrapper


# Factory function to create the desired model
def create_model(model_type: str = "neural_network", **model_params):
    """
    Create a model of the specified type
    
    Args:
        model_type: Type of model to create ('neural_network' or 'xgboost')
        **model_params: Parameters to pass to the model constructor
        
    Returns:
        The created model
    """
    if model_type.lower() == "neural_network":
        return NeuralNetworkModel(**model_params)
    elif model_type.lower() == "xgboost":
        return XGBoostWrapper(**model_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Model interpreter class for feature importance and bias analysis
class ModelInterpreter:
    """Class for interpreting model predictions and analyzing feature importance"""
    def __init__(
        self, 
        model: Union[NeuralNetworkModel, XGBoostWrapper],
        data_dir: str,
        processor_dir: str
    ):
        """
        Initialize the model interpreter
        
        Args:
            model: The trained model to interpret
            data_dir: Directory with processed data
            processor_dir: Directory with processor data
        """
        self.model = model
        self.data_dir = data_dir
        self.processor_dir = processor_dir
        
        # Load processor data to get feature mappings
        processor_path = os.path.join(processor_dir, 'processor_data.pkl')
        with open(processor_path, 'rb') as f:
            processor_data = pickle.load(f)
            
        self.demographics_encoder = processor_data.get('demographics_encoder')
        self.diagnoses_encoder = processor_data.get('diagnoses_encoder')
        
    def get_feature_importance(self):
        """Get feature importance (currently only supported for XGBoost)"""
        if isinstance(self.model, XGBoostWrapper):
            # Get raw importance scores
            importance = self.model.model.feature_importances_
            
            # Calculate feature start indices
            embedding_size = self.model.embedding_dim
            demo_size = self.model.demographics_dim if self.model.demographics_dim else 0
            diag_size = self.model.diagnoses_dim if self.model.diagnoses_dim else 0
            
            # Split importance by feature type
            embedding_importance = importance[:embedding_size]
            demo_importance = importance[embedding_size:embedding_size+demo_size] if demo_size > 0 else []
            diag_importance = importance[embedding_size+demo_size:] if diag_size > 0 else []
            
            # Map demographics feature indices to feature names if possible
            demo_features = {}
            if demo_size > 0 and self.demographics_encoder:
                feature_names = []
                for i, category in enumerate(self.demographics_encoder.categories_):
                    for j, value in enumerate(category):
                        feature_names.append(f"{i}_{value}")
                        
                # Map importance to feature names (approximate mapping)
                if len(feature_names) == len(demo_importance):
                    demo_features = {name: float(imp) for name, imp in zip(feature_names, demo_importance)}
                    
            # Map diagnoses feature indices to ICD codes if possible
            diag_features = {}
            if diag_size > 0 and self.diagnoses_encoder:
                # Reverse the diagnoses encoder
                rev_encoder = {v: k for k, v in self.diagnoses_encoder.items()}
                
                # Map importance to ICD codes
                diag_features = {rev_encoder.get(i, f"diag_{i}"): float(imp) 
                                for i, imp in enumerate(diag_importance)}
            
            return {
                'overall_embedding_importance': float(np.sum(embedding_importance)),
                'overall_demographics_importance': float(np.sum(demo_importance)) if len(demo_importance) > 0 else 0,
                'overall_diagnoses_importance': float(np.sum(diag_importance)) if len(diag_importance) > 0 else 0,
                'demographics_features': demo_features,
                'diagnoses_features': {k: v for k, v in sorted(diag_features.items(), 
                                                              key=lambda item: item[1], reverse=True)[:20]}
            }
        else:
            raise NotImplementedError("Feature importance is currently only supported for XGBoost models")
    
    def analyze_bias(self, data_path: str, sensitive_attributes: List[str] = None):
        """
        Analyze bias in model predictions across demographic groups
        
        Args:
            data_path: Path to the HDF5 data file
            sensitive_attributes: List of sensitive attribute names to analyze
            
        Returns:
            Dictionary with bias analysis results
        """
        if sensitive_attributes is None:
            sensitive_attributes = ['gender', 'race', 'insurance']
            
        # This is a placeholder implementation - in practice, you would:
        # 1. Load a sample of the data
        # 2. Get model predictions
        # 3. Calculate performance metrics across different demographic groups
        # 4. Compare performance to identify potential biases
        
        return {
            'status': 'Bias analysis not yet implemented',
            'attributes_to_analyze': sensitive_attributes
        }
    
    def _visualize_embeddings(self, num_samples: int = 1000):
        """
        Visualize text embeddings to understand patterns
        
        Args:
            num_samples: Number of samples to use for visualization
            
        Returns:
            Dictionary with visualization data
        """
        # In a full implementation, this would:
        # 1. Sample embeddings from the data
        # 2. Use dimensionality reduction (PCA, t-SNE, UMAP) to visualize
        # 3. Color points by target variable or prediction error
        
        return {
            'status': 'Embedding visualization not yet implemented',
            'num_samples': num_samples
        }
    
    def explain_prediction(self, patient_features: Dict):
        """
        Explain a prediction for a specific patient
        
        Args:
            patient_features: Dictionary with patient features
            
        Returns:
            Dictionary with explanation
        """
        # This is a placeholder - in practice, you would:
        # 1. Use techniques like SHAP values to explain predictions
        # 2. Identify key features contributing to the prediction
        # 3. Generate a human-readable explanation
        
        return {
            'status': 'Prediction explanation not yet implemented',
            'prediction': float(self.model.predict(patient_features)[0]) if hasattr(self.model, 'predict') else None
        }