import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Union, Optional, Tuple, List
import pickle
import os
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
        
    def forward(
        self, 
        text_embeddings: torch.Tensor,
        demographics: Optional[torch.Tensor] = None,
        diagnoses: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the model
        
        Args:
            text_embeddings: Tensor of shape (batch_size, embedding_dim)
            demographics: Optional tensor of shape (batch_size, demographics_dim)
            diagnoses: Optional tensor of shape (batch_size, diagnoses_dim)
            
        Returns:
            Tensor of shape (batch_size, 1) with predictions
        """
        # Concatenate inputs
        inputs = [text_embeddings]
        
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
                'hidden_dims': [layer.out_features for layer in self.model if isinstance(layer, nn.Linear)][:-1],
                'dropout_rate': self.model[3].p if len(self.model) > 3 and isinstance(self.model[3], nn.Dropout) else 0.3,
                'use_batch_norm': any(isinstance(layer, nn.BatchNorm1d) for layer in self.model)
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
        **xgb_params
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
        default_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'eta': 0.05,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_estimators': 500
        }
        
        # Update defaults with provided parameters
        self.xgb_params = {**default_params, **xgb_params}
        
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
            X: Dictionary with keys 'text_embeddings', 'demographics', 'diagnoses'
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
            X: Dictionary with keys 'text_embeddings', 'demographics', 'diagnoses'
            
        Returns:
            Array of predictions
        """
        combined_features = self._combine_features(X)
        return self.model.predict(combined_features)
    
    def _combine_features(self, X: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine different feature types into a single array"""
        features_to_combine = [X['text_embeddings']]
        
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
            **model_data['xgb_params']
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


# Model interpreter class (for future use)
class ModelInterpreter:
    """Class for interpreting model predictions and analyzing feature importance"""
    def __init__(self, model: Union[NeuralNetworkModel, XGBoostWrapper]):
        """
        Initialize the model interpreter
        
        Args:
            model: The trained model to interpret
        """
        self.model = model
        
    def get_feature_importance(self):
        """Get feature importance (currently only supported for XGBoost)"""
        if isinstance(self.model, XGBoostWrapper):
            return self.model.model.feature_importances_
        else:
            raise NotImplementedError("Feature importance is currently only supported for XGBoost models")
    
    def analyze_bias(self, sensitive_attributes: List[str], test_data, predictions):
        """
        Analyze bias in model predictions
        
        This is a placeholder for future implementation
        """
        # Placeholder for future implementation
        pass
    
    def explain_prediction(self, patient_data):
        """
        Explain a prediction for a specific patient
        
        This is a placeholder for future implementation
        """
        # Placeholder for future implementation
        pass
