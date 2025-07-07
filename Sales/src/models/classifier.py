from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import joblib
from pathlib import Path
import shap

class SalesClassifier:
    def __init__(self, classifier_type: str = "logistic"):
        """
        Initialize the sales classifier.
        
        Args:
            classifier_type: Type of classifier to use ("logistic" or "xgboost")
        """
        self.classifier_type = classifier_type
        if classifier_type == "logistic":
            self.model = LogisticRegression(max_iter=1000, random_state=42)
        elif classifier_type == "xgboost":
            self.model = XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported classifier type: {classifier_type}")
        
        self.explainer = None
    
    def train(self, 
              X_train: np.ndarray, 
              y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict:
        """
        Train the classifier.
        
        Args:
            X_train: Training features (embeddings)
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Dictionary of training metrics
        """
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Get predictions
        train_pred = self.model.predict(X_train)
        train_pred_proba = self.model.predict_proba(X_train)[:, 1]
        
        # Calculate training metrics
        metrics = {
            'train_accuracy': accuracy_score(y_train, train_pred),
            'train_precision': precision_score(y_train, train_pred, zero_division=0),
            'train_recall': recall_score(y_train, train_pred, zero_division=0),
            'train_f1': f1_score(y_train, train_pred, zero_division=0)
        }
        
        # Add AUC if we have both classes in training data
        if len(np.unique(y_train)) > 1:
            metrics['train_auc'] = roc_auc_score(y_train, train_pred_proba)
        else:
            metrics['train_auc'] = None
        
        # Calculate validation metrics if validation data is provided
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_pred_proba = self.model.predict_proba(X_val)[:, 1]
            
            metrics.update({
                'val_accuracy': accuracy_score(y_val, val_pred),
                'val_precision': precision_score(y_val, val_pred, zero_division=0),
                'val_recall': recall_score(y_val, val_pred, zero_division=0),
                'val_f1': f1_score(y_val, val_pred, zero_division=0)
            })
            
            # Add AUC if we have both classes in validation data
            if len(np.unique(y_val)) > 1:
                metrics['val_auc'] = roc_auc_score(y_val, val_pred_proba)
            else:
                metrics['val_auc'] = None
        
        # Initialize SHAP explainer
        if self.classifier_type == "xgboost":
            self.explainer = shap.TreeExplainer(self.model)
        else:
            self.explainer = shap.LinearExplainer(self.model, X_train)
        
        return metrics
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data.
        
        Args:
            X: Input features (embeddings)
            
        Returns:
            Tuple of (predictions, prediction probabilities)
        """
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        return predictions, probabilities
    
    def explain(self, X: np.ndarray) -> np.ndarray:
        """
        Generate SHAP values for feature importance.
        
        Args:
            X: Features to explain
            
        Returns:
            SHAP values for each feature
        """
        if self.explainer is None:
            raise ValueError("Model must be trained before generating explanations")
        
        return self.explainer.shap_values(X)
    
    def save_model(self, save_path: str):
        """
        Save the trained classifier.
        
        Args:
            save_path: Path to save the model
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, save_path)
    
    def load_model(self, load_path: str):
        """
        Load a trained classifier.
        
        Args:
            load_path: Path to the saved model
        """
        self.model = joblib.load(load_path) 