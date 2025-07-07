import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional
import pandas as pd
from pathlib import Path

class Visualizer:
    def __init__(self, output_dir: str = "visualizations"):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_embedding_space(self, 
                           embeddings: np.ndarray,
                           labels: np.ndarray,
                           title: str = "Embedding Space Visualization",
                           save_path: Optional[str] = None):
        """
        Create t-SNE visualization of embeddings.
        
        Args:
            embeddings: High-dimensional embeddings
            labels: Labels for each embedding
            title: Plot title
            save_path: Optional path to save the plot
        """
        # Adjust t-SNE parameters for small datasets
        n_samples = embeddings.shape[0]
        perplexity = min(30, n_samples - 1)  # Default is 30, but must be less than n_samples
        
        # Reduce dimensionality with t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Create scatter plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='coolwarm')
        plt.colorbar(scatter)
        plt.title(title)
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def plot_metrics_comparison(self,
                                metrics_dict: Dict[str, Dict[str, float]],
                                title: str = "Model Performance Comparison",
                                save_path: Optional[str] = None):
        """
        Create bar plot comparing model performance metrics.
        
        Args:
            metrics_dict: Dictionary of model metrics
            title: Plot title
            save_path: Optional path to save the plot
        """
        # Prepare data for plotting
        models = list(metrics_dict.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        
        # Create figure
        fig = go.Figure()
        
        # Add bars for each metric
        x_positions = np.arange(len(metrics))
        width = 0.35
        
        for i, model in enumerate(models):
            model_metrics = []
            for metric in metrics:
                # Get training metric
                train_key = f'train_{metric}'
                train_value = metrics_dict[model].get(train_key, 0)
                model_metrics.append(train_value)
            
            fig.add_trace(go.Bar(
                name=model,
                x=metrics,
                y=model_metrics,
                text=[f'{v:.2f}' if v is not None else 'N/A' for v in model_metrics],
                textposition='auto',
            ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Metric",
            yaxis_title="Score",
            barmode='group',
            yaxis=dict(range=[0, 1])
        )
        
        if save_path:
            fig.write_html(save_path)
    
    def plot_feature_importance(self,
                              importance_dict: Dict[str, float],
                              title: str = "Feature Importance",
                              save_path: Optional[str] = None):
        """
        Create bar plot of feature importance.
        
        Args:
            importance_dict: Dictionary of feature importance scores
            title: Plot title
            save_path: Optional path to save the plot
        """
        # Sort features by importance
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        features, scores = zip(*sorted_features)
        
        # Create bar plot
        plt.figure(figsize=(12, 6))
        plt.bar(features, scores)
        plt.xticks(rotation=45, ha='right')
        plt.title(title)
        plt.xlabel("Features")
        plt.ylabel("Importance Score")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def plot_confusion_matrix(self,
                            y_true: np.ndarray,
                            y_pred: np.ndarray,
                            title: str = "Confusion Matrix") -> None:
        """
        Create confusion matrix visualization.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            title: Plot title
        """
        confusion_matrix = pd.crosstab(
            pd.Series(y_true, name='Actual'),
            pd.Series(y_pred, name='Predicted')
        )
        
        fig = px.imshow(
            confusion_matrix,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['No Conversion', 'Conversion'],
            y=['No Conversion', 'Conversion'],
            title=title,
            color_continuous_scale='RdBu'
        )
        
        fig.write_html(self.output_dir / f"{title.lower().replace(' ', '_')}.html") 