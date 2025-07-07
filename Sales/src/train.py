import argparse
import logging
import os
from pathlib import Path
import json
import numpy as np
from data.processor import DataProcessor
from embeddings.models import EmbeddingModel
from models.classifier import SalesClassifier
from visualization.visualizer import Visualizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_pipeline(args):
    """
    Run the complete training pipeline.
    
    Args:
        args: Command line arguments
    """
    # Initialize paths
    data_dir = Path("data")
    raw_data_dir = data_dir / "raw"
    processed_data_dir = data_dir / "processed"
    models_dir = Path("models")
    results_dir = Path("results")
    
    # Create directories if they don't exist
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    data_processor = DataProcessor(raw_data_dir, processed_data_dir)
    visualizer = Visualizer()
    
    # Load and process data
    logger.info("Loading and processing data...")
    successful_transcripts = []
    failed_transcripts = []
    
    # Load sample transcripts
    for file in raw_data_dir.glob("*.json"):
        with open(file, 'r') as f:
            data = json.load(f)
            if data['label'] == 1:
                successful_transcripts.append(data['transcript'])
            else:
                failed_transcripts.append(data['transcript'])
    
    # Create training pairs
    logger.info("Creating training pairs for fine-tuning...")
    anchors, positives, negatives = data_processor.create_training_pairs(
        successful_transcripts,
        failed_transcripts,
        n_pairs=args.n_pairs
    )
    
    # Fine-tune embedding model
    logger.info("Fine-tuning embedding model...")
    fine_tuned_embedding_model = EmbeddingModel()
    fine_tuned_embedding_model.fine_tune(
        anchors=anchors,
        positives=positives,
        negatives=negatives,
        epochs=args.epochs
    )
    
    # Generate embeddings
    logger.info("Generating embeddings...")
    all_transcripts = successful_transcripts + failed_transcripts
    labels = np.array([1] * len(successful_transcripts) + [0] * len(failed_transcripts))
    
    # Get embeddings from both models
    generic_embedding_model = EmbeddingModel()
    generic_embeddings = generic_embedding_model.get_embeddings(all_transcripts)
    fine_tuned_embeddings = fine_tuned_embedding_model.get_embeddings(all_transcripts)
    
    # Train classifiers
    logger.info("Training classifiers...")
    generic_classifier = SalesClassifier(args.classifier_type)
    fine_tuned_classifier = SalesClassifier(args.classifier_type)
    
    # Split data into train and validation sets
    train_size = int(0.8 * len(labels))
    train_indices = np.random.choice(len(labels), train_size, replace=False)
    val_indices = np.array(list(set(range(len(labels))) - set(train_indices)))
    
    # Train and evaluate classifiers
    generic_metrics = generic_classifier.train(
        generic_embeddings[train_indices],
        labels[train_indices],
        generic_embeddings[val_indices],
        labels[val_indices]
    )
    
    fine_tuned_metrics = fine_tuned_classifier.train(
        fine_tuned_embeddings[train_indices],
        labels[train_indices],
        fine_tuned_embeddings[val_indices],
        labels[val_indices]
    )
    
    # Create visualizations
    logger.info("Creating visualizations...")
    visualizer.plot_embedding_space(
        generic_embeddings,
        labels,
        title="Generic Embeddings",
        save_path=results_dir / "generic_embeddings.png"
    )
    
    visualizer.plot_embedding_space(
        fine_tuned_embeddings,
        labels,
        title="Fine-tuned Embeddings",
        save_path=results_dir / "fine_tuned_embeddings.png"
    )
    
    visualizer.plot_metrics_comparison(
        {
            'Generic': generic_metrics,
            'Fine-tuned': fine_tuned_metrics
        },
        save_path=results_dir / "performance_comparison.html"
    )
    
    # Save models and results
    logger.info("Saving models and results...")
    fine_tuned_embedding_model.save_model(models_dir / "fine_tuned_embeddings")
    generic_classifier.save_model(models_dir / "generic_classifier.pkl")
    fine_tuned_classifier.save_model(models_dir / "fine_tuned_classifier.pkl")
    
    # Save metrics
    with open(results_dir / "metrics.json", 'w') as f:
        json.dump({
            'generic': generic_metrics,
            'fine_tuned': fine_tuned_metrics
        }, f, indent=4)
    
    logger.info("Training pipeline completed successfully!")

def main():
    parser = argparse.ArgumentParser(description="Train sales conversion prediction model")
    parser.add_argument("--classifier_type", type=str, default="logistic", choices=["logistic", "xgboost"],
                      help="Type of classifier to use")
    parser.add_argument("--epochs", type=int, default=10,
                      help="Number of epochs for fine-tuning")
    parser.add_argument("--n_pairs", type=int, default=1000,
                      help="Number of training pairs for fine-tuning")
    args = parser.parse_args()
    
    train_pipeline(args)

if __name__ == "__main__":
    main() 