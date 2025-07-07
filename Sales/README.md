# Sales Conversion Predictor

An AI system that fine-tunes embeddings for sales call transcripts to improve conversion likelihood prediction. The system uses contrastive learning to create better representations of sales conversations and predicts the likelihood of successful conversions.

## Features

- **Fine-tuned Embeddings**: Uses contrastive learning with triplet loss to create domain-specific embeddings for sales conversations
- **Multiple Classifiers**: Supports both Logistic Regression and XGBoost classifiers
- **Performance Comparison**: Compares generic embeddings vs. fine-tuned embeddings
- **Visualization**: Includes t-SNE plots and performance metric comparisons
- **SHAP Explanations**: Provides feature importance analysis for model interpretability

## Project Structure

```
.
├── app/
│   └── main.py              # FastAPI application
├── data/
│   └── raw/                 # Raw transcript data
├── models/                  # Saved models
├── notebooks/              # Jupyter notebooks
├── results/                # Visualizations and metrics
├── src/
│   ├── data/
│   │   └── processor.py    # Data processing utilities
│   ├── embeddings/
│   │   └── models.py       # Embedding models
│   ├── models/
│   │   └── classifier.py   # Classification models
│   ├── visualization/
│   │   └── visualizer.py   # Visualization utilities
│   └── train.py            # Training pipeline
└── requirements.txt        # Project dependencies
```

## Installation

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

Train the model with default parameters:
```bash
python src/train.py
```

Train with specific parameters:
```bash
python src/train.py --classifier_type xgboost --epochs 20 --n_pairs 3
```

Parameters:
- `--classifier_type`: Type of classifier to use (`logistic` or `xgboost`)
- `--epochs`: Number of epochs for fine-tuning embeddings
- `--n_pairs`: Number of training pairs for contrastive learning

### Results

The training pipeline generates:
1. Fine-tuned embedding model (saved in `models/fine_tuned_embeddings/`)
2. Trained classifiers (saved in `models/`)
3. Visualizations:
   - t-SNE plots of embeddings (`results/generic_embeddings.png` and `results/fine_tuned_embeddings.png`)
   - Performance comparison (`results/performance_comparison.html`)
4. Metrics summary (`results/metrics.json`)

## Data Format

The system expects sales call transcripts in JSON format:

```json
{
    "transcript": "Text of the sales call...",
    "label": 1,  // 1 for success, 0 for failure
    "metadata": {
        "industry": "Technology",
        "product": "Enterprise Software",
        "duration": 720,
        "call_time": "2024-03-17T09:30:00Z",
        "agent_id": "SA456",
        "customer_company_size": "50-100"
    }
}
```

## Model Architecture

1. **Embedding Model**:
   - Base: `sentence-transformers/all-MiniLM-L6-v2`
   - Fine-tuning: Contrastive learning with triplet loss
   - Input: Raw transcript text
   - Output: 384-dimensional embedding vector

2. **Classifiers**:
   - Logistic Regression: Linear classifier with L2 regularization
   - XGBoost: Gradient boosting with tree-based models
   - Input: Embedding vectors
   - Output: Conversion probability

## Performance

Current metrics with the sample dataset:
- Training accuracy: ~57%
- Training F1 score: ~73%
- Validation accuracy: 50%
- Validation F1 score: ~67%

Note: Performance is limited by the small sample dataset. Production deployment would require:
1. Larger training dataset
2. More diverse examples
3. Hyperparameter optimization
4. Additional feature engineering

## Future Improvements

1. **Data Augmentation**:
   - Synthetic transcript generation
   - Industry-specific variations
   - Different conversation patterns

2. **Model Enhancements**:
   - Multi-task learning for metadata prediction
   - Attention mechanisms for key phrases
   - Ensemble methods

3. **Feature Engineering**:
   - Temporal patterns in conversations
   - Customer sentiment analysis
   - Industry-specific indicators

4. **Production Features**:
   - Real-time prediction API
   - Batch processing pipeline
   - Model monitoring and retraining
   - A/B testing framework

## License

MIT License 