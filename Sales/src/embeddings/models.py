import torch
from torch import nn
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer, losses, models
from typing import List, Tuple, Optional
import numpy as np
from pathlib import Path
import openai
import os
from dotenv import load_dotenv

load_dotenv()

class TripletDataset(Dataset):
    def __init__(self, anchors: List[str], positives: List[str], negatives: List[str]):
        self.anchors = anchors
        self.positives = positives
        self.negatives = negatives
    
    def __getitem__(self, idx):
        return {
            'anchor': self.anchors[idx],
            'positive': self.positives[idx],
            'negative': self.negatives[idx]
        }
    
    def __len__(self):
        return len(self.anchors)

class EmbeddingModel:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the pre-trained model to use
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # OpenAI setup for generic embeddings comparison
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
    
    def get_generic_embeddings(self, texts: List[str], use_openai: bool = False) -> np.ndarray:
        """
        Get embeddings using the generic pre-trained model.
        
        Args:
            texts: List of texts to embed
            use_openai: Whether to use OpenAI's embedding model
            
        Returns:
            Array of embeddings
        """
        if use_openai and self.openai_api_key:
            # Use OpenAI's text-embedding-3-small model
            response = openai.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )
            embeddings = np.array([embedding.embedding for embedding in response.data])
        else:
            # Use sentence-transformers model
            embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        return embeddings
    
    def fine_tune(self, 
                  anchors: List[str], 
                  positives: List[str], 
                  negatives: List[str],
                  epochs: int = 10,
                  batch_size: int = 32,
                  learning_rate: float = 2e-5):
        """
        Fine-tune the model using contrastive learning.
        
        Args:
            anchors: List of anchor texts
            positives: List of positive texts
            negatives: List of negative texts
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for optimization
        """
        # Create triplet dataset
        train_dataset = TripletDataset(anchors, positives, negatives)
        
        def collate_fn(batch):
            anchors = [item['anchor'] for item in batch]
            positives = [item['positive'] for item in batch]
            negatives = [item['negative'] for item in batch]
            return anchors, positives, negatives
        
        # Configure training
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=batch_size,
            collate_fn=collate_fn
        )
        
        # Initialize optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # Train the model
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_dataloader:
                anchors, positives, negatives = batch
                
                # Compute embeddings with gradients
                with torch.set_grad_enabled(True):
                    # Get tokenized inputs
                    anchor_inputs = self.model.tokenize(anchors)
                    positive_inputs = self.model.tokenize(positives)
                    negative_inputs = self.model.tokenize(negatives)
                    
                    # Move inputs to device
                    for key in anchor_inputs:
                        anchor_inputs[key] = anchor_inputs[key].to(self.device)
                        positive_inputs[key] = positive_inputs[key].to(self.device)
                        negative_inputs[key] = negative_inputs[key].to(self.device)
                    
                    # Get embeddings
                    anchor_embeddings = self.model(anchor_inputs)['sentence_embedding']
                    positive_embeddings = self.model(positive_inputs)['sentence_embedding']
                    negative_embeddings = self.model(negative_inputs)['sentence_embedding']
                    
                    # Normalize embeddings
                    anchor_embeddings = torch.nn.functional.normalize(anchor_embeddings, p=2, dim=1)
                    positive_embeddings = torch.nn.functional.normalize(positive_embeddings, p=2, dim=1)
                    negative_embeddings = torch.nn.functional.normalize(negative_embeddings, p=2, dim=1)
                    
                    # Compute distances
                    distance_pos = torch.nn.functional.cosine_similarity(anchor_embeddings, positive_embeddings)
                    distance_neg = torch.nn.functional.cosine_similarity(anchor_embeddings, negative_embeddings)
                    
                    # Compute triplet loss
                    margin = 0.5
                    loss = torch.mean(torch.clamp(margin - distance_pos + distance_neg, min=0))
                    total_loss += loss.item()
                    
                    # Backward pass and optimization
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            avg_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
        
        self.model.eval()
    
    def save_model(self, save_path: str):
        """
        Save the fine-tuned model.
        
        Args:
            save_path: Path to save the model
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(save_path))
    
    def load_model(self, load_path: str):
        """
        Load a fine-tuned model.
        
        Args:
            load_path: Path to the saved model
        """
        self.model = SentenceTransformer(load_path)
        self.model.to(self.device)
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings using the current model (generic or fine-tuned).
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Array of embeddings
        """
        return self.model.encode(texts, convert_to_numpy=True)
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity score
        """
        emb1 = self.get_embeddings([text1])[0]
        emb2 = self.get_embeddings([text2])[0]
        
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)) 