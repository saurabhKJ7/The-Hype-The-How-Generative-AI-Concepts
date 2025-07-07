from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class EmbeddingService:
    def __init__(self):
        self.model_name = 'tfidf'
        self.vectorizer = TfidfVectorizer(
            max_features=768,  # Keep same dimension for compatibility
            stop_words='english',
            ngram_range=(1, 2)
        )
        # Initialize with empty document to set up vocabulary
        self.vectorizer.fit([''])
        
    async def embed_text(self, text: str) -> List[float]:
        try:
            # Transform the text
            vector = self.vectorizer.transform([text])
            
            # Convert to dense array and normalize
            dense_vector = vector.toarray()[0]
            norm = np.linalg.norm(dense_vector)
            if norm > 0:
                dense_vector = dense_vector / norm
            
            return dense_vector.tolist()
        except Exception as e:
            print(f"Embedding error: {str(e)}")
            # Return zero vector as fallback
            return [0.0] * 768  # Keep same dimension for compatibility 