import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json

class DataProcessor:
    def __init__(self, raw_data_path: str, processed_data_path: str):
        """
        Initialize the DataProcessor.
        
        Args:
            raw_data_path: Path to raw transcript data
            processed_data_path: Path to save processed data
        """
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        
    def load_transcript(self, file_path: str) -> Dict:
        """
        Load a single transcript file.
        
        Args:
            file_path: Path to the transcript file
            
        Returns:
            Dict containing transcript data
        """
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def preprocess_transcript(self, transcript: str) -> str:
        """
        Preprocess a single transcript.
        
        Args:
            transcript: Raw transcript text
            
        Returns:
            Preprocessed transcript text
        """
        # Convert to lowercase
        transcript = transcript.lower()
        
        # Remove special characters but keep punctuation
        transcript = ' '.join(transcript.split())
        
        return transcript
    
    def create_training_pairs(self, 
                            successful_transcripts: List[str], 
                            failed_transcripts: List[str],
                            n_pairs: int = 1000) -> Tuple[List[str], List[str], List[str]]:
        """
        Create training pairs for contrastive learning.
        
        Args:
            successful_transcripts: List of successful call transcripts
            failed_transcripts: List of failed call transcripts
            n_pairs: Maximum number of training pairs to generate
            
        Returns:
            Tuple of (anchor, positive, negative) examples
        """
        if len(successful_transcripts) < 2:
            raise ValueError("Need at least 2 successful transcripts for training pairs")
        
        if len(failed_transcripts) < 1:
            raise ValueError("Need at least 1 failed transcript for training pairs")
        
        # Calculate maximum possible pairs
        max_pairs = min(
            len(successful_transcripts) * (len(successful_transcripts) - 1) // 2,
            len(failed_transcripts)
        )
        
        # Adjust n_pairs if necessary
        n_pairs = min(n_pairs, max_pairs)
        
        anchors = []
        positives = []
        negatives = []
        
        # Create all possible pairs of successful transcripts
        successful_pairs = []
        for i in range(len(successful_transcripts)):
            for j in range(i + 1, len(successful_transcripts)):
                successful_pairs.append((successful_transcripts[i], successful_transcripts[j]))
        
        # Randomly select pairs
        selected_pairs = np.random.choice(len(successful_pairs), size=n_pairs, replace=False)
        
        for pair_idx in selected_pairs:
            anchor, positive = successful_pairs[pair_idx]
            negative = np.random.choice(failed_transcripts)
            
            anchors.append(anchor)
            positives.append(positive)
            negatives.append(negative)
            
        return anchors, positives, negatives
    
    def prepare_dataset(self, 
                       data: List[Dict],
                       split_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare the dataset for training and testing.
        
        Args:
            data: List of transcript dictionaries
            split_ratio: Train/test split ratio
            
        Returns:
            Tuple of (train_df, test_df)
        """
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Preprocess transcripts
        df['processed_transcript'] = df['transcript'].apply(self.preprocess_transcript)
        
        # Split into train and test
        train_size = int(len(df) * split_ratio)
        train_df = df[:train_size]
        test_df = df[train_size:]
        
        return train_df, test_df
    
    def save_processed_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """
        Save processed datasets.
        
        Args:
            train_df: Training DataFrame
            test_df: Testing DataFrame
        """
        train_df.to_parquet(self.processed_data_path / 'train.parquet')
        test_df.to_parquet(self.processed_data_path / 'test.parquet')
    
    def load_processed_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load processed datasets.
        
        Returns:
            Tuple of (train_df, test_df)
        """
        train_df = pd.read_parquet(self.processed_data_path / 'train.parquet')
        test_df = pd.read_parquet(self.processed_data_path / 'test.parquet')
        return train_df, test_df 