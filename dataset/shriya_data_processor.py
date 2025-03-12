import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, Tuple, Generator, Iterator
import torch
from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import re
import pickle
import os
import h5py
import json
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PatientDataIndex:
    """Lightweight class to store patient data indices rather than actual data"""
    patient_id: str
    hadm_id: str
    data_index: int  # Index to locate data in storage
    split: str  # 'train', 'val', or 'test'
    
    # Additional metadata for quick filtering
    time_until_next_admission: float = None

class StreamingDataProcessor:
    def __init__(
        self, 
        output_dir: str,
        embedding_model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        embedding_dim: int = 768,
        demographics_columns: List[str] = None,
        chunk_size: int = 1000,
        max_text_length: int = 512,
        random_state: int = 42
    ):
        """
        Initialize the streaming data processor
        
        Args:
            output_dir: Directory to store processed data
            embedding_model_name: Pre-trained model to use for text embeddings
            embedding_dim: Dimension of the embeddings
            demographics_columns: Columns to use as demographic features
            chunk_size: Number of rows to process at once
            max_text_length: Maximum length for text to process (for memory efficiency)
            random_state: Random state for reproducibility
        """
        self.output_dir = output_dir
        self.embedding_model_name = embedding_model_name
        self.embedding_dim = embedding_dim
        self.chunk_size = chunk_size
        self.max_text_length = max_text_length
        self.random_state = random_state
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Define demographic columns if not provided
        self.demographics_columns = demographics_columns or [
            'gender', 'anchor_age', 'insurance', 'language', 
            'marital_status', 'race', 'admission_type'
        ]
        
        # Initialize encoders/scalers (will be fit during preprocessing)
        self.demographics_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.diagnoses_encoder = None  # Will be created during preprocessing
        self.target_scaler = StandardScaler()
        
        # Initialize embedding model
        self.tokenizer = None
        self.embedding_model = None
        
        # Store patient indices
        self.patient_indices = []
        
        # Track the current data index
        self.current_index = 0
        
    def create_storage(self):
        """Create HDF5 storage for processed data"""
        # Create a new file (will overwrite if exists)
        storage_path = os.path.join(self.output_dir, 'processed_data.h5')
        with h5py.File(storage_path, 'w') as f:
            # Create datasets with explicit data types
            f.create_dataset('text_embeddings', (0, self.embedding_dim), 
                             maxshape=(None, self.embedding_dim), dtype='float32',
                             chunks=(min(self.chunk_size, 100), self.embedding_dim))
            
            # We'll determine these sizes after processing the first chunk
            f.create_dataset('demographics', (0, 1), maxshape=(None, None), dtype='float32')
            f.create_dataset('diagnoses', (0, 1), maxshape=(None, None), dtype='float32')
            f.create_dataset('targets', (0, 1), maxshape=(None, 1), dtype='float32')
            
        logger.info(f"Created storage file at {storage_path}")
        return storage_path
        
    def load_embedding_model(self):
        """Load the embedding model and tokenizer"""
        if self.tokenizer is None or self.embedding_model is None:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
            self.embedding_model = AutoModel.from_pretrained(self.embedding_model_name)
        
    def clean_text(self, text: str) -> str:
        """Clean the discharge summary text"""
        if not isinstance(text, str):
            return ""
        
        # Truncate if too long
        text = text[:self.max_text_length] if len(text) > self.max_text_length else text
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip().lower()
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text using the pretrained model"""
        if not self.tokenizer or not self.embedding_model:
            self.load_embedding_model()
            
        # Ensure we have text to embed
        if not text or len(text) < 5:
            # Return zeros if no valid text
            return np.zeros(self.embedding_dim, dtype=np.float32)
        
        # Tokenize and get embeddings
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=self.max_text_length
        )
        
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
            
        # Use the [CLS] token embedding as the document embedding
        embedding = outputs.last_hidden_state[:, 0, :].numpy()[0].astype(np.float32)
        return embedding
    
    def _resize_dataset(self, h5file, dataset_name, new_size, new_shape=None):
        """Resize a dataset in the HDF5 file"""
        current_shape = h5file[dataset_name].shape
        if new_shape is None:
            new_shape = (new_size, current_shape[1])
        h5file[dataset_name].resize(new_shape)
    
    def _append_to_storage(self, storage_path, data_dict, start_idx):
        """Append processed data to storage"""
        with h5py.File(storage_path, 'a') as f:
            end_idx = start_idx + len(data_dict['text_embeddings'])
            
            # Resize datasets if needed
            if end_idx > f['text_embeddings'].shape[0]:
                self._resize_dataset(f, 'text_embeddings', end_idx)
                self._resize_dataset(f, 'demographics', end_idx)
                self._resize_dataset(f, 'diagnoses', end_idx)
                self._resize_dataset(f, 'targets', end_idx)
            
            # Store data with explicit type conversion
            f['text_embeddings'][start_idx:end_idx] = data_dict['text_embeddings'].astype(np.float32)
            f['demographics'][start_idx:end_idx] = data_dict['demographics'].astype(np.float32)
            f['diagnoses'][start_idx:end_idx] = data_dict['diagnoses'].astype(np.float32)
            
            # Make sure targets are properly shaped and typed
            targets_reshaped = data_dict['targets'].reshape(-1, 1).astype(np.float32)
            f['targets'][start_idx:end_idx] = targets_reshaped
    
    def _initialize_encoders(self, demographics_df, diagnoses_df):
        """Initialize and fit encoders on the first chunk of data"""
        logger.info("Initializing encoders with first data chunk")
        
        # Fit demographics encoder
        for col in demographics_df.columns:
            if demographics_df[col].dtype == 'object':
                demographics_df[col].fillna('unknown', inplace=True)
            else:
                demographics_df[col].fillna(demographics_df[col].median(), inplace=True)
                
        self.demographics_encoder.fit(demographics_df)
        
        # Create a list of all unique diagnoses
        all_diagnoses = []
        for idx, row in diagnoses_df.iterrows():
            icd_codes = str(row['diagnoses_icd']).split(',') if pd.notna(row['diagnoses_icd']) else []
            all_diagnoses.extend([code.strip() for code in icd_codes if code.strip()])
        
        unique_diagnoses = list(set(all_diagnoses))
        self.diagnoses_encoder = {diag: i for i, diag in enumerate(unique_diagnoses)}
        
    def _update_storage_shapes(self, storage_path, demo_shape, diag_shape):
        """Update shapes of datasets after processing first chunk"""
        with h5py.File(storage_path, 'a') as f:
            # Resize demographics and diagnoses datasets with correct feature dimensions
            current_size = f['demographics'].shape[0]
            f['demographics'].resize((current_size, demo_shape[1]))
            f['diagnoses'].resize((current_size, diag_shape[1]))
    
    def _process_time_value(self, time_value):
        """Process time_until_next_admission to handle string values"""
        if pd.isna(time_value):
            return -1.0
        
        # Handle string values
        if isinstance(time_value, str):
            # Handle strings like '180+'
            if time_value.endswith('+'):
                try:
                    # Extract the numeric part and add 1 to indicate it's a minimum
                    return float(time_value.rstrip('+')) + 1.0
                except ValueError:
                    # If we can't convert, use a default value
                    logger.warning(f"Could not convert time value: {time_value}, using default")
                    return 180.0
            
            # Handle other string formats if needed
            try:
                return float(time_value)
            except ValueError:
                logger.warning(f"Could not convert time value: {time_value}, using default")
                return -1.0
        
        # If it's already a number, just return it
        return float(time_value)
    
    def process_chunk(self, chunk: pd.DataFrame, storage_path: str, is_first_chunk: bool = False):
        """Process a chunk of data and append to storage"""
        try:
            # Extract necessary columns
            demographics_df = chunk[self.demographics_columns].copy()
            diagnoses_df = chunk[['diagnoses_icd', 'diagnoses_long_title']].copy()
            
            # Initialize encoders on first chunk
            if is_first_chunk:
                self._initialize_encoders(demographics_df, diagnoses_df)
            
            # Process demographics
            for col in demographics_df.columns:
                if demographics_df[col].dtype == 'object':
                    demographics_df[col].fillna('unknown', inplace=True)
                else:
                    demographics_df[col].fillna(demographics_df[col].median(), inplace=True)
                    
            encoded_demographics = self.demographics_encoder.transform(demographics_df).astype(np.float32)
            
            # Process diagnoses
            processed_diagnoses = []
            for idx, row in diagnoses_df.iterrows():
                # Initialize zero vector
                patient_diagnoses = np.zeros(len(self.diagnoses_encoder), dtype=np.float32)
                
                # Fill in diagnoses that exist
                if pd.notna(row['diagnoses_icd']):
                    icd_codes = str(row['diagnoses_icd']).split(',')
                    for code in icd_codes:
                        code = code.strip()
                        if code in self.diagnoses_encoder:
                            patient_diagnoses[self.diagnoses_encoder[code]] = 1.0
                
                processed_diagnoses.append(patient_diagnoses)
                
            processed_diagnoses = np.array(processed_diagnoses, dtype=np.float32)
            
            # Process text embeddings
            text_embeddings = []
            for text in chunk['text']:
                cleaned_text = self.clean_text(text)
                embedding = self.get_text_embedding(cleaned_text)
                text_embeddings.append(embedding)
                
            text_embeddings = np.array(text_embeddings, dtype=np.float32)
            
            # Process targets - handle string values like '180+'
            targets = []
            for time_value in chunk['time_until_next_admission']:
                targets.append(self._process_time_value(time_value))
            targets = np.array(targets, dtype=np.float32)
            
            # Update storage
            data_dict = {
                'text_embeddings': text_embeddings,
                'demographics': encoded_demographics,
                'diagnoses': processed_diagnoses,
                'targets': targets
            }
            
            # If first chunk, we need to update dataset shapes
            if is_first_chunk:
                with h5py.File(storage_path, 'a') as f:
                    self._resize_dataset(f, 'demographics', f['demographics'].shape[0], 
                                       (f['demographics'].shape[0], encoded_demographics.shape[1]))
                    self._resize_dataset(f, 'diagnoses', f['diagnoses'].shape[0],
                                       (f['diagnoses'].shape[0], processed_diagnoses.shape[1]))
            
            # Append data to storage
            self._append_to_storage(storage_path, data_dict, self.current_index)
            
            # Create patient indices
            chunk_indices = []
            for idx, row in chunk.iterrows():
                patient_idx = PatientDataIndex(
                    patient_id=str(row['subject_id']) if pd.notna(row['subject_id']) else "unknown",
                    hadm_id=str(row['hadm_id']) if pd.notna(row['hadm_id']) else "unknown",
                    data_index=self.current_index + idx - chunk.index[0],  # Calculate correct index
                    split=None,  # Will be assigned later
                    time_until_next_admission=self._process_time_value(row['time_until_next_admission'])
                )
                chunk_indices.append(patient_idx)
                
            # Update current index
            self.current_index += len(chunk)
            
            return chunk_indices
            
        except Exception as e:
            logger.error(f"Error processing chunk: {str(e)}")
            # Print more detailed debug info
            import traceback
            logger.error(traceback.format_exc())
            raise
        
    def preprocess_data(self, csv_path: str):
        """Process data in chunks and store to disk"""
        logger.info(f"Processing data from {csv_path} in chunks of {self.chunk_size}")
        
        # Create storage
        storage_path = self.create_storage()
        
        # Process data in chunks
        chunk_iter = pd.read_csv(csv_path, chunksize=self.chunk_size, na_values=['', 'NULL', 'null', 'NA'])
        
        is_first_chunk = True
        all_indices = []
        
        for i, chunk in enumerate(tqdm(chunk_iter, desc="Processing chunks")):
            logger.info(f"Processing chunk {i+1}")
            
            # Process chunk
            chunk_indices = self.process_chunk(chunk, storage_path, is_first_chunk)
            all_indices.extend(chunk_indices)
            
            is_first_chunk = False
            
            # For testing, process only a few chunks
            # if i >= 2:  # Uncomment this to process only 3 chunks during testing
            #    break
        
        # Split indices into train/val/test
        train_indices, test_indices = train_test_split(
            all_indices, test_size=0.2, random_state=self.random_state
        )
        
        train_indices, val_indices = train_test_split(
            train_indices, test_size=0.25, random_state=self.random_state
        )
        
        # Assign splits
        for idx in train_indices:
            idx.split = 'train'
        for idx in val_indices:
            idx.split = 'val'
        for idx in test_indices:
            idx.split = 'test'
            
        # Save all indices
        self.patient_indices = train_indices + val_indices + test_indices
        
        # Save indices to file
        self._save_indices()
        
        # Save processor data
        self._save_processor_data()
        
        # Return split counts
        return {
            'train': len(train_indices),
            'val': len(val_indices),
            'test': len(test_indices),
            'total': len(self.patient_indices)
        }
    
    def _save_indices(self):
        """Save patient indices to disk"""
        indices_path = os.path.join(self.output_dir, 'patient_indices.pkl')
        with open(indices_path, 'wb') as f:
            pickle.dump(self.patient_indices, f)
        logger.info(f"Saved {len(self.patient_indices)} patient indices to {indices_path}")
    
    def _save_processor_data(self):
        """Save processor data to disk"""
        processor_data = {
            'demographics_encoder': self.demographics_encoder,
            'diagnoses_encoder': self.diagnoses_encoder,
            'demographics_columns': self.demographics_columns,
            'embedding_model_name': self.embedding_model_name,
            'embedding_dim': self.embedding_dim
        }
        
        processor_path = os.path.join(self.output_dir, 'processor_data.pkl')
        with open(processor_path, 'wb') as f:
            pickle.dump(processor_data, f)
        logger.info(f"Saved processor data to {processor_path}")
    
    @classmethod
    def load_processor(cls, output_dir: str) -> 'StreamingDataProcessor':
        """Load a saved processor"""
        processor_path = os.path.join(output_dir, 'processor_data.pkl')
        with open(processor_path, 'rb') as f:
            processor_data = pickle.load(f)
            
        processor = cls(
            output_dir=output_dir,
            embedding_model_name=processor_data['embedding_model_name'],
            embedding_dim=processor_data['embedding_dim'],
            demographics_columns=processor_data['demographics_columns']
        )
        
        processor.demographics_encoder = processor_data['demographics_encoder']
        processor.diagnoses_encoder = processor_data['diagnoses_encoder']
        
        # Load indices
        indices_path = os.path.join(output_dir, 'patient_indices.pkl')
        with open(indices_path, 'rb') as f:
            processor.patient_indices = pickle.load(f)
            
        return processor
    
    def get_split_indices(self, split: str) -> List[PatientDataIndex]:
        """Get indices for a specific split"""
        return [idx for idx in self.patient_indices if idx.split == split]
    
    def get_feature_dims(self) -> Dict[str, int]:
        """Get dimensions of different feature types"""
        # Open the storage file to check dimensions
        storage_path = os.path.join(self.output_dir, 'processed_data.h5')
        with h5py.File(storage_path, 'r') as f:
            dims = {
                'text_embedding': f['text_embeddings'].shape[1],
                'demographics': f['demographics'].shape[1],
                'diagnoses': f['diagnoses'].shape[1]
            }
        return dims


class StreamingPatientDataset:
    """Dataset that loads data on-demand from storage"""
    def __init__(self, data_path: str, indices: List[PatientDataIndex], batch_size: int = 32):
        """
        Initialize the dataset
        
        Args:
            data_path: Path to the HDF5 file with processed data
            indices: List of PatientDataIndex objects for this split
            batch_size: Batch size for efficient reading
        """
        self.data_path = data_path
        self.indices = indices
        self.batch_size = batch_size
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """Get a single item by index"""
        patient_idx = self.indices[idx]
        data_idx = patient_idx.data_index
        
        with h5py.File(self.data_path, 'r') as f:
            return {
                'text_embedding': torch.tensor(f['text_embeddings'][data_idx], dtype=torch.float32),
                'demographics': torch.tensor(f['demographics'][data_idx], dtype=torch.float32),
                'diagnoses': torch.tensor(f['diagnoses'][data_idx], dtype=torch.float32),
                'target': torch.tensor(f['targets'][data_idx][0], dtype=torch.float32)
            }
    
    def get_batch_indices(self, indices):
        # Get the data indices for the patient indices
        data_indices = [self.indices[i].data_index for i in indices]
        
        # Create a mapping from original position to sorted position
        sorted_indices_map = {idx: i for i, idx in enumerate(sorted(data_indices))}
        
        # Sort indices for HDF5 (required)
        sorted_indices = sorted(data_indices)
        
        with h5py.File(self.data_path, 'r') as f:
            # Get data using sorted indices
            batch_data = {
                'text_embedding': f['text_embeddings'][sorted_indices],
                'demographics': f['demographics'][sorted_indices],
                'diagnoses': f['diagnoses'][sorted_indices],
                'targets': f['targets'][sorted_indices].flatten()
            }
            
            # Convert to tensors and return in original order
            return {
                'text_embedding': torch.tensor(batch_data['text_embedding'], dtype=torch.float32),
                'demographics': torch.tensor(batch_data['demographics'], dtype=torch.float32),
                'diagnoses': torch.tensor(batch_data['diagnoses'], dtype=torch.float32),
                'target': torch.tensor(batch_data['targets'], dtype=torch.float32)
            }


class StreamingDataLoader:
    """Custom data loader that streams data from disk in batches"""
    def __init__(self, dataset: StreamingPatientDataset, batch_size: int = 32, shuffle: bool = False):
        """
        Initialize the data loader
        
        Args:
            dataset: The StreamingPatientDataset
            batch_size: Batch size
            shuffle: Whether to shuffle the data
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        # Shuffle indices if needed
        if self.shuffle:
            np.random.shuffle(self.indices)
            
        # Create batches
        batches = [self.indices[i:i+self.batch_size] 
                  for i in range(0, len(self.indices), self.batch_size)]
        
        for batch_indices in batches:
            try:
                yield self.dataset.get_batch_indices(batch_indices)
            except Exception as e:
                logger.error(f"Error loading batch: {str(e)}")
                # Skip problematic batches
                continue


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process large CSV data in streaming mode")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV data file")
    parser.add_argument("--output_dir", type=str, default="streamed_data", help="Directory to save processed data")
    parser.add_argument("--chunk_size", type=int, default=500, help="Number of rows to process at once")
    parser.add_argument("--embedding_model", type=str, default="emilyalsentzer/Bio_ClinicalBERT", 
                        help="Pre-trained model for text embeddings")
    parser.add_argument("--max_text_length", type=int, default=256,
                        help="Maximum length of text to process (for memory efficiency)")
    args = parser.parse_args()
    
    # Process data
    processor = StreamingDataProcessor(
        output_dir=args.output_dir,
        embedding_model_name=args.embedding_model,
        chunk_size=args.chunk_size,
        max_text_length=args.max_text_length
    )
    
    split_counts = processor.preprocess_data(args.csv_path)
    logger.info(f"Processed data split counts: {split_counts}")