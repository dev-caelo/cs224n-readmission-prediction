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
import time
import signal
import sys

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

class GPUDataProcessor:
    def __init__(
        self, 
        output_dir: str,
        embedding_model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        embedding_dim: int = 768,
        demographics_columns: List[str] = None,
        chunk_size: int = 5000,  # Increased for GPU processing
        max_text_length: int = 256,
        embedding_batch_size: int = 64,  # Batch size for GPU processing
        random_state: int = 42,
        save_every: int = 5  # Save intermediate results every N chunks
    ):
        """
        Initialize the GPU-accelerated data processor
        
        Args:
            output_dir: Directory to store processed data
            embedding_model_name: Pre-trained model to use for text embeddings
            embedding_dim: Dimension of the embeddings
            demographics_columns: Columns to use as demographic features
            chunk_size: Number of rows to process at once
            max_text_length: Maximum length for text to process
            embedding_batch_size: Batch size for embedding generation on GPU
            random_state: Random state for reproducibility
            save_every: Save intermediate results every N chunks
        """
        self.output_dir = output_dir
        self.embedding_model_name = embedding_model_name
        self.embedding_dim = embedding_dim
        self.chunk_size = chunk_size
        self.max_text_length = max_text_length
        self.embedding_batch_size = embedding_batch_size
        self.random_state = random_state
        self.save_every = save_every
        
        # Setup GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
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
        
        # Flag for graceful shutdown
        self.interrupted = False
        
        # Set up signal handler for graceful interruption
        signal.signal(signal.SIGINT, self._handle_interrupt)
        
    def _handle_interrupt(self, signum, frame):
        """Handle keyboard interrupt by setting flag for graceful shutdown"""
        logger.info("Interrupt received. Will complete current chunk and save progress...")
        self.interrupted = True
        
    def create_storage(self):
        """Create HDF5 storage for processed data or open existing one"""
        storage_path = os.path.join(self.output_dir, 'processed_data.h5')
        
        # Check if file exists to determine if we're resuming
        if os.path.exists(storage_path):
            logger.info(f"Found existing storage file at {storage_path}, will append to it")
            with h5py.File(storage_path, 'r') as f:
                # Check if we have processed data
                if 'text_embeddings' in f and f['text_embeddings'].shape[0] > 0:
                    self.current_index = f['text_embeddings'].shape[0]
                    logger.info(f"Resuming from index {self.current_index}")
                    
                    # Also load patient indices if they exist
                    indices_path = os.path.join(self.output_dir, 'patient_indices.pkl')
                    if os.path.exists(indices_path):
                        with open(indices_path, 'rb') as idx_file:
                            self.patient_indices = pickle.load(idx_file)
                            logger.info(f"Loaded {len(self.patient_indices)} existing patient indices")
                    
                    # Load processor data if it exists
                    processor_path = os.path.join(self.output_dir, 'processor_data.pkl')
                    if os.path.exists(processor_path):
                        with open(processor_path, 'rb') as proc_file:
                            processor_data = pickle.load(proc_file)
                            self.demographics_encoder = processor_data['demographics_encoder']
                            self.diagnoses_encoder = processor_data['diagnoses_encoder']
                            logger.info("Loaded existing encoders")
        else:
            # Create new file
            with h5py.File(storage_path, 'w') as f:
                # Create datasets with explicit data types
                f.create_dataset('text_embeddings', (0, self.embedding_dim), 
                                maxshape=(None, self.embedding_dim), dtype='float32',
                                chunks=(min(self.chunk_size, 1000), self.embedding_dim))
                
                # We'll determine these sizes after processing the first chunk
                f.create_dataset('demographics', (0, 1), maxshape=(None, None), dtype='float32')
                f.create_dataset('diagnoses', (0, 1), maxshape=(None, None), dtype='float32')
                f.create_dataset('targets', (0, 1), maxshape=(None, 1), dtype='float32')
                
            logger.info(f"Created new storage file at {storage_path}")
            
        return storage_path
        
    def load_embedding_model(self):
        """Load the embedding model and tokenizer and move to GPU if available"""
        if self.tokenizer is None or self.embedding_model is None:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
            self.embedding_model = AutoModel.from_pretrained(self.embedding_model_name)
            self.embedding_model = self.embedding_model.to(self.device)
            self.embedding_model.eval()  # Set model to evaluation mode
            logger.info(f"Model loaded and moved to {self.device}")
        
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
    
    def get_batch_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a batch of texts using GPU acceleration"""
        if not self.tokenizer or not self.embedding_model:
            self.load_embedding_model()
            
        # Clean texts and handle empty ones
        cleaned_texts = [self.clean_text(text) for text in texts]
        
        # Create a mask for empty texts to handle them separately
        empty_mask = [len(text) < 5 for text in cleaned_texts]
        
        # If all texts are empty, return zeros
        if all(empty_mask):
            return np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
        
        # Filter out empty texts for processing
        valid_indices = [i for i, is_empty in enumerate(empty_mask) if not is_empty]
        valid_texts = [cleaned_texts[i] for i in valid_indices]
        
        # Tokenize and get embeddings
        inputs = self.tokenizer(
            valid_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_text_length
        )
        
        # Move inputs to GPU
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
        
        # Get CLS token embeddings and move to CPU
        valid_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)
        
        # Create result array with zeros for empty texts
        result = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
        
        # Fill in embeddings for valid texts
        for i, orig_idx in enumerate(valid_indices):
            result[orig_idx] = valid_embeddings[i]
        
        return result
    
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
                demographics_df[col] = demographics_df[col].fillna('unknown')
            else:
                demographics_df[col] = demographics_df[col].fillna(demographics_df[col].median())
                
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
            if is_first_chunk and self.diagnoses_encoder is None:
                self._initialize_encoders(demographics_df, diagnoses_df)
            
            # Process demographics
            for col in demographics_df.columns:
                if demographics_df[col].dtype == 'object':
                    demographics_df[col] = demographics_df[col].fillna('unknown')
                else:
                    demographics_df[col] = demographics_df[col].fillna(demographics_df[col].median())
                    
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
            
            # Process text embeddings in batches using GPU
            texts = chunk['text'].tolist()
            text_embeddings = []
            
            # Process in smaller batches for GPU efficiency
            for i in range(0, len(texts), self.embedding_batch_size):
                batch_texts = texts[i:i + self.embedding_batch_size]
                batch_embeddings = self.get_batch_embeddings(batch_texts)
                text_embeddings.append(batch_embeddings)
                
                # Log GPU memory usage
                if torch.cuda.is_available() and i % (self.embedding_batch_size * 10) == 0 and i > 0:
                    logger.info(f"GPU memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            
            # Concatenate batches
            text_embeddings = np.vstack(text_embeddings)
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
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
            if is_first_chunk and not os.path.exists(os.path.join(self.output_dir, 'processor_data.pkl')):
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
            # Clean up GPU memory on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Print more detailed debug info
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def _save_intermediate_state(self, all_indices):
        """Save current state to enable resuming processing later"""
        logger.info("Saving intermediate state...")
        
        # Create temporary train/val/test split with current indices
        if len(all_indices) >= 3:  # Need at least 3 samples for a minimal split
            temp_indices = all_indices.copy()
            train_indices, test_indices = train_test_split(
                temp_indices, test_size=0.2, random_state=self.random_state
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
        
        logger.info(f"Intermediate state saved with {len(all_indices)} samples processed")
        
    def preprocess_data(self, csv_path: str, skip_rows: int = 0):
        """Process data in chunks and store to disk"""
        logger.info(f"Processing data from {csv_path} in chunks of {self.chunk_size}")
        
        # Create or open storage
        storage_path = self.create_storage()
        
        # Determine if we're resuming from an interruption
        resume_from = 0
        if self.current_index > 0:
            # Calculate how many rows to skip
            resume_from = (self.current_index // self.chunk_size) * self.chunk_size
            logger.info(f"Resuming processing from row {resume_from}")
        
        # Add any additional rows to skip
        total_skip_rows = resume_from + skip_rows
        
        # Process data in chunks
        chunk_iter = pd.read_csv(csv_path, 
                                chunksize=self.chunk_size, 
                                na_values=['', 'NULL', 'null', 'NA'],
                                skiprows=range(1, total_skip_rows + 1) if total_skip_rows > 0 else None,
                                header=0 if total_skip_rows == 0 else None)
        
        is_first_chunk = len(self.patient_indices) == 0
        all_indices = self.patient_indices.copy()  # Start with any existing indices
        
        # Track processing time for ETA calculation
        start_time = time.time()
        chunks_processed = 0
        
        for i, chunk in enumerate(tqdm(chunk_iter, desc="Processing chunks")):
            # Recreate header if we're resuming
            if total_skip_rows > 0 and i == 0:
                # Rename columns to match the original header
                chunk.columns = pd.read_csv(csv_path, nrows=1).columns
            
            chunks_processed += 1
            current_chunk = i + (total_skip_rows // self.chunk_size) + 1
            logger.info(f"Processing chunk {current_chunk}")
            
            # Process chunk
            chunk_indices = self.process_chunk(chunk, storage_path, is_first_chunk)
            all_indices.extend(chunk_indices)
            
            is_first_chunk = False
            
            # Calculate ETA
            elapsed_time = time.time() - start_time
            time_per_chunk = elapsed_time / chunks_processed
            estimated_total_chunks = 277746 // self.chunk_size  # Based on dataset size
            estimated_remaining_chunks = estimated_total_chunks - current_chunk
            estimated_remaining_time = estimated_remaining_chunks * time_per_chunk
            
            logger.info(f"Processed {current_chunk}/{estimated_total_chunks} chunks " +
                      f"(~{current_chunk*self.chunk_size} rows). " +
                      f"Estimated time remaining: {estimated_remaining_time/3600:.1f} hours")
            
            # Save intermediate state periodically
            if current_chunk % self.save_every == 0:
                self._save_intermediate_state(all_indices)
            
            # Check for interruption
            if self.interrupted:
                logger.info("Interrupted by user. Saving current state and exiting...")
                self._save_intermediate_state(all_indices)
                logger.info(f"Processed {current_chunk} chunks before interruption")
                return {
                    'train': len([idx for idx in all_indices if idx.split == 'train']),
                    'val': len([idx for idx in all_indices if idx.split == 'val']),
                    'test': len([idx for idx in all_indices if idx.split == 'test']),
                    'total': len(all_indices),
                    'status': 'interrupted'
                }
        
        # After all chunks, create final train/val/test split
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
            'total': len(self.patient_indices),
            'status': 'complete'
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
    def load_processor(cls, output_dir: str) -> 'GPUDataProcessor':
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


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process data using GPU acceleration")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV data file")
    parser.add_argument("--output_dir", type=str, default="gpu_processed_data", help="Directory to save processed data")
    parser.add_argument("--chunk_size", type=int, default=5000, help="Number of rows to process at once")
    parser.add_argument("--embedding_model", type=str, default="emilyalsentzer/Bio_ClinicalBERT", 
                        help="Pre-trained model for text embeddings")
    parser.add_argument("--max_text_length", type=int, default=256,
                        help="Maximum length of text to process (for memory efficiency)")
    parser.add_argument("--embedding_batch_size", type=int, default=64,
                        help="Batch size for embedding generation")
    parser.add_argument("--save_every", type=int, default=2,
                        help="Save intermediate results every N chunks")
    args = parser.parse_args()
    
    # Process data
    processor = GPUDataProcessor(
        output_dir=args.output_dir,
        embedding_model_name=args.embedding_model,
        chunk_size=args.chunk_size,
        max_text_length=args.max_text_length,
        embedding_batch_size=args.embedding_batch_size,
        save_every=args.save_every
    )
    
    split_counts = processor.preprocess_data(args.csv_path)
    logger.info(f"Processed data split counts: {split_counts}")
