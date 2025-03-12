import numpy as np
import h5py
import pickle
import os

def extract_embeddings_in_original_order(data_dir, output_path=None):
    """
    Extract text embeddings from processed data and save in original patient order.
    
    Args:
        data_dir: Directory containing processed data files
        output_path: Path to save the numpy array (default: data_dir/embeddings.npy)
    
    Returns:
        Path to the saved embeddings file
    """
    # Load patient indices
    with open(os.path.join(data_dir, 'patient_indices.pkl'), 'rb') as f:
        patient_indices = pickle.load(f)
    
    # Sort patient indices by data_index to get original order
    sorted_indices = sorted(patient_indices, key=lambda x: x.data_index)
    
    # Get data indices in sorted order
    data_indices = [idx.data_index for idx in sorted_indices]
    
    # Load embeddings from HDF5 file
    with h5py.File(os.path.join(data_dir, 'processed_data.h5'), 'r') as f:
        # Get embedding dimension
        embedding_dim = f['text_embeddings'].shape[1]
        
        # Create array for all embeddings
        embeddings = np.zeros((len(data_indices), embedding_dim), dtype=np.float32)
        
        # Fill array with embeddings in correct order
        print(f"Extracting {len(data_indices)} embeddings...")
        
        # Process in batches to avoid memory issues
        batch_size = 1000
        for i in range(0, len(data_indices), batch_size):
            batch_indices = data_indices[i:i+batch_size]
            batch_embeddings = f['text_embeddings'][batch_indices]
            embeddings[i:i+len(batch_indices)] = batch_embeddings
    
    # Create patient ID mapping
    patient_id_map = {idx.data_index: idx.patient_id for idx in patient_indices}
    
    # Save the embeddings
    if output_path is None:
        output_path = os.path.join(data_dir, 'embeddings.npy')
    
    np.save(output_path, embeddings)
    print(f"Saved embeddings to {output_path}")
    
    # Also save patient IDs in same order
    patient_ids = [patient_id_map[idx] for idx in data_indices]
    patient_ids_path = os.path.join(data_dir, 'patient_ids.npy')
    np.save(patient_ids_path, patient_ids)
    print(f"Saved corresponding patient IDs to {patient_ids_path}")
    
    return output_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Extract embeddings in original patient order')
    parser.add_argument('--data_dir', type=str, default='gpu_processed_data',
                        help='Directory containing processed data files')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Path to save the numpy array (default: data_dir/embeddings.npy)')
    args = parser.parse_args()
    
    extract_embeddings_in_original_order(args.data_dir, args.output_path)