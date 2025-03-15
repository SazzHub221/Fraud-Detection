import os
import numpy as np
import pandas as pd
import faiss
import pickle
import logging
from typing import List, Dict, Tuple, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VectorDatabase:
    def __init__(self, dimension=384, index_type='flat', metric='cosine'):
        """
        Initialize the vector database.
        
        Args:
            dimension: Dimension of the embedding vectors
            index_type: Type of FAISS index ('flat', 'ivf', 'pq', 'hnsw')
            metric: Distance metric ('cosine' or 'euclidean')
        """
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        self.index = None
        self.transaction_ids = []
        self.is_trained = False
    
    def create_index(self):
        """Create the FAISS index based on specified parameters."""
        if self.index is not None:
            logger.warning("Index already exists. Creating a new one.")
        
        # Choose the appropriate index type
        if self.metric == 'cosine':
            # Normalize vectors for cosine similarity
            index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
            logger.info(f"Created cosine similarity index with dimension {self.dimension}")
        else:
            # Euclidean distance index
            if self.index_type == 'flat':
                index = faiss.IndexFlatL2(self.dimension)
                logger.info(f"Created flat L2 index with dimension {self.dimension}")
            elif self.index_type == 'ivf':
                # IVF index requires training
                quantizer = faiss.IndexFlatL2(self.dimension)
                nlist = 100  # Number of clusters
                index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_L2)
                logger.info(f"Created IVF index with dimension {self.dimension}, nlist={nlist}")
            elif self.index_type == 'pq':
                # Product quantization index
                m = 8  # Number of subquantizers
                nbits = 8  # Bits per subquantizer
                index = faiss.IndexPQ(self.dimension, m, nbits)
                logger.info(f"Created PQ index with dimension {self.dimension}, m={m}, nbits={nbits}")
            elif self.index_type == 'hnsw':
                # HNSW index
                m = 16  # Connections per node
                index = faiss.IndexHNSWFlat(self.dimension, m, faiss.METRIC_L2)
                logger.info(f"Created HNSW index with dimension {self.dimension}, m={m}")
            else:
                raise ValueError(f"Unknown index_type: {self.index_type}")
        
        self.index = index
        self.is_trained = False
    
    def normalize_vectors(self, vectors):
        """Normalize vectors for cosine similarity."""
        if self.metric == 'cosine':
            # L2 normalization
            faiss.normalize_L2(vectors)
        return vectors
    
    def add_embeddings(self, embeddings, transaction_ids=None):
        """
        Add embeddings to the index.
        
        Args:
            embeddings: Numpy array of embeddings
            transaction_ids: List of transaction IDs corresponding to embeddings
        """
        if self.index is None:
            self.dimension = embeddings.shape[1]
            self.create_index()
        
        # Convert to numpy array and proper type
        embeddings = np.array(embeddings).astype('float32')
        
        # Normalize vectors if using cosine similarity
        embeddings = self.normalize_vectors(embeddings)
        
        # Train the index if needed
        if not self.is_trained and self.index_type in ['ivf', 'pq']:
            logger.info(f"Training {self.index_type} index...")
            self.index.train(embeddings)
            self.is_trained = True
        
        # Add vectors to the index
        self.index.add(embeddings)
        
        # Store transaction IDs
        if transaction_ids is not None:
            self.transaction_ids.extend(transaction_ids)
        else:
            # If no IDs provided, use sequential IDs
            start_idx = len(self.transaction_ids)
            self.transaction_ids.extend(list(range(start_idx, start_idx + len(embeddings))))
        
        logger.info(f"Added {len(embeddings)} embeddings to the index. Total: {len(self.transaction_ids)}")
    
    def search(self, query_embedding, k=10):
        """
        Search for similar embeddings.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of nearest neighbors to return
            
        Returns:
            Tuple of (distances, transaction_ids)
        """
        if self.index is None:
            raise ValueError("Index not initialized. Call create_index() first.")
        
        # Convert to numpy array and proper type
        query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
        
        # Normalize vector if using cosine similarity
        query_embedding = self.normalize_vectors(query_embedding)
        
        # Search the index
        if self.index_type in ['ivf'] and not self.is_trained:
            logger.warning("IVF index not trained. Search results may be inaccurate.")
        
        distances, indices = self.index.search(query_embedding, k)
        
        # Map indices to transaction IDs
        result_ids = []
        for idx_list in indices:
            id_list = [self.transaction_ids[i] if i < len(self.transaction_ids) else None for i in idx_list]
            result_ids.append(id_list)
        
        return distances, result_ids
    
    def batch_search(self, query_embeddings, k=10):
        """
        Search for similar embeddings in batch.
        
        Args:
            query_embeddings: Batch of query embedding vectors
            k: Number of nearest neighbors to return
            
        Returns:
            Tuple of (distances, transaction_ids)
        """
        if self.index is None:
            raise ValueError("Index not initialized. Call create_index() first.")
        
        # Convert to numpy array and proper type
        query_embeddings = np.array(query_embeddings).astype('float32')
        
        # Normalize vectors if using cosine similarity
        query_embeddings = self.normalize_vectors(query_embeddings)
        
        # Search the index
        distances, indices = self.index.search(query_embeddings, k)
        
        # Map indices to transaction IDs
        result_ids = []
        for idx_list in indices:
            id_list = [self.transaction_ids[i] if i < len(self.transaction_ids) and i >= 0 else None for i in idx_list]
            result_ids.append(id_list)
        
        return distances, result_ids
    
    def save(self, directory):
        """
        Save the index and metadata to disk.
        
        Args:
            directory: Directory to save the index
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save the FAISS index
        index_path = os.path.join(directory, "faiss_index.bin")
        faiss.write_index(self.index, index_path)
        
        # Save metadata
        metadata = {
            'dimension': self.dimension,
            'index_type': self.index_type,
            'metric': self.metric,
            'transaction_ids': self.transaction_ids,
            'is_trained': self.is_trained
        }
        metadata_path = os.path.join(directory, "metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Saved index to {index_path} and metadata to {metadata_path}")
    
    @classmethod
    def load(cls, directory):
        """
        Load the index and metadata from disk.
        
        Args:
            directory: Directory containing the saved index
            
        Returns:
            VectorDatabase instance
        """
        # Load metadata
        metadata_path = os.path.join(directory, "metadata.pkl")
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Create instance with metadata
        instance = cls(
            dimension=metadata['dimension'],
            index_type=metadata['index_type'],
            metric=metadata['metric']
        )
        
        # Load the index
        index_path = os.path.join(directory, "faiss_index.bin")
        instance.index = faiss.read_index(index_path)
        
        # Restore metadata
        instance.transaction_ids = metadata['transaction_ids']
        instance.is_trained = metadata['is_trained']
        
        logger.info(f"Loaded index from {index_path} with {len(instance.transaction_ids)} transaction IDs")
        return instance

def build_vector_database(embeddings_df, transaction_id_col='transaction_id', save_dir=None):
    """
    Build a vector database from a DataFrame with embeddings.
    
    Args:
        embeddings_df: DataFrame with 'embedding' column and transaction IDs
        transaction_id_col: Column name for transaction IDs
        save_dir: Directory to save the index (optional)
        
    Returns:
        VectorDatabase instance
    """
    # Extract embeddings and transaction IDs
    if isinstance(embeddings_df['embedding'][0], list):
        embeddings = np.array(embeddings_df['embedding'].tolist()).astype('float32')
    else:
        embeddings = np.vstack(embeddings_df['embedding'].values).astype('float32')
    
    transaction_ids = embeddings_df[transaction_id_col].tolist()
    
    # Create and populate the vector database
    dimension = embeddings.shape[1]
    db = VectorDatabase(dimension=dimension)
    db.create_index()
    db.add_embeddings(embeddings, transaction_ids)
    
    # Save if requested
    if save_dir:
        db.save(save_dir)
    
    return db

if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from src.data.embeddings import load_embeddings
    
    # Load embeddings
    embeddings_path = "data/embeddings/transaction_embeddings.csv"
    embeddings_data_path = embeddings_path.replace('.csv', '_embeddings.pkl')
    
    if os.path.exists(embeddings_path) and os.path.exists(embeddings_data_path):
        df = load_embeddings(embeddings_path, embeddings_data_path)
        
        # Create vector database
        os.makedirs("data/vector_db", exist_ok=True)
        db = build_vector_database(df, transaction_id_col='transaction_id', save_dir="data/vector_db")
        
        # Test search
        test_embedding = df['embedding'][0]
        distances, ids = db.search(test_embedding, k=5)
        
        logger.info(f"Search results: {ids}")
        logger.info(f"Distances: {distances}")
    else:
        logger.error(f"Embeddings not found at {embeddings_path}. Run embeddings.py first.")