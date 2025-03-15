import os
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import logging
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
        """
        Initialize the embedding generator with a transformer model.
        
        Args:
            model_name: Name of the sentence transformer model to use
            device: Device to run the model on ('cpu' or 'cuda')
        """
        # Determine device automatically if not specified
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        logger.info(f"Loading embedding model {model_name} on {device}...")
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
    
    def generate_embeddings(self, texts, batch_size=32, show_progress=True):
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to generate embeddings for
            batch_size: Batch size for embedding generation
            show_progress: Whether to show a progress bar
            
        Returns:
            Numpy array of embeddings
        """
        if not texts:
            return np.array([])
        
        logger.info(f"Generating embeddings for {len(texts)} texts with batch size {batch_size}...")
        
        # Handle progress bar display
        if show_progress:
            embeddings = self.model.encode(
                texts, 
                batch_size=batch_size, 
                show_progress_bar=True,
                convert_to_numpy=True
            )
        else:
            embeddings = self.model.encode(
                texts, 
                batch_size=batch_size, 
                show_progress_bar=False,
                convert_to_numpy=True
            )
        
        logger.info(f"Generated {len(embeddings)} embeddings with shape {embeddings.shape}")
        return embeddings
    
    def process_dataframe(self, df, text_column='cleaned_description', batch_size=32):
        """
        Generate embeddings for text data in a DataFrame.
        
        Args:
            df: DataFrame containing transaction data
            text_column: Column containing text to embed
            batch_size: Batch size for embedding generation
            
        Returns:
            DataFrame with embeddings added as a new column
        """
        # Check if text column exists
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        
        # Generate embeddings
        texts = df[text_column].fillna('').tolist()
        embeddings = self.generate_embeddings(texts, batch_size=batch_size)
        
        # Create a copy to avoid modifying the original
        df_with_embeddings = df.copy()
        
        # Store the embeddings as a list of numpy arrays
        df_with_embeddings['embedding'] = list(embeddings)
        
        return df_with_embeddings
    
    def save_embeddings(self, df, output_path, separate_file=True):
        """
        Save DataFrame with embeddings to file.
        
        Args:
            df: DataFrame with embeddings
            output_path: Path to save the data
            separate_file: Whether to save embeddings in a separate file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if separate_file:
            # Extract embeddings
            embeddings = np.array(df['embedding'].tolist())
            
            # Save dataframe without embeddings
            df_copy = df.drop(columns=['embedding'])
            df_copy.to_csv(output_path, index=False)
            
            # Save embeddings to a separate pickle file
            embeddings_path = output_path.replace('.csv', '_embeddings.pkl')
            with open(embeddings_path, 'wb') as f:
                pickle.dump(embeddings, f)
            
            logger.info(f"Saved DataFrame to {output_path} and embeddings to {embeddings_path}")
        else:
            # Save full dataframe with embeddings as pickle
            pickle_path = output_path.replace('.csv', '.pkl')
            df.to_pickle(pickle_path)
            logger.info(f"Saved DataFrame with embeddings to {pickle_path}")

def load_embeddings(csv_path, embeddings_path=None):
    """
    Load a DataFrame and its embeddings from files.
    
    Args:
        csv_path: Path to the CSV file
        embeddings_path: Path to the embeddings pickle file
        
    Returns:
        DataFrame with embeddings
    """
    df = pd.read_csv(csv_path)
    
    if embeddings_path is None:
        embeddings_path = csv_path.replace('.csv', '_embeddings.pkl')
    
    # Check if embeddings file exists
    if not os.path.exists(embeddings_path):
        logger.warning(f"Embeddings file {embeddings_path} not found. Returning DataFrame without embeddings.")
        return df
    
    # Load embeddings
    with open(embeddings_path, 'rb') as f:
        embeddings = pickle.load(f)
    
    # Add embeddings to DataFrame
    df['embedding'] = list(embeddings)
    
    return df

if __name__ == "__main__":
    # Example usage
    from preprocessing import TransactionPreprocessor
    
    # Create directories if they don't exist
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/embeddings", exist_ok=True)
    
    # Load and preprocess data if needed
    preprocessor = TransactionPreprocessor(download_nltk=True)
    
    # Check if processed data exists
    processed_path = "data/processed/preprocessed_transactions.csv"
    if not os.path.exists(processed_path):
        # Load raw data
        raw_path = "data/transactions_sample.csv"
        if os.path.exists(raw_path):
            df = preprocessor.load_data(raw_path)
            df = preprocessor.preprocess(df)
            preprocessor.save_preprocessed_data(df, processed_path)
        else:
            logger.error(f"Raw data file {raw_path} not found.")
            exit(1)
    else:
        df = pd.read_csv(processed_path)
    
    # Generate embeddings
    generator = EmbeddingGenerator()
    df_with_embeddings = generator.process_dataframe(df)
    
    # Save the results
    generator.save_embeddings(
        df_with_embeddings,
        "data/embeddings/transaction_embeddings.csv",
        separate_file=True
    )