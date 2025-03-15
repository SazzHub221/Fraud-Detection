import pandas as pd
import numpy as np
import os
import logging
import sys
import time
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.preprocessing import TransactionPreprocessor
from src.data.embeddings import EmbeddingGenerator, load_embeddings
from src.database.vector_db import VectorDatabase
from ..models.anomaly_detector import AnomalyDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """Configuration for the fraud detection pipeline."""
    vector_db_path: Path
    similarity_threshold: float
    anomaly_threshold: float
    batch_size: int = 32
    use_gpu: bool = False
    model_name: str = "all-MiniLM-L6-v2"

class FraudDetectionPipeline:
    """Main pipeline for fraud detection processing."""
    
    def __init__(
        self,
        config: PipelineConfig,
        preprocessor: Optional[TransactionPreprocessor] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        anomaly_detector: Optional[AnomalyDetector] = None
    ) -> None:
        """Initialize the fraud detection pipeline.
        
        Args:
            config: Pipeline configuration
            preprocessor: Optional custom preprocessor
            embedding_generator: Optional custom embedding generator
            anomaly_detector: Optional custom anomaly detector
        """
        self.config = config
        self.preprocessor = preprocessor or TransactionPreprocessor()
        self.embedding_generator = embedding_generator or EmbeddingGenerator(
            model_name=config.model_name,
            device="cuda" if config.use_gpu else "cpu"
        )
        self.anomaly_detector = anomaly_detector or AnomalyDetector(
            similarity_threshold=config.similarity_threshold,
            anomaly_threshold=config.anomaly_threshold
        )
        
        try:
            self.vector_db = VectorDatabase.load(config.vector_db_path)
            logger.info(f"Loaded vector database from {config.vector_db_path}")
        except Exception as e:
            logger.error(f"Failed to load vector database: {e}")
            raise RuntimeError("Vector database initialization failed") from e

    def preprocess_transactions(
        self, 
        transactions: pd.DataFrame
    ) -> pd.DataFrame:
        """Preprocess transaction data.
        
        Args:
            transactions: Raw transaction data
            
        Returns:
            Preprocessed transaction data
            
        Raises:
            ValueError: If required columns are missing
        """
        required_cols = ['transaction_id', 'description', 'amount']
        missing_cols = [col for col in required_cols if col not in transactions.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        try:
            return self.preprocessor.preprocess(transactions)
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise RuntimeError("Transaction preprocessing failed") from e

    def generate_embeddings(
        self, 
        transactions: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Generate embeddings for transaction descriptions.
        
        Args:
            transactions: Preprocessed transaction data
            
        Returns:
            Tuple of (DataFrame with embeddings, numpy array of embeddings)
            
        Raises:
            ValueError: If text column is missing
        """
        if 'cleaned_description' not in transactions.columns:
            raise ValueError("Missing 'cleaned_description' column")
            
        try:
            df_with_embeddings = self.embedding_generator.process_dataframe(
                transactions,
                text_column='cleaned_description',
                batch_size=self.config.batch_size
            )
            embeddings = np.vstack(df_with_embeddings['embedding'].values)
            return df_with_embeddings, embeddings
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise RuntimeError("Embedding generation failed") from e

    def detect_anomalies(
        self, 
        transactions: pd.DataFrame,
        embeddings: np.ndarray
    ) -> pd.DataFrame:
        """Detect anomalies in transactions using embeddings.
        
        Args:
            transactions: Transaction data with embeddings
            embeddings: Numpy array of embeddings
            
        Returns:
            DataFrame with anomaly detection results
        """
        try:
            # Perform similarity search
            distances, neighbor_ids = self.vector_db.batch_search(
                embeddings,
                k=10  # Number of nearest neighbors
            )
            
            # Detect anomalies
            results = self.anomaly_detector.detect_anomalies(
                embeddings=embeddings,
                transaction_ids=transactions['transaction_id'].tolist(),
                distances=distances,
                neighbor_ids=neighbor_ids
            )
            
            # Merge results with original data
            return pd.merge(
                transactions,
                results,
                on='transaction_id',
                how='left'
            )
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            raise RuntimeError("Anomaly detection failed") from e

    def process_transactions(
        self, 
        transactions: pd.DataFrame,
        save_results: bool = True
    ) -> pd.DataFrame:
        """Process transactions through the complete pipeline.
        
        Args:
            transactions: Raw transaction data
            save_results: Whether to save results to disk
            
        Returns:
            DataFrame with detection results
        """
        try:
            # Step 1: Preprocess
            logger.info("Preprocessing transactions...")
            processed_df = self.preprocess_transactions(transactions)
            
            # Step 2: Generate embeddings
            logger.info("Generating embeddings...")
            df_with_embeddings, embeddings = self.generate_embeddings(processed_df)
            
            # Step 3: Detect anomalies
            logger.info("Detecting anomalies...")
            results = self.detect_anomalies(df_with_embeddings, embeddings)
            
            if save_results:
                output_path = Path("reports") / f"detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                results.to_csv(output_path, index=False)
                logger.info(f"Results saved to {output_path}")
            
            return results
            
        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}")
            raise RuntimeError("Pipeline processing failed") from e

    def generate_alerts(
        self, 
        anomalous_transactions: pd.DataFrame
    ) -> List[Dict[str, Union[str, float]]]:
        """Generate alerts for anomalous transactions.
        
        Args:
            anomalous_transactions: DataFrame containing anomalous transactions
            
        Returns:
            List of alert dictionaries
        """
        try:
            alerts = []
            for _, row in anomalous_transactions.iterrows():
                alert = {
                    'transaction_id': row['transaction_id'],
                    'timestamp': datetime.now().isoformat(),
                    'anomaly_score': float(row.get('anomaly_score', 0)),
                    'confidence': float(row.get('confidence', 0)),
                    'description': row.get('description', ''),
                    'amount': float(row.get('amount', 0))
                }
                alerts.append(alert)
            
            logger.info(f"Generated {len(alerts)} alerts")
            return alerts
            
        except Exception as e:
            logger.error(f"Alert generation failed: {e}")
            raise RuntimeError("Alert generation failed") from e

if __name__ == "__main__":
    # Create a pipeline instance
    pipeline = FraudDetectionPipeline(
        config=PipelineConfig(
            vector_db_path="data/vector_db",
            similarity_threshold=0.8,
            anomaly_threshold=0.95
        )
    )
    
    # Check if we have sample data
    test_data_path = "data/raw/test_transactions.csv"
    if not os.path.exists(test_data_path):
        logger.error(f"Test data not found at {test_data_path}")
        exit(1)
    
    # Load test data
    test_data = pd.read_csv(test_data_path)
    logger.info(f"Loaded {len(test_data)} test transactions")
    
    # Process the first 10 transactions
    sample = test_data.head(10)
    results = pipeline.process_transactions(sample)
    
    # Print results
    if 'is_anomalous' in results.columns:
        anomalies = results[results['is_anomalous']]
        logger.info(f"Found {len(anomalies)} anomalies in sample")
        
        if not anomalies.empty:
            # Generate alerts
            alerts = pipeline.generate_alerts(anomalies)
            
            # Display first alert
            if alerts:
                logger.info(f"Sample alert: {alerts[0]}")
    
    logger.info("Fraud detection pipeline test complete")