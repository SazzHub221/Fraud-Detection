import os
import sys
import unittest
import pandas as pd
import numpy as np
import shutil
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.generation import TransactionGenerator
from src.data.preprocessing import TransactionPreprocessor
from src.data.embeddings import EmbeddingGenerator
from src.database.vector_db import VectorDatabase, build_vector_database
from src.models.anomaly_detector import AnomalyDetector
from src.pipeline.fraud_detection import FraudDetectionPipeline

class TestFraudDetectionPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all tests"""
        # Create test directories
        os.makedirs("tests/test_data/raw", exist_ok=True)
        os.makedirs("tests/test_data/processed", exist_ok=True)
        os.makedirs("tests/test_data/embeddings", exist_ok=True)
        os.makedirs("tests/test_data/vector_db", exist_ok=True)
        
        # Generate small test dataset
        generator = TransactionGenerator(num_customers=100, num_merchants=50, fraud_ratio=0.05)
        cls.test_data = generator.generate_transactions(num_transactions=200)
        cls.test_data.to_csv("tests/test_data/raw/test_transactions.csv", index=False)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests"""
        # Remove test directories and files
        if os.path.exists("tests/test_data"):
            shutil.rmtree("tests/test_data")
    
    def test_pipeline_components(self):
        """Test that each pipeline component works with the output of the previous component"""
        # 1. Load test data
        data_path = "tests/test_data/raw/test_transactions.csv"
        df = pd.read_csv(data_path)
        self.assertTrue(len(df) > 0)
        
        # 2. Preprocess data
        preprocessor = TransactionPreprocessor(download_nltk=True)
        processed_df = preprocessor.preprocess(df)
        self.assertTrue("cleaned_description" in processed_df.columns)
        
        # Save for next component
        processed_path = "tests/test_data/processed/processed_test.csv"
        preprocessor.save_processed_data(processed_df, processed_path)
        
        # 3. Generate embeddings
        embedder = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
        df_with_embeddings = embedder.process_dataframe(processed_df, text_column="cleaned_description", batch_size=32)
        self.assertTrue("embedding" in df_with_embeddings.columns)
        
        # Save embeddings
        embeddings_path = "tests/test_data/embeddings/test_embeddings.csv"
        embedder.save_embeddings(df_with_embeddings, embeddings_path)
        
        # 4. Build vector database
        vector_db = build_vector_database(df_with_embeddings, "transaction_id", save_dir="tests/test_data/vector_db")
        self.assertTrue(vector_db is not None)
        
        # 5. Initialize fraud detection pipeline
        pipeline = FraudDetectionPipeline(
            vector_db_path="tests/test_data/vector_db",
            similarity_threshold=0.8,
            anomaly_threshold=0.95
        )
        
        # 6. Process test transactions
        sample_size = min(20, len(df))
        sample = df.head(sample_size)
        results = pipeline.process_transactions(sample)
        
        # Check results
        self.assertEqual(len(results), sample_size)
        self.assertTrue("is_anomalous" in results.columns)
        
        # 7. Generate alerts
        if results["is_anomalous"].sum() > 0:
            anomalies = results[results["is_anomalous"]]
            alerts = pipeline.generate_alerts(anomalies)
            self.assertTrue(len(alerts) > 0)
    
    def test_end_to_end_pipeline(self):
        """Test the full pipeline as a single process"""
        # Load test data
        data_path = "tests/test_data/raw/test_transactions.csv"
        
        # Initialize pipeline
        pipeline = FraudDetectionPipeline(
            preprocessor=TransactionPreprocessor(download_nltk=True),
            embedding_generator=EmbeddingGenerator(model_name="all-MiniLM-L6-v2"),
            vector_db_path="tests/test_data/vector_db",
            similarity_threshold=0.8,
            anomaly_threshold=0.95
        )
        
        # Run pipeline directly on the input file
        results = pipeline.run_from_file(data_path)
        
        # Check results
        self.assertTrue(len(results) > 0)
        self.assertTrue("is_anomalous" in results.columns)

if __name__ == "__main__":
    unittest.main()