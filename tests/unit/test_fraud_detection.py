import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from src.pipeline.fraud_detection import FraudDetectionPipeline, PipelineConfig
from src.data.preprocessing import TransactionPreprocessor
from src.data.embeddings import EmbeddingGenerator
from src.models.anomaly_detector import AnomalyDetector

@pytest.fixture
def sample_transactions():
    """Create sample transaction data for testing."""
    return pd.DataFrame({
        'transaction_id': ['tx1', 'tx2', 'tx3'],
        'description': [
            'Payment for groceries',
            'Online purchase electronics',
            'ATM withdrawal'
        ],
        'amount': [100.0, 500.0, 200.0],
        'timestamp': pd.date_range(start='2024-01-01', periods=3),
        'merchant': ['Grocery Store', 'Electronics Shop', 'ATM']
    })

@pytest.fixture
def mock_vector_db():
    """Create a mock vector database."""
    mock_db = Mock()
    mock_db.batch_search.return_value = (
        np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),  # distances
        np.array([[1, 2], [3, 4], [5, 6]])  # neighbor_ids
    )
    return mock_db

@pytest.fixture
def pipeline_config():
    """Create a pipeline configuration."""
    return PipelineConfig(
        vector_db_path=Path("data/vector_db"),
        similarity_threshold=0.8,
        anomaly_threshold=0.95,
        batch_size=32,
        use_gpu=False,
        model_name="all-MiniLM-L6-v2"
    )

class TestFraudDetectionPipeline:
    """Test cases for the FraudDetectionPipeline class."""

    def test_pipeline_initialization(self, pipeline_config):
        """Test pipeline initialization with default components."""
        pipeline = FraudDetectionPipeline(config=pipeline_config)
        
        assert isinstance(pipeline.preprocessor, TransactionPreprocessor)
        assert isinstance(pipeline.embedding_generator, EmbeddingGenerator)
        assert isinstance(pipeline.anomaly_detector, AnomalyDetector)

    def test_preprocess_transactions(self, pipeline_config, sample_transactions):
        """Test transaction preprocessing."""
        pipeline = FraudDetectionPipeline(config=pipeline_config)
        
        # Process transactions
        processed_df = pipeline.preprocess_transactions(sample_transactions)
        
        # Check results
        assert 'cleaned_description' in processed_df.columns
        assert len(processed_df) == len(sample_transactions)
        assert all(processed_df['transaction_id'] == sample_transactions['transaction_id'])

    def test_preprocess_transactions_missing_columns(self, pipeline_config):
        """Test preprocessing with missing required columns."""
        pipeline = FraudDetectionPipeline(config=pipeline_config)
        
        # Create invalid data
        invalid_df = pd.DataFrame({
            'some_column': ['a', 'b', 'c']
        })
        
        # Check if it raises ValueError
        with pytest.raises(ValueError) as exc_info:
            pipeline.preprocess_transactions(invalid_df)
        
        assert "Missing required columns" in str(exc_info.value)

    @patch('src.data.embeddings.EmbeddingGenerator.process_dataframe')
    def test_generate_embeddings(self, mock_process_df, pipeline_config, sample_transactions):
        """Test embedding generation."""
        # Setup mock embeddings
        mock_embeddings = np.random.rand(3, 384)  # 3 transactions, 384 dimensions
        mock_process_df.return_value = pd.DataFrame({
            'transaction_id': sample_transactions['transaction_id'],
            'embedding': [emb for emb in mock_embeddings]
        })
        
        pipeline = FraudDetectionPipeline(config=pipeline_config)
        
        # Add cleaned_description column
        sample_transactions['cleaned_description'] = sample_transactions['description']
        
        # Generate embeddings
        df_with_embeddings, embeddings = pipeline.generate_embeddings(sample_transactions)
        
        # Check results
        assert 'embedding' in df_with_embeddings.columns
        assert embeddings.shape == (3, 384)
        assert len(df_with_embeddings) == len(sample_transactions)

    def test_detect_anomalies(self, pipeline_config, mock_vector_db, sample_transactions):
        """Test anomaly detection."""
        # Setup pipeline with mock vector database
        pipeline = FraudDetectionPipeline(config=pipeline_config)
        pipeline.vector_db = mock_vector_db
        
        # Create sample embeddings
        embeddings = np.random.rand(3, 384)
        
        # Add required columns
        sample_transactions['embedding'] = [emb for emb in embeddings]
        
        # Detect anomalies
        results = pipeline.detect_anomalies(sample_transactions, embeddings)
        
        # Check results
        assert 'is_anomalous' in results.columns
        assert 'anomaly_score' in results.columns
        assert len(results) == len(sample_transactions)

    @patch('src.pipeline.fraud_detection.FraudDetectionPipeline.preprocess_transactions')
    @patch('src.pipeline.fraud_detection.FraudDetectionPipeline.generate_embeddings')
    @patch('src.pipeline.fraud_detection.FraudDetectionPipeline.detect_anomalies')
    def test_process_transactions_end_to_end(
        self, mock_detect, mock_generate, mock_preprocess, 
        pipeline_config, sample_transactions
    ):
        """Test end-to-end transaction processing."""
        # Setup mocks
        mock_preprocess.return_value = sample_transactions
        mock_generate.return_value = (sample_transactions, np.random.rand(3, 384))
        mock_detect.return_value = pd.DataFrame({
            'transaction_id': sample_transactions['transaction_id'],
            'is_anomalous': [False, True, False],
            'anomaly_score': [0.1, 0.9, 0.3]
        })
        
        pipeline = FraudDetectionPipeline(config=pipeline_config)
        
        # Process transactions
        results = pipeline.process_transactions(sample_transactions, save_results=False)
        
        # Verify all steps were called
        mock_preprocess.assert_called_once()
        mock_generate.assert_called_once()
        mock_detect.assert_called_once()
        
        # Check results
        assert 'is_anomalous' in results.columns
        assert 'anomaly_score' in results.columns
        assert len(results) == len(sample_transactions)

    def test_generate_alerts(self, pipeline_config, sample_transactions):
        """Test alert generation for anomalous transactions."""
        pipeline = FraudDetectionPipeline(config=pipeline_config)
        
        # Add anomaly information
        anomalous_transactions = sample_transactions.copy()
        anomalous_transactions['is_anomalous'] = [False, True, False]
        anomalous_transactions['anomaly_score'] = [0.1, 0.9, 0.3]
        
        # Generate alerts
        alerts = pipeline.generate_alerts(
            anomalous_transactions[anomalous_transactions['is_anomalous']]
        )
        
        # Check alerts
        assert len(alerts) == 1
        alert = alerts[0]
        assert 'transaction_id' in alert
        assert 'timestamp' in alert
        assert 'anomaly_score' in alert
        assert 'amount' in alert
        assert isinstance(alert['anomaly_score'], float)
        assert isinstance(alert['amount'], float) 