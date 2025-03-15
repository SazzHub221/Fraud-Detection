import os
import sys
import unittest
import numpy as np
import pandas as pd
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.anomaly_detector import AnomalyDetector

class TestAnomalyDetector(unittest.TestCase):
    def setUp(self):
        # Create a small test dataset with embeddings
        self.num_samples = 100
        self.embedding_dim = 384
        
        # Generate random embeddings
        np.random.seed(42)
        self.embeddings = np.random.rand(self.num_samples, self.embedding_dim)
        
        # Create transaction IDs
        self.transaction_ids = [f"tx_{i}" for i in range(self.num_samples)]
        
        # Create some outliers
        outlier_indices = [3, 15, 42, 87]
        for idx in outlier_indices:
            self.embeddings[idx] = np.random.rand(self.embedding_dim) * 5  # Make outlier values larger
        
        # Initialize detector
        self.detector = AnomalyDetector(
            similarity_threshold=0.8,
            min_cluster_size=5,
            anomaly_threshold=0.95,
            use_clustering=True
        )
    
    def test_detect_anomalies_by_clustering(self):
        """Test clustering-based anomaly detection"""
        results = self.detector.detect_anomalies_by_clustering(
            self.embeddings, 
            self.transaction_ids
        )
        
        # Check results dataframe
        self.assertEqual(len(results), self.num_samples)
        self.assertTrue('is_anomalous_cluster' in results.columns)
        self.assertTrue('anomaly_score_cluster' in results.columns)
        
        # There should be some anomalies
        self.assertTrue(results['is_anomalous_cluster'].sum() > 0)
    
    def test_detect_anomalies_by_similarity(self):
        """Test similarity-based anomaly detection"""
        # Mock distance matrix and neighbor IDs
        distances = np.random.rand(self.num_samples, 10)
        neighbor_ids = [[f"tx_{i+j}" for j in range(10)] for i in range(self.num_samples)]
        
        # Make some distances high (anomalous)
        distances[3, :] = 0.98
        distances[15, :] = 0.97
        distances[42, :] = 0.99
        
        results = self.detector.detect_anomalies_by_similarity(
            distances, 
            neighbor_ids, 
            self.transaction_ids
        )
        
        # Check results dataframe
        self.assertEqual(len(results), self.num_samples)
        self.assertTrue('is_anomalous' in results.columns)
        self.assertTrue('anomaly_score' in results.columns)
        
        # There should be some anomalies
        self.assertTrue(results['is_anomalous'].sum() > 0)
    
    def test_anomaly_report_generation(self):
        """Test anomaly report generation"""
        # Create dummy results DataFrame
        results = pd.DataFrame({
            'transaction_id': self.transaction_ids,
            'is_anomalous': [i % 20 == 0 for i in range(self.num_samples)],
            'anomaly_score': np.random.rand(self.num_samples)
        })
        
        # Generate report
        report = self.detector.generate_anomaly_report(results)
        
        # Check report structure
        self.assertTrue('summary' in report)
        self.assertTrue('score_distribution' in report)
        self.assertTrue('config' in report)
        
        # Check summary values
        self.assertEqual(report['summary']['total_transactions'], self.num_samples)
        self.assertEqual(report['summary']['anomaly_count'], sum([i % 20 == 0 for i in range(self.num_samples)]))

if __name__ == '__main__':
    unittest.main()