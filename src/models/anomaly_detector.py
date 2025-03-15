import numpy as np
import pandas as pd
import logging
import time
from typing import List, Dict, Tuple, Optional, Union
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AnomalyDetector:
    """
    Anomaly detector for transaction data using vector similarity and clustering approaches
    """
    
    def __init__(self, 
                 similarity_threshold: float = 0.8, 
                 min_cluster_size: int = 5,
                 anomaly_threshold: float = 0.95,
                 use_clustering: bool = True):
        """
        Initialize the anomaly detector.
        
        Args:
            similarity_threshold: Threshold for similarity scores (higher = more similar)
            min_cluster_size: Minimum cluster size for DBSCAN
            anomaly_threshold: Threshold for anomaly scores (higher = more anomalous)
            use_clustering: Whether to use clustering in addition to similarity search
        """
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
        self.anomaly_threshold = anomaly_threshold
        self.use_clustering = use_clustering
        self.dbscan = None
        
    def detect_anomalies_by_similarity(self, 
                                      distances: np.ndarray, 
                                      neighbor_ids: List[List[str]], 
                                      transaction_ids: List[str]) -> pd.DataFrame:
        """
        Detect anomalies based on similarity search results.
        
        Args:
            distances: Distance matrix from similarity search (n_samples, k_neighbors)
            neighbor_ids: Neighbor IDs from similarity search
            transaction_ids: IDs of the transactions being analyzed
            
        Returns:
            DataFrame with anomaly scores and flags
        """
        results = []
        
        for i, (dist_row, ids_row, tx_id) in enumerate(zip(distances, neighbor_ids, transaction_ids)):
            # Filter out None values (which can occur if fewer than k neighbors are found)
            valid_indices = [j for j, id_val in enumerate(ids_row) if id_val is not None]
            valid_distances = dist_row[valid_indices]
            
            if len(valid_distances) == 0:
                # No valid neighbors found
                avg_distance = 1.0  # Maximum distance
                is_anomalous = True
            else:
                # Calculate average distance to valid neighbors
                avg_distance = np.mean(valid_distances)
                
                # Distance closer to 1 means more anomalous for cosine similarity
                is_anomalous = avg_distance > self.anomaly_threshold
            
            # Store results
            results.append({
                'transaction_id': tx_id,
                'avg_distance': float(avg_distance),  # Convert from numpy to Python float
                'nearest_neighbors': [n for n in ids_row if n is not None],
                'nearest_distances': valid_distances.tolist(),
                'is_anomalous': bool(is_anomalous),
                'anomaly_score': min(1.0, float(avg_distance) / self.anomaly_threshold)
            })
        
        return pd.DataFrame(results)
    
    def detect_anomalies_by_clustering(self, 
                                      embeddings: np.ndarray, 
                                      transaction_ids: List[str],
                                      eps: float = 0.3) -> pd.DataFrame:
        """
        Detect anomalies using DBSCAN clustering.
        
        Args:
            embeddings: Embedding vectors
            transaction_ids: IDs of the transactions being analyzed
            eps: DBSCAN epsilon parameter (maximum distance between samples)
            
        Returns:
            DataFrame with cluster assignments and anomaly flags
        """
        # Initialize DBSCAN
        self.dbscan = DBSCAN(
            eps=eps, 
            min_samples=self.min_cluster_size, 
            metric='cosine',
            n_jobs=-1
        )
        
        # Fit DBSCAN
        start_time = time.time()
        cluster_labels = self.dbscan.fit_predict(embeddings)
        clustering_time = time.time() - start_time
        
        logger.info(f"DBSCAN clustering completed in {clustering_time:.4f} seconds")
        
        # Count points in each cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(i)
        
        # Calculate results
        results = []
        for i, (tx_id, cluster) in enumerate(zip(transaction_ids, cluster_labels)):
            # Transactions with cluster label -1 are outliers
            is_anomalous = cluster == -1
            
            # Calculate distance to cluster centroid for non-outliers
            if not is_anomalous and cluster in clusters:
                # Get all points in this cluster
                cluster_indices = clusters[cluster]
                cluster_points = embeddings[cluster_indices]
                
                # Calculate centroid
                centroid = np.mean(cluster_points, axis=0)
                
                # Calculate cosine similarity with centroid
                embedding = embeddings[i]
                similarity = np.dot(embedding, centroid) / (np.linalg.norm(embedding) * np.linalg.norm(centroid))
                
                # Convert similarity to distance (1 - similarity)
                distance_to_centroid = 1 - similarity
            else:
                # For outliers, set maximum distance
                distance_to_centroid = 1.0
            
            # Store results
            results.append({
                'transaction_id': tx_id,
                'cluster': int(cluster),
                'cluster_size': len(clusters.get(cluster, [])),
                'distance_to_centroid': float(distance_to_centroid),
                'is_anomalous_cluster': bool(is_anomalous),
                'anomaly_score_cluster': min(1.0, float(distance_to_centroid) / self.anomaly_threshold)
            })
        
        return pd.DataFrame(results)
    
    def detect_anomalies(self, 
                         embeddings: np.ndarray, 
                         transaction_ids: List[str],
                         distances: Optional[np.ndarray] = None,
                         neighbor_ids: Optional[List[List[str]]] = None) -> pd.DataFrame:
        """
        Combined anomaly detection using both similarity search and clustering.
        
        Args:
            embeddings: Embedding vectors
            transaction_ids: IDs of the transactions
            distances: Distance matrix from similarity search (optional)
            neighbor_ids: Neighbor IDs from similarity search (optional)
            
        Returns:
            DataFrame with anomaly detection results
        """
        results_df = pd.DataFrame({'transaction_id': transaction_ids})
        
        # Detect anomalies by similarity if distances and neighbor_ids provided
        if distances is not None and neighbor_ids is not None:
            similarity_results = self.detect_anomalies_by_similarity(
                distances=distances,
                neighbor_ids=neighbor_ids,
                transaction_ids=transaction_ids
            )
            results_df = pd.merge(results_df, similarity_results, on='transaction_id', how='left')
        
        # Detect anomalies by clustering if requested
        if self.use_clustering:
            cluster_results = self.detect_anomalies_by_clustering(
                embeddings=embeddings,
                transaction_ids=transaction_ids
            )
            results_df = pd.merge(results_df, cluster_results, on='transaction_id', how='left')
            
            # Combine anomaly scores if both methods are used
            if distances is not None and neighbor_ids is not None:
                # Average the anomaly scores from both methods
                results_df['combined_anomaly_score'] = (
                    results_df['anomaly_score'] + results_df['anomaly_score_cluster']
                ) / 2
                
                # Flag as anomalous if either method flags it
                results_df['combined_is_anomalous'] = (
                    results_df['is_anomalous'] | results_df['is_anomalous_cluster']
                )
        
        return results_df
    
    def analyze_anomaly_distribution(self, anomaly_scores: np.ndarray, output_path: Optional[str] = None):
        """
        Analyze the distribution of anomaly scores and determine optimal threshold.
        
        Args:
            anomaly_scores: Array of anomaly scores
            output_path: Path to save the plot (optional)
            
        Returns:
            Dictionary with analysis results
        """
        # Calculate statistics
        stats = {
            'mean': np.mean(anomaly_scores),
            'median': np.median(anomaly_scores),
            'std': np.std(anomaly_scores),
            'min': np.min(anomaly_scores),
            'max': np.max(anomaly_scores),
            'percentiles': {
                '90': np.percentile(anomaly_scores, 90),
                '95': np.percentile(anomaly_scores, 95),
                '99': np.percentile(anomaly_scores, 99)
            }
        }
        
        # Create plot
        plt.figure(figsize=(10, 6))
        sns.histplot(anomaly_scores, kde=True, bins=50)
        
        # Add vertical lines for percentiles
        plt.axvline(x=stats['percentiles']['90'], color='yellow', linestyle='--', label='90th percentile')
        plt.axvline(x=stats['percentiles']['95'], color='orange', linestyle='--', label='95th percentile')
        plt.axvline(x=stats['percentiles']['99'], color='red', linestyle='--', label='99th percentile')
        
        # Add current threshold
        plt.axvline(x=self.anomaly_threshold, color='black', linestyle='-', label=f'Current threshold ({self.anomaly_threshold})')
        
        plt.title('Distribution of Anomaly Scores')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Frequency')
        plt.legend()
        
        # Save or display plot
        if output_path:
            plt.savefig(output_path)
            logger.info(f"Saved anomaly distribution plot to {output_path}")
        else:
            plt.show()
        
        plt.close()
        
        return stats
    
    def generate_anomaly_report(self, anomaly_results: pd.DataFrame, output_path: Optional[str] = None):
        """
        Generate a comprehensive anomaly detection report.
    
        Args:
            anomaly_results: DataFrame with anomaly detection results
            output_path: Path to save the report (optional)
        
        Returns:
            Report as a dictionary
        """
        # Determine which anomaly column to use
        anomaly_col = None
        score_col = None
    
        if 'combined_is_anomalous' in anomaly_results.columns:
            anomaly_col = 'combined_is_anomalous'
            score_col = 'combined_anomaly_score'
        elif 'is_anomalous' in anomaly_results.columns:
            anomaly_col = 'is_anomalous'
            score_col = 'anomaly_score'
        elif 'is_anomalous_cluster' in anomaly_results.columns:
            anomaly_col = 'is_anomalous_cluster'
            score_col = 'anomaly_score_cluster'
        else:
            logger.warning("No anomaly flags found in results")
            anomaly_col = None
    
        # Calculate summary statistics
        if anomaly_col:
            anomaly_count = int(anomaly_results[anomaly_col].sum())
            total_count = len(anomaly_results)
            anomaly_rate = anomaly_count / total_count if total_count > 0 else 0
        else:
            anomaly_count = 0
            total_count = len(anomaly_results)
            anomaly_rate = 0
    
        # Calculate score distribution
        score_distribution = {}
        if score_col and score_col in anomaly_results.columns:
            score_distribution = {
                'mean': float(anomaly_results[score_col].mean()),
                'median': float(anomaly_results[score_col].median()),
                'std': float(anomaly_results[score_col].std()),
                'min': float(anomaly_results[score_col].min()),
                'max': float(anomaly_results[score_col].max()),
            }
    
        # Create report
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_transactions': total_count,
                'anomaly_count': anomaly_count,
                'anomaly_rate': anomaly_rate
            },
            'score_distribution': score_distribution,
            'config': {
                'similarity_threshold': self.similarity_threshold,
                'min_cluster_size': self.min_cluster_size,
                'anomaly_threshold': self.anomaly_threshold,
                'use_clustering': self.use_clustering
            }
        }
    
        # Save report if path provided
        if output_path:
            import json
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=4)
            logger.info(f"Saved anomaly report to {output_path}")
    
        return report


# Example usage
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    
    from src.data.embeddings import load_embeddings
    
    # Create output directory
    os.makedirs("reports", exist_ok=True)
    
    try:
        # Load embeddings
        embeddings_path = "data/embeddings/transaction_embeddings.csv"
        embeddings_data_path = embeddings_path.replace('.csv', '_embeddings.pkl')
        
        if os.path.exists(embeddings_path) and os.path.exists(embeddings_data_path):
            df = load_embeddings(embeddings_path, embeddings_data_path)
            
            # Extract embeddings and transaction IDs
            embeddings = np.vstack(df['embedding'].values)
            transaction_ids = df['transaction_id'].tolist()
            
            # Initialize anomaly detector
            detector = AnomalyDetector(
                similarity_threshold=0.8,
                min_cluster_size=5,
                anomaly_threshold=0.95,
                use_clustering=True
            )
            
            # Perform clustering-based anomaly detection
            logger.info("Performing anomaly detection...")
            results = detector.detect_anomalies_by_clustering(embeddings, transaction_ids)
            
            # Analyze results - note the corrected column name
            anomaly_count = results['is_anomalous_cluster'].sum()
            logger.info(f"Found {anomaly_count} anomalies out of {len(results)} transactions")
            
            # Generate and save anomaly report
            report = detector.generate_anomaly_report(
                results, 
                output_path="reports/anomaly_report.json"
            )
            
            # Analyze anomaly distribution
            detector.analyze_anomaly_distribution(
                results['anomaly_score_cluster'].values,
                output_path="reports/anomaly_distribution.png"
            )
            
        else:
            logger.error(f"Embeddings not found at {embeddings_path}. Run embeddings.py first.")
    
    except Exception as e:
        logger.exception(f"Error in anomaly detection: {e}")