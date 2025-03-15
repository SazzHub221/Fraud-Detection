import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_recall_curve, auc, precision_score, recall_score, f1_score
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.embeddings import load_embeddings
from src.models.anomaly_detector import AnomalyDetector
from src.database.vector_db import VectorDatabase

# Configure output directories
os.makedirs("reports/performance_metrics", exist_ok=True)

class TraditionalAnomalyDetector:
    """Traditional anomaly detection methods for comparison"""
    
    def __init__(self, method='isolation_forest', **kwargs):
        """Initialize the anomaly detector"""
        self.method = method
        
        if method == 'isolation_forest':
            self.model = IsolationForest(
                n_estimators=kwargs.get('n_estimators', 100),
                contamination=kwargs.get('contamination', 0.01),
                random_state=kwargs.get('random_state', 42)
            )
        
        elif method == 'local_outlier_factor':
            self.model = LocalOutlierFactor(
                n_neighbors=kwargs.get('n_neighbors', 20),
                contamination=kwargs.get('contamination', 0.01)
            )
        
        elif method == 'one_class_svm':
            self.model = OneClassSVM(
                nu=kwargs.get('nu', 0.01),
                kernel=kwargs.get('kernel', 'rbf'),
                gamma=kwargs.get('gamma', 'auto')
            )
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def detect_anomalies(self, X):
        """Detect anomalies in the data"""
        if self.method == 'local_outlier_factor':
            # LOF returns -1 for outliers, 1 for inliers
            y_pred = self.model.fit_predict(X)
            is_anomaly = y_pred == -1
            # LOF provides negative scores for outliers
            scores = -self.model.negative_outlier_factor_
            
        else:
            # Fit and predict
            self.model.fit(X)
            # Isolation Forest and One-Class SVM return 1 for inliers, -1 for outliers
            y_pred = self.model.predict(X)
            is_anomaly = y_pred == -1
            
            if self.method == 'isolation_forest':
                # For Isolation Forest, calculate anomaly score (higher = more anomalous)
                scores = -self.model.decision_function(X)
            else:
                # For One-Class SVM
                scores = -self.model.decision_function(X)
        
        return is_anomaly, scores

def run_comparison(embeddings_df, vector_db_path=None):
    """Run comparative analysis of different anomaly detection methods"""
    print("Running comparative analysis of anomaly detection methods...")
    
    # Extract embeddings and ground truth
    embeddings = np.vstack(embeddings_df['embedding'].values)
    transaction_ids = embeddings_df['transaction_id'].tolist()
    
    # Get ground truth if available (for supervised evaluation)
    has_ground_truth = 'is_fraud' in embeddings_df.columns
    if has_ground_truth:
        ground_truth = embeddings_df['is_fraud'].values
        print(f"Found ground truth with {sum(ground_truth)} fraud transactions out of {len(ground_truth)}")
    else:
        print("No ground truth found. Will analyze detection rates only.")
    
    # Initialize methods for comparison
    methods = {
        'LLM + Similarity': {
            'detector': AnomalyDetector(similarity_threshold=0.8, anomaly_threshold=0.95, use_clustering=False),
            'requires_vector_db': True
        },
        'LLM + Clustering': {
            'detector': AnomalyDetector(similarity_threshold=0.8, min_cluster_size=5, anomaly_threshold=0.95, use_clustering=True),
            'requires_vector_db': False
        },
        'Isolation Forest': {
            'detector': TraditionalAnomalyDetector(method='isolation_forest', contamination=0.01),
            'requires_vector_db': False
        },
        'Local Outlier Factor': {
            'detector': TraditionalAnomalyDetector(method='local_outlier_factor', contamination=0.01),
            'requires_vector_db': False
        },
        'One-Class SVM': {
            'detector': TraditionalAnomalyDetector(method='one_class_svm', nu=0.01),
            'requires_vector_db': False
        }
    }
    
    # Load vector database if needed
    vector_db = None
    if vector_db_path and os.path.exists(vector_db_path):
        vector_db = VectorDatabase.load(vector_db_path)
        print(f"Loaded vector database from {vector_db_path}")
        
        # Search for neighbors
        distances, neighbor_ids = vector_db.batch_search(embeddings, k=10)
    
    # Run each method and collect results
    results = []
    
    for method_name, config in methods.items():
        print(f"Running method: {method_name}")
        detector = config['detector']
        requires_vector_db = config['requires_vector_db']
        
        # Skip methods that require vector DB if it's not available
        if requires_vector_db and vector_db is None:
            print(f"Skipping {method_name} because vector database is not available")
            continue
        
        # Measure execution time
        start_time = time.time()
        
        # Run detection
        if method_name == 'LLM + Similarity':
            # Use similarity-based detection
            detection_results = detector.detect_anomalies_by_similarity(distances, neighbor_ids, transaction_ids)
            is_anomaly = detection_results['is_anomalous'].values
            anomaly_scores = detection_results['anomaly_score'].values
            
        elif method_name == 'LLM + Clustering':
            # Use clustering-based detection
            detection_results = detector.detect_anomalies_by_clustering(embeddings, transaction_ids)
            is_anomaly = detection_results['is_anomalous_cluster'].values
            anomaly_scores = detection_results['anomaly_score_cluster'].values
            
        else:
            # Use traditional methods
            is_anomaly, anomaly_scores = detector.detect_anomalies(embeddings)
        
        execution_time = time.time() - start_time
        
        # Calculate detection rate
        detection_rate = is_anomaly.mean()
        
        # Calculate metrics if ground truth is available
        if has_ground_truth:
            precision = precision_score(ground_truth, is_anomaly)
            recall = recall_score(ground_truth, is_anomaly)
            f1 = f1_score(ground_truth, is_anomaly)
            
            # Calculate precision-recall curve and AUC
            precision_curve, recall_curve, _ = precision_recall_curve(ground_truth, anomaly_scores)
            pr_auc = auc(recall_curve, precision_curve)
        else:
            precision = None
            recall = None
            f1 = None
            pr_auc = None
        
        # Store results
        results.append({
            'method': method_name,
            'execution_time': execution_time,
            'detection_rate': detection_rate,
            'num_anomalies': is_anomaly.sum(),
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'pr_auc': pr_auc
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = f"reports/performance_metrics/method_comparison_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)
    
    # Create plots
    plot_comparison_results(results_df, has_ground_truth, timestamp)
    
    return results_df

def plot_comparison_results(results_df, has_ground_truth, timestamp):
    """Plot the comparison results"""
    # Set up the figure
    fig_height = 15 if has_ground_truth else 9
    fig, axes = plt.subplots(3 if has_ground_truth else 2, 1, figsize=(12, fig_height))
    
    # Plot detection rate
    sns.barplot(x='method', y='detection_rate', data=results_df, ax=axes[0])
    axes[0].set_title('Anomaly Detection Rate by Method')
    axes[0].set_xlabel('Method')
    axes[0].set_ylabel('Detection Rate')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
    
    # Plot execution time
    sns.barplot(x='method', y='execution_time', data=results_df, ax=axes[1])
    axes[1].set_title('Execution Time by Method')
    axes[1].set_xlabel('Method')
    axes[1].set_ylabel('Execution Time (seconds)')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
    
    # Plot metrics if ground truth is available
    if has_ground_truth:
        # Melt the DataFrame to have one row per metric-method combination
        metrics = ['precision', 'recall', 'f1_score', 'pr_auc']
        melted_df = pd.melt(results_df, id_vars=['method'], value_vars=metrics, 
                            var_name='metric', value_name='value')
        
        # Plot metrics
        sns.barplot(x='method', y='value', hue='metric', data=melted_df, ax=axes[2])
        axes[2].set_title('Performance Metrics by Method')
        axes[2].set_xlabel('Method')
        axes[2].set_ylabel('Score')
        axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45, ha='right')
        axes[2].legend(title='Metric')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plot_path = f"reports/performance_metrics/method_comparison_{timestamp}.png"
    plt.savefig(plot_path)
    print(f"Saved comparison plot to {plot_path}")
    
    # Close figure
    plt.close(fig)

def main():
    """Main function to run the comparative analysis"""
    # Load embeddings
    try:
        embeddings_path = "data/embeddings/transaction_embeddings.csv"
        embeddings_data_path = embeddings_path.replace('.csv', '_embeddings.pkl')
        
        if os.path.exists(embeddings_path) and os.path.exists(embeddings_data_path):
            df = load_embeddings(embeddings_path, embeddings_data_path)
            
            # Run comparative analysis
            vector_db_path = "data/vector_db"
            results = run_comparison(df, vector_db_path)
            
            print("\nComparison Results:")
            print(results)
            
        else:
            print(f"Embeddings not found at {embeddings_path}")
            sys.exit(1)
    
    except Exception as e:
        print(f"Error in comparative analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()