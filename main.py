import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
import json
from datetime import datetime
import time
os.makedirs("logs", exist_ok=True)
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/fraud_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Make sure local modules can be imported
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import local modules
from src.data.generation import TransactionGenerator
from src.data.preprocessing import TransactionPreprocessor
from src.data.embeddings import EmbeddingGenerator
from src.database.vector_db import VectorDatabase, build_vector_database
from src.models.anomaly_detector import AnomalyDetector
from src.pipeline.fraud_detection import FraudDetectionPipeline
from src.pipeline.alert_system import AlertSystem

def setup_directories():
    """Create necessary directories for the project"""
    directories = [
        "data/raw",
        "data/processed",
        "data/embeddings",
        "data/vector_db",
        "models",
        "reports",
        "logs",
        "data/alerts"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Directory created: {directory}")

def generate_data(args):
    """Generate synthetic transaction data"""
    logger.info("Generating synthetic transaction data...")
    
    # Initialize the generator with the right parameters
    generator = TransactionGenerator(
        num_customers=10000, 
        num_merchants=1000, 
        fraud_ratio=args.fraud_ratio,
        seed=args.seed
    )
    
    # Generate test dataset
    logger.info("Generating test dataset...")
    test_df = generator.generate_transactions(
        num_transactions=args.test_size,  # Changed from n_transactions to num_transactions
        start_date=None,
        end_date=None
    )
    generator.save_transactions(test_df, "data/raw/test_transactions.csv")
    
    # Generate full dataset
    logger.info("Generating full dataset...")
    full_df = generator.generate_transactions(
        num_transactions=args.train_size,  # Changed from n_transactions to num_transactions
        start_date=None,
        end_date=None
    )
    generator.save_transactions(full_df, "data/raw/transactions.csv")
    
    logger.info("Data generation completed")
    return test_df, full_df

def preprocess_data(df, args):
    """Preprocess transaction data"""
    logger.info("Preprocessing transaction data...")
    
    preprocessor = TransactionPreprocessor(download_nltk=True)
    processed_df = preprocessor.preprocess(df)
    
    # Save processed data
    output_path = "data/processed/processed_transactions.csv"
    preprocessor.save_preprocessed_data(processed_df, output_path)
    
    logger.info(f"Preprocessing completed. Saved to {output_path}")
    return processed_df

def generate_embeddings(df, args):
    """Generate embeddings for transaction data"""
    logger.info("Generating embeddings...")
    
    # Initialize embedding generator
    embedding_generator = EmbeddingGenerator(
        model_name=args.embedding_model,
        device=args.device
    )
    
    # Generate embeddings
    df_with_embeddings = embedding_generator.process_dataframe(
        df, 
        text_column='cleaned_description',
        batch_size=args.batch_size
    )
    
    # Save embeddings
    output_path = "data/embeddings/transaction_embeddings.csv"
    embedding_generator.save_embeddings(
        df_with_embeddings, 
        output_path, 
        separate_file=True  # Use the correct parameter name
    )
    
    logger.info(f"Embedding generation completed. Saved to {output_path}")
    return df_with_embeddings

def build_vector_db(df_with_embeddings, args):
    """Build vector database from embeddings"""
    logger.info("Building vector database...")
    
    # Create and save vector database
    db = build_vector_database(
        df_with_embeddings,
        transaction_id_col='transaction_id',
        save_dir="data/vector_db"
    )
    
    logger.info(f"Vector database built with {len(df_with_embeddings)} embeddings")
    return db

def detect_anomalies(df_with_embeddings, vector_db, args):
    """Detect anomalies in transaction data"""
    logger.info("Detecting anomalies...")
    
    # Initialize anomaly detector
    detector = AnomalyDetector(
        similarity_threshold=args.similarity_threshold,
        min_cluster_size=args.min_cluster_size,
        anomaly_threshold=args.anomaly_threshold,
        use_clustering=args.use_clustering
    )
    
    # Extract embeddings and transaction IDs
    embeddings = np.vstack(df_with_embeddings['embedding'].values)
    transaction_ids = df_with_embeddings['transaction_id'].tolist()
    
    # Perform vector similarity search
    distances, neighbor_ids = vector_db.batch_search(
        embeddings, 
        k=args.nearest_neighbors
    )
    
    # Detect anomalies
    results = detector.detect_anomalies(
        embeddings=embeddings,
        transaction_ids=transaction_ids,
        distances=distances,
        neighbor_ids=neighbor_ids
    )
    
    # Merge results with original data
    merged_results = pd.merge(
        df_with_embeddings,
        results,
        on='transaction_id',
        how='left'
    )
    
    # Generate anomaly report
    report_path = f"reports/anomaly_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    detector.generate_anomaly_report(results, output_path=report_path)
    
    # Visualize anomaly distribution
    plot_path = f"reports/anomaly_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    if 'combined_anomaly_score' in results.columns:
        detector.analyze_anomaly_distribution(
            results['combined_anomaly_score'].values,
            output_path=plot_path
        )
    elif 'anomaly_score' in results.columns:
        detector.analyze_anomaly_distribution(
            results['anomaly_score'].values,
            output_path=plot_path
        )
    
    logger.info(f"Anomaly detection completed. Found {results['is_anomalous'].sum()} anomalies.")
    return merged_results

def generate_alerts(anomalous_transactions, args):
    """Generate alerts for anomalous transactions"""
    logger.info("Generating alerts...")
    
    # Initialize alert system
    alert_system = AlertSystem(alert_log_path="data/alerts/alert_log.json")
    
    # Generate alerts from anomalous transactions
    if 'is_anomalous' in anomalous_transactions.columns:
        anomalies = anomalous_transactions[anomalous_transactions['is_anomalous']]
    elif 'combined_is_anomalous' in anomalous_transactions.columns:
        anomalies = anomalous_transactions[anomalous_transactions['combined_is_anomalous']]
    else:
        logger.warning("No anomaly flags found in transactions")
        return []
    
    # Generate and save alerts
    alerts = alert_system.generate_alerts_from_transactions(anomalies)
    alert_system.save_alerts(alerts)
    
    logger.info(f"Generated {len(alerts)} alerts")
    return alerts

def run_fraud_detection_pipeline(args):
    """Run the complete fraud detection pipeline"""
    logger.info("Starting fraud detection pipeline...")
    
    # Initialize pipeline
    pipeline = FraudDetectionPipeline(
        vector_db_path="data/vector_db",
        similarity_threshold=args.similarity_threshold,
        anomaly_threshold=args.anomaly_threshold
    )
    
    # Load test data or use argument-provided data
    if args.input_file:
        if os.path.exists(args.input_file):
            test_data = pd.read_csv(args.input_file)
            logger.info(f"Loaded {len(test_data)} transactions from {args.input_file}")
        else:
            logger.error(f"Input file not found: {args.input_file}")
            return
    else:
        test_path = "data/raw/test_transactions.csv"
        if not os.path.exists(test_path):
            logger.warning(f"Test data not found at {test_path}. Generating new data...")
            test_data, _ = generate_data(args)
        else:
            test_data = pd.read_csv(test_path)
            logger.info(f"Loaded {len(test_data)} transactions from {test_path}")
    
    # Process transactions through pipeline
    logger.info("Processing transactions through pipeline...")
    if args.sample_size:
        sample = test_data.sample(min(args.sample_size, len(test_data)), random_state=args.seed)
    else:
        sample = test_data
    
    # Process through complete pipeline
    results = pipeline.process_transactions(sample)
    
    # Generate alerts for anomalies
    if 'is_anomalous' in results.columns and results['is_anomalous'].sum() > 0:
        alerts = pipeline.generate_alerts(results[results['is_anomalous']])
        logger.info(f"Generated {len(alerts)} alerts")
    
    # Save results
    output_path = f"reports/detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results.to_csv(output_path, index=False)
    logger.info(f"Saved detection results to {output_path}")
    
    logger.info("Fraud detection pipeline completed")
    return results

def end_to_end_process(args):
    """Run the complete end-to-end process"""
    start_time = time.time()
    logger.info("Starting end-to-end fraud detection process...")
    
    # Step 1: Setup directories
    setup_directories()
    
    # Step 2: Generate data if needed
    if args.generate_data or not os.path.exists("data/raw/transactions.csv"):
        test_df, full_df = generate_data(args)
    else:
        logger.info("Using existing data...")
        full_df = pd.read_csv("data/raw/transactions.csv")
        test_df = pd.read_csv("data/raw/test_transactions.csv")
    
    # Step 3: Preprocess data
    processed_df = preprocess_data(full_df, args)
    
    # Step 4: Generate embeddings
    df_with_embeddings = generate_embeddings(processed_df, args)
    
    # Step 5: Build vector database
    vector_db = build_vector_db(df_with_embeddings, args)
    
    # Step 6: Detect anomalies
    results = detect_anomalies(df_with_embeddings, vector_db, args)
    
    # Step 7: Generate alerts
    alerts = generate_alerts(results, args)
    
    # Calculate total processing time
    total_time = time.time() - start_time
    logger.info(f"End-to-end process completed in {total_time:.2f} seconds")
    
    # Summary
    anomaly_count = results['is_anomalous'].sum() if 'is_anomalous' in results.columns else 0
    logger.info("=== Summary ===")
    logger.info(f"Total transactions processed: {len(full_df)}")
    logger.info(f"Anomalies detected: {anomaly_count} ({anomaly_count/len(full_df)*100:.2f}%)")
    logger.info(f"Alerts generated: {len(alerts)}")
    logger.info(f"Results saved to reports/ directory")
    
    return {
        'total_transactions': len(full_df),
        'anomalies_detected': int(anomaly_count),
        'alerts_generated': len(alerts),
        'processing_time_seconds': total_time
    }

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Fraud Detection System')
    
    # Operation mode
    parser.add_argument('--mode', type=str, default='end-to-end',
                        choices=['end-to-end', 'data-gen', 'preprocess', 'embeddings', 
                                'vector-db', 'anomaly-detection', 'pipeline'],
                        help='Operation mode')
    
    # Data generation params
    parser.add_argument('--train-size', type=int, default=100000,
                        help='Number of training transactions to generate')
    parser.add_argument('--test-size', type=int, default=1000,
                        help='Number of test transactions to generate')
    parser.add_argument('--fraud-ratio', type=float, default=0.01,
                        help='Ratio of fraudulent transactions to generate')
    parser.add_argument('--generate-data', action='store_true',
                        help='Force generation of new data')
    
    # Preprocessing params
    
    # Embedding params
    parser.add_argument('--embedding-model', type=str, default='all-MiniLM-L6-v2',
                        help='Embedding model name')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for embedding generation')
    parser.add_argument('--device', type=str, default=None,
                        help='Device for embedding generation (cpu or cuda)')
    
    # Anomaly detection params
    parser.add_argument('--similarity-threshold', type=float, default=0.8,
                        help='Similarity threshold for anomaly detection')
    parser.add_argument('--anomaly-threshold', type=float, default=0.95,
                        help='Anomaly threshold for detection')
    parser.add_argument('--min-cluster-size', type=int, default=5,
                        help='Minimum cluster size for DBSCAN')
    parser.add_argument('--nearest-neighbors', type=int, default=10,
                        help='Number of nearest neighbors to search')
    parser.add_argument('--use-clustering', action='store_true',
                        help='Use clustering for anomaly detection')
    
    # Pipeline params
    parser.add_argument('--input-file', type=str, default=None,
                        help='Input file for fraud detection pipeline')
    parser.add_argument('--sample-size', type=int, default=None,
                        help='Sample size for processing (None = all)')
    
    # General params
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()

def main():

    setup_directories()

    """Main entry point"""
    # Parse arguments
    args = parse_arguments()
    
    try:
        # Run selected mode
        if args.mode == 'end-to-end':
            results = end_to_end_process(args)
            
            # Save summary
            summary_path = f"reports/summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(summary_path, 'w') as f:
                json.dump(results, f, indent=4)
            logger.info(f"Summary saved to {summary_path}")
            
        elif args.mode == 'data-gen':
            generate_data(args)
        
        elif args.mode == 'preprocess':
            if os.path.exists("data/raw/transactions.csv"):
                df = pd.read_csv("data/raw/transactions.csv")
                preprocess_data(df, args)
            else:
                logger.error("Raw data not found. Run with --mode data-gen first.")
        
        elif args.mode == 'embeddings':
            if os.path.exists("data/processed/processed_transactions.csv"):
                df = pd.read_csv("data/processed/processed_transactions.csv")
                generate_embeddings(df, args)
            else:
                logger.error("Processed data not found. Run with --mode preprocess first.")
        
        elif args.mode == 'vector-db':
            if os.path.exists("data/embeddings/transaction_embeddings.csv"):
                # Load embeddings
                from src.data.embeddings import load_embeddings
                embeddings_path = "data/embeddings/transaction_embeddings.csv"
                embeddings_data_path = embeddings_path.replace('.csv', '_embeddings.pkl')
                df = load_embeddings(embeddings_path, embeddings_data_path)
                build_vector_db(df, args)
            else:
                logger.error("Embeddings not found. Run with --mode embeddings first.")
        
        elif args.mode == 'anomaly-detection':
            if os.path.exists("data/vector_db/metadata.pkl"):
                # Load embeddings
                from src.data.embeddings import load_embeddings
                embeddings_path = "data/embeddings/transaction_embeddings.csv"
                embeddings_data_path = embeddings_path.replace('.csv', '_embeddings.pkl')
                df = load_embeddings(embeddings_path, embeddings_data_path)
                
                # Load vector database
                vector_db = VectorDatabase.load("data/vector_db")
                
                # Run anomaly detection
                detect_anomalies(df, vector_db, args)
            else:
                logger.error("Vector database not found. Run with --mode vector-db first.")
        
        elif args.mode == 'pipeline':
            run_fraud_detection_pipeline(args)
    
    except Exception as e:
        logger.exception(f"Error in fraud detection system: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()