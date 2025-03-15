import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.database.vector_db import VectorDatabase
from src.data.embeddings import load_embeddings

# Configure output directories
os.makedirs("reports/performance_metrics", exist_ok=True)

def benchmark_search_performance(vector_db, embeddings, transaction_ids, k_values=[5, 10, 20, 50, 100]):
    """Benchmark the performance of similarity search with different k values"""
    results = []
    
    # Use a sample of embeddings for querying
    sample_size = min(1000, len(embeddings))
    sample_indices = np.random.choice(len(embeddings), sample_size, replace=False)
    query_embeddings = embeddings[sample_indices]
    
    for k in k_values:
        print(f"Testing search with k={k}")
        
        # Measure search time
        start_time = time.time()
        distances, neighbors = vector_db.batch_search(query_embeddings, k=k)
        end_time = time.time()
        
        search_time = end_time - start_time
        avg_time_per_query = search_time / sample_size
        
        # Calculate query throughput
        throughput = sample_size / search_time
        
        results.append({
            'k': k,
            'total_time': search_time,
            'avg_time_per_query': avg_time_per_query,
            'queries_per_second': throughput,
            'num_queries': sample_size
        })
    
    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = f"reports/performance_metrics/search_benchmark_{timestamp}.csv"
    df_results.to_csv(csv_path, index=False)
    
    # Create plots
    plot_benchmark_results(df_results, timestamp)
    
    return df_results

def plot_benchmark_results(df_results, timestamp):
    """Plot the benchmark results"""
    # Set up the figure
    fig, ax = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot average query time vs k
    sns.barplot(x='k', y='avg_time_per_query', data=df_results, ax=ax[0])
    ax[0].set_title('Average Query Time vs. k')
    ax[0].set_xlabel('k (number of neighbors)')
    ax[0].set_ylabel('Average Time per Query (seconds)')
    
    # Plot throughput vs k
    sns.barplot(x='k', y='queries_per_second', data=df_results, ax=ax[1])
    ax[1].set_title('Query Throughput vs. k')
    ax[1].set_xlabel('k (number of neighbors)')
    ax[1].set_ylabel('Queries per Second')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plot_path = f"reports/performance_metrics/search_benchmark_{timestamp}.png"
    plt.savefig(plot_path)
    print(f"Saved benchmark plot to {plot_path}")
    
    # Close figure
    plt.close(fig)

def main():
    """Main function to run the benchmark"""
    # Load embeddings
    try:
        embeddings_path = "data/embeddings/transaction_embeddings.csv"
        embeddings_data_path = embeddings_path.replace('.csv', '_embeddings.pkl')
        
        if os.path.exists(embeddings_path) and os.path.exists(embeddings_data_path):
            df = load_embeddings(embeddings_path, embeddings_data_path)
            
            # Extract embeddings and transaction IDs
            embeddings = np.vstack(df['embedding'].values)
            transaction_ids = df['transaction_id'].tolist()
            
            # Load vector database
            vector_db_path = "data/vector_db"
            if not os.path.exists(vector_db_path):
                print(f"Vector database not found at {vector_db_path}")
                sys.exit(1)
            
            vector_db = VectorDatabase.load(vector_db_path)
            print(f"Loaded vector database with {len(transaction_ids)} transactions")
            
            # Run benchmarks
            print("Starting search performance benchmark...")
            results = benchmark_search_performance(vector_db, embeddings, transaction_ids)
            
            print("\nBenchmark Results:")
            print(results)
            
        else:
            print(f"Embeddings not found at {embeddings_path}")
            sys.exit(1)
    
    except Exception as e:
        print(f"Error in benchmark: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()