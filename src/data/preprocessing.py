import pandas as pd
import numpy as np
from datetime import datetime
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer

class TransactionPreprocessor:
    def __init__(self, download_nltk=True):
        """
        Initialize the transaction preprocessor.
        
        Args:
            download_nltk: Whether to download NLTK resources
        """
        if download_nltk:
            try:
                nltk.download('punkt')
                nltk.download('stopwords')
                self.stop_words = set(stopwords.words('english'))
            except:
                print("Warning: Could not download NLTK resources. Using empty stopwords set.")
                self.stop_words = set()
        else:
            self.stop_words = set()
    
    def load_data(self, filepath):
        """Load transaction data from CSV file"""
        return pd.read_csv(filepath)
    
    def clean_text(self, text):
        """Clean and normalize text descriptions"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Tokenize
        tokenizer = TreebankWordTokenizer()
        tokens = tokenizer.tokenize(text.lower())
        
        # Remove stopwords
        filtered_tokens = [word for word in tokens if word not in self.stop_words]
        
        # Rejoin tokens
        return ' '.join(filtered_tokens)
    
    def add_temporal_features(self, df):
        """Add temporal features to the dataframe"""
        # Convert date to datetime if it's not already
        if df['date'].dtype != 'datetime64[ns]':
            df['date'] = pd.to_datetime(df['date'])
        
        # Extract features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['hour'] = df['date'].dt.hour
        df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        
        # Time since previous transaction per customer
        df = df.sort_values(['customer_id', 'date'])
        df['prev_tx_time'] = df.groupby('customer_id')['date'].shift(1)
        df['time_since_prev_tx'] = (df['date'] - df['prev_tx_time']).dt.total_seconds() / 3600  # hours
        
        return df
    
    def add_statistical_features(self, df):
        """Add statistical features based on customer patterns"""
        # Calculate average and std of amount per customer
        customer_stats = df.groupby('customer_id')['amount'].agg(['mean', 'std']).reset_index()
        customer_stats.columns = ['customer_id', 'customer_avg_amount', 'customer_std_amount']
        
        # Merge with original dataframe
        df = pd.merge(df, customer_stats, on='customer_id', how='left')
        
        # Calculate z-score of transaction amount
        df['amount_zscore'] = (df['amount'] - df['customer_avg_amount']) / df['customer_std_amount'].replace(0, 1)
        
        # Calculate merchant frequency per customer
        merchant_freq = df.groupby(['customer_id', 'merchant_id']).size().reset_index()
        merchant_freq.columns = ['customer_id', 'merchant_id', 'merchant_frequency']
        
        # Merge with original dataframe
        df = pd.merge(df, merchant_freq, on=['customer_id', 'merchant_id'], how='left')
        
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataframe"""
        # Fill missing numerical values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        
        # Fill missing categorical values
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
        
        return df
    
    def preprocess(self, df=None, filepath=None, clean_descriptions=True):
        """
        Preprocess the transaction data.
        
        Args:
            df: DataFrame containing transaction data
            filepath: Path to CSV file (alternative to df)
            clean_descriptions: Whether to clean text descriptions
            
        Returns:
            Preprocessed DataFrame
        """
        if df is None and filepath is not None:
            df = self.load_data(filepath)
        elif df is None:
            raise ValueError("Either df or filepath must be provided")
        
        print(f"Preprocessing {len(df)} transactions...")
        
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Clean descriptions
        if clean_descriptions:
            print("Cleaning text descriptions...")
            df['cleaned_description'] = df['description'].apply(self.clean_text)
        
        # Add temporal features
        print("Adding temporal features...")
        df = self.add_temporal_features(df)
        
        # Add statistical features
        print("Adding statistical features...")
        df = self.add_statistical_features(df)
        
        # Handle missing values
        print("Handling missing values...")
        df = self.handle_missing_values(df)
        
        print("Preprocessing completed.")
        return df
    
    def save_preprocessed_data(self, df, filepath):
        """Save preprocessed data to file"""
        df.to_csv(filepath, index=False)
        print(f"Saved preprocessed data to {filepath}")
        return filepath

# Example usage
if __name__ == "__main__":
    preprocessor = TransactionPreprocessor()
    df = preprocessor.load_data("data/transactions_sample.csv")
    processed_df = preprocessor.preprocess(df)
    preprocessor.save_preprocessed_data(processed_df, "data/preprocessed_transactions.csv")