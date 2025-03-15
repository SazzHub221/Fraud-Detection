import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import uuid

class TransactionGenerator:
    def __init__(self, num_customers=10000, num_merchants=1000, fraud_ratio=0.01,seed=42):
        """
        Initialize the transaction generator.
        
        Args:
            num_customers: Number of unique customers
            num_merchants: Number of unique merchants
            fraud_ratio: Proportion of fraudulent transactions
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        self.num_customers = num_customers
        self.num_merchants = num_merchants
        self.fraud_ratio = fraud_ratio
        
        # Generate customer and merchant IDs
        self.customer_ids = [f"CUST_{uuid.uuid4().hex[:8]}" for _ in range(num_customers)]
        self.merchant_ids = [f"MERCH_{uuid.uuid4().hex[:8]}" for _ in range(num_merchants)]
        
        # Merchant categories and descriptions
        self.categories = ['Retail', 'Food', 'Travel', 'Entertainment', 'Services', 
                          'Healthcare', 'Technology', 'Automotive', 'Utilities', 'Other']
        
        self.merchant_categories = {merchant: random.choice(self.categories) 
                                   for merchant in self.merchant_ids}
        
        # Normal transaction patterns per customer (mean and std of amount)
        self.customer_patterns = {
            customer: {
                'mean_amount': random.uniform(20, 200),
                'std_amount': random.uniform(5, 50),
                'frequent_merchants': random.sample(self.merchant_ids, random.randint(5, 20)),
                'avg_transactions_per_month': random.randint(5, 30)
            }
            for customer in self.customer_ids
        }
    
    def _generate_description(self, merchant_id, amount, is_fraud=False):
        """Generate a transaction description based on merchant and amount"""
        category = self.merchant_categories[merchant_id]
        merchant_name = merchant_id.replace("MERCH_", "")
        
        # Base descriptions by category
        desc_templates = {
            'Retail': [f"Purchase at {merchant_name} Store", f"Shopping at {merchant_name}", f"{merchant_name} Retail"],
            'Food': [f"Restaurant {merchant_name}", f"Food delivery {merchant_name}", f"Groceries at {merchant_name}"],
            'Travel': [f"Flight booking {merchant_name}", f"Hotel {merchant_name}", f"Travel agency {merchant_name}"],
            'Entertainment': [f"Movie tickets {merchant_name}", f"Concert {merchant_name}", f"Subscription {merchant_name}"],
            'Services': [f"Services {merchant_name}", f"Consulting {merchant_name}", f"Maintenance {merchant_name}"],
            'Healthcare': [f"Medical {merchant_name}", f"Pharmacy {merchant_name}", f"Health insurance {merchant_name}"],
            'Technology': [f"Tech purchase {merchant_name}", f"Software {merchant_name}", f"Electronics {merchant_name}"],
            'Automotive': [f"Car repair {merchant_name}", f"Gas station {merchant_name}", f"Auto parts {merchant_name}"],
            'Utilities': [f"Utility payment {merchant_name}", f"Phone bill {merchant_name}", f"Internet {merchant_name}"],
            'Other': [f"Payment to {merchant_name}", f"Service from {merchant_name}", f"Purchase {merchant_name}"]
        }
        
        # Choose a description template
        description = random.choice(desc_templates[category])
        
        # Add amount related info
        if amount > 1000:
            description += " LARGE PURCHASE"
        
        # Add suspicious markers for fraudulent transactions
        if is_fraud:
            fraud_markers = [
                " URGENT", " INTERNATIONAL", " VERIFY", " CONFIRM", 
                " SECURITY", " UNUSUAL LOCATION", " FOREIGN CURRENCY", " IMMEDIATE"
            ]
            description += random.choice(fraud_markers)
        
        return description
    
    def generate_transactions(self, num_transactions=1000000, start_date=None, end_date=None):
        """
        Generate a dataset of transactions.
        
        Args:
            num_transactions: Number of transactions to generate
            start_date: Beginning date for transactions
            end_date: Ending date for transactions
            
        Returns:
            DataFrame containing transaction data
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
        if end_date is None:
            end_date = datetime.now()
        
        # Calculate date range
        date_range = (end_date - start_date).days
        
        transactions = []
        
        # Generate normal transactions based on customer patterns
        normal_count = int(num_transactions * (1 - self.fraud_ratio))
        
        print(f"Generating {normal_count} normal transactions...")
        
        for i in range(normal_count):
            if i % 100000 == 0 and i > 0:
                print(f"Generated {i} normal transactions")
                
            # Select customer randomly, weighted by their transaction frequency
            customer_id = random.choice(self.customer_ids)
            pattern = self.customer_patterns[customer_id]
            
            # Select merchant, with higher probability for frequent merchants
            if random.random() < 0.8:  # 80% chance to use frequent merchants
                merchant_id = random.choice(pattern['frequent_merchants'])
            else:
                merchant_id = random.choice(self.merchant_ids)
                
            # Generate transaction amount based on customer's pattern
            amount = max(0.01, np.random.normal(pattern['mean_amount'], pattern['std_amount']))
            amount = round(amount, 2)
            
            # Generate random date within range
            days_offset = random.randint(0, date_range)
            transaction_date = start_date + timedelta(days=days_offset)
            
            # Generate description
            description = self._generate_description(merchant_id, amount, is_fraud=False)
            
            # Create transaction record
            transaction = {
                'transaction_id': f"TXN_{uuid.uuid4().hex}",
                'customer_id': customer_id,
                'merchant_id': merchant_id,
                'date': transaction_date,
                'amount': amount,
                'description': description,
                'category': self.merchant_categories[merchant_id],
                'is_fraud': False
            }
            transactions.append(transaction)
        
        # Generate fraudulent transactions
        fraud_count = int(num_transactions * self.fraud_ratio)
        print(f"Generating {fraud_count} fraudulent transactions...")
        
        for i in range(fraud_count):
            # Select random customer
            customer_id = random.choice(self.customer_ids)
            
            # For fraud, often use uncommon merchants for that customer
            uncommon_merchants = list(set(self.merchant_ids) - set(self.customer_patterns[customer_id]['frequent_merchants']))
            merchant_id = random.choice(uncommon_merchants if uncommon_merchants else self.merchant_ids)
            
            # Generate unusual amount (either very small or very large)
            if random.random() < 0.5:
                # Very small amount (micro-transactions)
                amount = round(random.uniform(0.01, 1.0), 2)
            else:
                # Very large amount
                pattern = self.customer_patterns[customer_id]
                amount = pattern['mean_amount'] + pattern['std_amount'] * random.uniform(5, 15)
                amount = round(amount, 2)
            
            # Generate random date, with higher probability for recent transactions
            recency_bias = random.uniform(0.7, 1.0)  # 70-100% of the date range
            days_offset = int(date_range * recency_bias)
            transaction_date = start_date + timedelta(days=days_offset)
            
            # Generate description with fraud indicators
            description = self._generate_description(merchant_id, amount, is_fraud=True)
            
            # Create transaction record
            transaction = {
                'transaction_id': f"TXN_{uuid.uuid4().hex}",
                'customer_id': customer_id,
                'merchant_id': merchant_id,
                'date': transaction_date,
                'amount': amount,
                'description': description,
                'category': self.merchant_categories[merchant_id],
                'is_fraud': True
            }
            transactions.append(transaction)
        
        # Convert to DataFrame and shuffle
        df = pd.DataFrame(transactions)
        df = df.sample(frac=1).reset_index(drop=True)  # Shuffle
        
        return df
    
    def save_transactions(self, df, filepath):
        """Save transactions to a file"""
        df.to_csv(filepath, index=False)
        print(f"Saved {len(df)} transactions to {filepath}")
        
        # Save a small sample for testing
        sample = df.sample(min(1000, len(df)))
        sample_path = filepath.replace('.csv', '_sample.csv')
        sample.to_csv(sample_path, index=False)
        print(f"Saved sample of {len(sample)} transactions to {sample_path}")
        
        return filepath

# Example usage
if __name__ == "__main__":
    generator = TransactionGenerator(num_customers=50000, num_merchants=5000, fraud_ratio=0.01)
    transactions = generator.generate_transactions(num_transactions=1000000)
    generator.save_transactions(transactions, "data/transactions.csv")