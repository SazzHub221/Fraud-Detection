import pandas as pd
import json
import os
import logging
import time
from datetime import datetime
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AlertSystem:
    def __init__(self, alert_log_path='data/alerts/alert_log.json'):
        """
        Initialize the alert system
        
        Args:
            alert_log_path: Path to store alert logs
        """
        self.alert_log_path = alert_log_path
        # Create alerts directory
        os.makedirs(os.path.dirname(alert_log_path), exist_ok=True)
        
        # Load existing alerts if available
        self.past_alerts = []
        if os.path.exists(alert_log_path):
            try:
                with open(alert_log_path, 'r') as f:
                    self.past_alerts = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                logger.warning(f"Could not load existing alerts from {alert_log_path}")
    
    def generate_alert(self, transaction, anomaly_score, reason, evidence=None):
        """
        Generate a single alert for an anomalous transaction
        
        Args:
            transaction: Transaction data
            anomaly_score: Anomaly score (0-1)
            reason: Reason for the alert
            evidence: Supporting evidence (e.g., nearest neighbors)
            
        Returns:
            Alert dictionary
        """
        alert = {
            'alert_id': f"alert_{int(time.time())}_{transaction.get('transaction_id', 'unknown')}",
            'timestamp': datetime.now().isoformat(),
            'transaction_id': transaction.get('transaction_id', 'unknown'),
            'customer_id': transaction.get('customer_id', 'unknown'),
            'amount': transaction.get('amount', 0),
            'merchant': transaction.get('merchant', 'unknown'),
            'description': transaction.get('description', ''),
            'anomaly_score': anomaly_score,
            'reason': reason,
            'status': 'new',
            'evidence': evidence or {}
        }
        
        return alert
    
    def generate_alerts_from_transactions(self, anomalous_transactions):
        """
        Generate alerts for a batch of anomalous transactions
        
        Args:
            anomalous_transactions: DataFrame with anomalous transactions
            
        Returns:
            List of alert dictionaries
        """
        alerts = []
        
        for _, tx in anomalous_transactions.iterrows():
            # Generate evidence based on nearest neighbors
            evidence = {
                'nearest_neighbors': tx.get('nearest_neighbors', []),
                'nearest_distances': tx.get('nearest_distances', []),
                'avg_distance': tx.get('avg_distance', 0)
            }
            
            # Determine reason for alert
            if 'is_fraud' in tx and tx['is_fraud']:
                reason = "Transaction flagged as potentially fraudulent"
            else:
                reason = f"Anomaly score {tx.get('anomaly_score', 0):.2f} exceeds threshold"
            
            # Create alert
            alert = self.generate_alert(
                transaction=tx.to_dict(),
                anomaly_score=tx.get('anomaly_score', 1.0),
                reason=reason,
                evidence=evidence
            )
            
            alerts.append(alert)
        
        return alerts
    
    def save_alerts(self, alerts):
        """
        Save alerts to the alert log
        
        Args:
            alerts: List of alert dictionaries
            
        Returns:
            Path to the alert log
        """
        # Add new alerts to the past alerts
        self.past_alerts.extend(alerts)
        
        # Save to file
        with open(self.alert_log_path, 'w') as f:
            json.dump(self.past_alerts, f, indent=2)
        
        logger.info(f"Saved {len(alerts)} new alerts to {self.alert_log_path}")
        return self.alert_log_path
    
    def get_alerts(self, status=None, days=None, customer_id=None):
        """
        Get alerts based on filters
        
        Args:
            status: Filter by alert status
            days: Filter by days (recent alerts)
            customer_id: Filter by customer ID
            
        Returns:
            List of matching alerts
        """
        filtered_alerts = self.past_alerts.copy()
        
        # Filter by status
        if status:
            filtered_alerts = [a for a in filtered_alerts if a.get('status') == status]
        
        # Filter by days
        if days:
            cutoff = datetime.now() - pd.Timedelta(days=days)
            cutoff_str = cutoff.isoformat()
            filtered_alerts = [a for a in filtered_alerts if a.get('timestamp', '') >= cutoff_str]
        
        # Filter by customer ID
        if customer_id:
            filtered_alerts = [a for a in filtered_alerts if a.get('customer_id') == customer_id]
        
        return filtered_alerts
    
    def update_alert_status(self, alert_id, status, notes=None):
        """
        Update the status of an alert
        
        Args:
            alert_id: ID of the alert to update
            status: New status ('new', 'investigating', 'resolved', 'false_positive')
            notes: Optional notes about the status update
            
        Returns:
            Updated alert or None if not found
        """
        for alert in self.past_alerts:
            if alert.get('alert_id') == alert_id:
                alert['status'] = status
                alert['updated_at'] = datetime.now().isoformat()
                
                if notes:
                    if 'notes' not in alert:
                        alert['notes'] = []
                    
                    alert['notes'].append({
                        'timestamp': datetime.now().isoformat(),
                        'text': notes
                    })
                
                # Save changes
                self.save_alerts([])
                return alert
        
        logger.warning(f"Alert ID {alert_id} not found")
        return None

if __name__ == "__main__":
    # Create an alert system
    alert_system = AlertSystem()
    
    # Generate a sample alert
    sample_tx = {
        'transaction_id': 'tx_123456',
        'customer_id': 'cust_789',
        'amount': 999.99,
        'merchant': 'Unusual Electronics',
        'description': 'Large purchase at new merchant'
    }
    
    alert = alert_system.generate_alert(
        transaction=sample_tx,
        anomaly_score=0.95,
        reason="Unusually large transaction",
        evidence={
            'typical_amount': 50.00,
            'nearest_transactions': ['tx_111', 'tx_222']
        }
    )
    
    # Save the alert
    alert_system.save_alerts([alert])
    
    # Retrieve and update status
    alerts = alert_system.get_alerts(days=1)
    if alerts:
        alert_id = alerts[0]['alert_id']
        updated = alert_system.update_alert_status(
            alert_id=alert_id,
            status='investigating',
            notes='Contacting customer to verify transaction'
        )
        
        if updated:
            logger.info(f"Updated alert status: {updated['status']}")
    
    logger.info("Alert system test complete")