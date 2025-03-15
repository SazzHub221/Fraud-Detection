from typing import Dict, List, Optional, Union
import time
import logging
import json
from pathlib import Path
from datetime import datetime
import psutil
import numpy as np
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """System performance metrics."""
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    timestamp: str

@dataclass
class PipelineMetrics:
    """Pipeline performance metrics."""
    preprocessing_time: float
    embedding_time: float
    detection_time: float
    total_time: float
    transactions_processed: int
    anomalies_detected: int
    alerts_generated: int
    timestamp: str

class PerformanceMonitor:
    """Monitor system and pipeline performance."""
    
    def __init__(self, metrics_dir: Union[str, Path]) -> None:
        """Initialize performance monitor.
        
        Args:
            metrics_dir: Directory to store metrics
        """
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.system_metrics: List[SystemMetrics] = []
        self.pipeline_metrics: List[PipelineMetrics] = []
        self._start_time = time.time()
        
    def start_monitoring(self) -> None:
        """Start monitoring system metrics."""
        self._start_time = time.time()
        self._collect_system_metrics()
        
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        metrics = SystemMetrics(
            cpu_percent=psutil.cpu_percent(),
            memory_percent=psutil.virtual_memory().percent,
            disk_usage_percent=psutil.disk_usage('/').percent,
            timestamp=datetime.now().isoformat()
        )
        self.system_metrics.append(metrics)
        return metrics
        
    def record_pipeline_metrics(
        self,
        preprocessing_time: float,
        embedding_time: float,
        detection_time: float,
        transactions_processed: int,
        anomalies_detected: int,
        alerts_generated: int
    ) -> None:
        """Record pipeline performance metrics.
        
        Args:
            preprocessing_time: Time spent on preprocessing
            embedding_time: Time spent on embedding generation
            detection_time: Time spent on anomaly detection
            transactions_processed: Number of transactions processed
            anomalies_detected: Number of anomalies detected
            alerts_generated: Number of alerts generated
        """
        metrics = PipelineMetrics(
            preprocessing_time=preprocessing_time,
            embedding_time=embedding_time,
            detection_time=detection_time,
            total_time=time.time() - self._start_time,
            transactions_processed=transactions_processed,
            anomalies_detected=anomalies_detected,
            alerts_generated=alerts_generated,
            timestamp=datetime.now().isoformat()
        )
        self.pipeline_metrics.append(metrics)
        self._save_metrics()
        
    def _save_metrics(self) -> None:
        """Save metrics to disk."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save system metrics
        system_metrics_file = self.metrics_dir / f'system_metrics_{timestamp}.json'
        with open(system_metrics_file, 'w') as f:
            json.dump(
                [asdict(m) for m in self.system_metrics],
                f,
                indent=2
            )
            
        # Save pipeline metrics
        pipeline_metrics_file = self.metrics_dir / f'pipeline_metrics_{timestamp}.json'
        with open(pipeline_metrics_file, 'w') as f:
            json.dump(
                [asdict(m) for m in self.pipeline_metrics],
                f,
                indent=2
            )
            
    def get_summary_stats(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics of metrics.
        
        Returns:
            Dictionary containing summary statistics
        """
        if not self.pipeline_metrics:
            return {}
            
        pipeline_stats = {
            'preprocessing_time': {
                'mean': np.mean([m.preprocessing_time for m in self.pipeline_metrics]),
                'std': np.std([m.preprocessing_time for m in self.pipeline_metrics]),
                'min': np.min([m.preprocessing_time for m in self.pipeline_metrics]),
                'max': np.max([m.preprocessing_time for m in self.pipeline_metrics])
            },
            'embedding_time': {
                'mean': np.mean([m.embedding_time for m in self.pipeline_metrics]),
                'std': np.std([m.embedding_time for m in self.pipeline_metrics]),
                'min': np.min([m.embedding_time for m in self.pipeline_metrics]),
                'max': np.max([m.embedding_time for m in self.pipeline_metrics])
            },
            'detection_time': {
                'mean': np.mean([m.detection_time for m in self.pipeline_metrics]),
                'std': np.std([m.detection_time for m in self.pipeline_metrics]),
                'min': np.min([m.detection_time for m in self.pipeline_metrics]),
                'max': np.max([m.detection_time for m in self.pipeline_metrics])
            },
            'total_time': {
                'mean': np.mean([m.total_time for m in self.pipeline_metrics]),
                'std': np.std([m.total_time for m in self.pipeline_metrics]),
                'min': np.min([m.total_time for m in self.pipeline_metrics]),
                'max': np.max([m.total_time for m in self.pipeline_metrics])
            },
            'transactions_processed': {
                'total': sum(m.transactions_processed for m in self.pipeline_metrics),
                'mean': np.mean([m.transactions_processed for m in self.pipeline_metrics])
            },
            'anomalies_detected': {
                'total': sum(m.anomalies_detected for m in self.pipeline_metrics),
                'mean': np.mean([m.anomalies_detected for m in self.pipeline_metrics])
            },
            'alerts_generated': {
                'total': sum(m.alerts_generated for m in self.pipeline_metrics),
                'mean': np.mean([m.alerts_generated for m in self.pipeline_metrics])
            }
        }
        
        system_stats = {
            'cpu_percent': {
                'mean': np.mean([m.cpu_percent for m in self.system_metrics]),
                'std': np.std([m.cpu_percent for m in self.system_metrics]),
                'min': np.min([m.cpu_percent for m in self.system_metrics]),
                'max': np.max([m.cpu_percent for m in self.system_metrics])
            },
            'memory_percent': {
                'mean': np.mean([m.memory_percent for m in self.system_metrics]),
                'std': np.std([m.memory_percent for m in self.system_metrics]),
                'min': np.min([m.memory_percent for m in self.system_metrics]),
                'max': np.max([m.memory_percent for m in self.system_metrics])
            },
            'disk_usage_percent': {
                'mean': np.mean([m.disk_usage_percent for m in self.system_metrics]),
                'std': np.std([m.disk_usage_percent for m in self.system_metrics]),
                'min': np.min([m.disk_usage_percent for m in self.system_metrics]),
                'max': np.max([m.disk_usage_percent for m in self.system_metrics])
            }
        }
        
        return {
            'pipeline': pipeline_stats,
            'system': system_stats
        }

class MetricsLogger:
    """Logger for tracking and storing metrics."""
    
    def __init__(self, log_dir: Union[str, Path]) -> None:
        """Initialize metrics logger.
        
        Args:
            log_dir: Directory to store logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        self.logger = logging.getLogger('metrics')
        self.logger.setLevel(logging.INFO)
        
        # Add file handler
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.log_dir / f'metrics_{timestamp}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Add formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
    def log_metric(
        self,
        metric_name: str,
        value: Union[int, float],
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Log a metric value.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            tags: Optional tags for the metric
        """
        message = {
            'metric': metric_name,
            'value': value,
            'timestamp': datetime.now().isoformat()
        }
        if tags:
            message['tags'] = tags
            
        self.logger.info(json.dumps(message))
        
    def log_event(
        self,
        event_name: str,
        description: str,
        severity: str = 'INFO',
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Log an event.
        
        Args:
            event_name: Name of the event
            description: Event description
            severity: Event severity level
            tags: Optional tags for the event
        """
        message = {
            'event': event_name,
            'description': description,
            'severity': severity,
            'timestamp': datetime.now().isoformat()
        }
        if tags:
            message['tags'] = tags
            
        if severity == 'ERROR':
            self.logger.error(json.dumps(message))
        elif severity == 'WARNING':
            self.logger.warning(json.dumps(message))
        else:
            self.logger.info(json.dumps(message)) 