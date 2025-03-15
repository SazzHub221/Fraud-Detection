from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

@dataclass
class ValidationRule:
    """Rule for validating data fields."""
    field: str
    rule_type: str
    parameters: Dict[str, Any]
    description: str

@dataclass
class ValidationResult:
    """Result of data validation."""
    rule: ValidationRule
    is_valid: bool
    failures: List[Dict[str, Any]]
    timestamp: str

class DataValidator:
    """Validator for transaction data."""
    
    def __init__(self, rules_file: Optional[Union[str, Path]] = None) -> None:
        """Initialize data validator.
        
        Args:
            rules_file: Optional path to validation rules file
        """
        self.rules: List[ValidationRule] = []
        if rules_file:
            self.load_rules(rules_file)
        else:
            self._set_default_rules()
            
    def _set_default_rules(self) -> None:
        """Set default validation rules."""
        self.rules = [
            ValidationRule(
                field='transaction_id',
                rule_type='not_null',
                parameters={},
                description='Transaction ID must not be null'
            ),
            ValidationRule(
                field='transaction_id',
                rule_type='unique',
                parameters={},
                description='Transaction ID must be unique'
            ),
            ValidationRule(
                field='amount',
                rule_type='range',
                parameters={'min': 0, 'max': 1000000},
                description='Amount must be between 0 and 1,000,000'
            ),
            ValidationRule(
                field='description',
                rule_type='not_null',
                parameters={},
                description='Description must not be null'
            ),
            ValidationRule(
                field='description',
                rule_type='string_length',
                parameters={'min': 5, 'max': 1000},
                description='Description length must be between 5 and 1000 characters'
            ),
            ValidationRule(
                field='timestamp',
                rule_type='datetime',
                parameters={'format': '%Y-%m-%d %H:%M:%S'},
                description='Timestamp must be in correct datetime format'
            )
        ]
            
    def load_rules(self, rules_file: Union[str, Path]) -> None:
        """Load validation rules from file.
        
        Args:
            rules_file: Path to rules file
        """
        try:
            with open(rules_file, 'r') as f:
                rules_data = json.load(f)
            
            self.rules = [
                ValidationRule(**rule) for rule in rules_data
            ]
            logger.info(f"Loaded {len(self.rules)} validation rules")
        except Exception as e:
            logger.error(f"Error loading validation rules: {e}")
            self._set_default_rules()
            
    def save_rules(self, rules_file: Union[str, Path]) -> None:
        """Save validation rules to file.
        
        Args:
            rules_file: Path to save rules
        """
        try:
            rules_data = [
                {
                    'field': rule.field,
                    'rule_type': rule.rule_type,
                    'parameters': rule.parameters,
                    'description': rule.description
                }
                for rule in self.rules
            ]
            
            with open(rules_file, 'w') as f:
                json.dump(rules_data, f, indent=2)
            logger.info(f"Saved {len(self.rules)} validation rules")
        except Exception as e:
            logger.error(f"Error saving validation rules: {e}")
            
    def add_rule(
        self,
        field: str,
        rule_type: str,
        parameters: Dict[str, Any],
        description: str
    ) -> None:
        """Add a new validation rule.
        
        Args:
            field: Field to validate
            rule_type: Type of validation rule
            parameters: Rule parameters
            description: Rule description
        """
        rule = ValidationRule(
            field=field,
            rule_type=rule_type,
            parameters=parameters,
            description=description
        )
        self.rules.append(rule)
        
    def validate_data(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Validate data against rules.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            List of validation results
        """
        results = []
        
        for rule in self.rules:
            if rule.field not in data.columns:
                results.append(ValidationResult(
                    rule=rule,
                    is_valid=False,
                    failures=[{'error': f"Field {rule.field} not found in data"}],
                    timestamp=datetime.now().isoformat()
                ))
                continue
                
            failures = []
            
            if rule.rule_type == 'not_null':
                null_rows = data[data[rule.field].isnull()]
                if not null_rows.empty:
                    failures.extend([
                        {'row': idx, 'value': None}
                        for idx in null_rows.index
                    ])
                    
            elif rule.rule_type == 'unique':
                duplicates = data[data[rule.field].duplicated()]
                if not duplicates.empty:
                    failures.extend([
                        {'row': idx, 'value': val}
                        for idx, val in duplicates[rule.field].items()
                    ])
                    
            elif rule.rule_type == 'range':
                min_val = rule.parameters.get('min')
                max_val = rule.parameters.get('max')
                invalid_rows = data[
                    (data[rule.field] < min_val) |
                    (data[rule.field] > max_val)
                ]
                if not invalid_rows.empty:
                    failures.extend([
                        {'row': idx, 'value': val}
                        for idx, val in invalid_rows[rule.field].items()
                    ])
                    
            elif rule.rule_type == 'string_length':
                min_len = rule.parameters.get('min', 0)
                max_len = rule.parameters.get('max', float('inf'))
                invalid_rows = data[
                    (data[rule.field].str.len() < min_len) |
                    (data[rule.field].str.len() > max_len)
                ]
                if not invalid_rows.empty:
                    failures.extend([
                        {'row': idx, 'value': val}
                        for idx, val in invalid_rows[rule.field].items()
                    ])
                    
            elif rule.rule_type == 'datetime':
                format = rule.parameters.get('format', '%Y-%m-%d %H:%M:%S')
                invalid_rows = data[
                    pd.to_datetime(data[rule.field], format=format, errors='coerce').isnull()
                ]
                if not invalid_rows.empty:
                    failures.extend([
                        {'row': idx, 'value': val}
                        for idx, val in invalid_rows[rule.field].items()
                    ])
            
            results.append(ValidationResult(
                rule=rule,
                is_valid=len(failures) == 0,
                failures=failures,
                timestamp=datetime.now().isoformat()
            ))
            
        return results
        
    def generate_validation_report(
        self,
        results: List[ValidationResult],
        output_file: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """Generate validation report.
        
        Args:
            results: List of validation results
            output_file: Optional path to save report
            
        Returns:
            Report dictionary
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_rules': len(results),
            'passed_rules': sum(1 for r in results if r.is_valid),
            'failed_rules': sum(1 for r in results if not r.is_valid),
            'results': [
                {
                    'field': r.rule.field,
                    'rule_type': r.rule.rule_type,
                    'description': r.rule.description,
                    'is_valid': r.is_valid,
                    'failure_count': len(r.failures),
                    'failures': r.failures[:10]  # Limit number of failures in report
                }
                for r in results
            ]
        }
        
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    json.dump(report, f, indent=2)
                logger.info(f"Validation report saved to {output_file}")
            except Exception as e:
                logger.error(f"Error saving validation report: {e}")
                
        return report

def validate_embeddings(embeddings: np.ndarray) -> Dict[str, Any]:
    """Validate embedding array.
    
    Args:
        embeddings: Numpy array of embeddings
        
    Returns:
        Dictionary of validation results
    """
    results = {
        'shape': embeddings.shape,
        'dtype': str(embeddings.dtype),
        'has_nan': np.isnan(embeddings).any(),
        'has_inf': np.isinf(embeddings).any(),
        'min_value': float(embeddings.min()),
        'max_value': float(embeddings.max()),
        'mean': float(embeddings.mean()),
        'std': float(embeddings.std())
    }
    
    # Check for zero vectors
    zero_vectors = (embeddings == 0).all(axis=1)
    results['zero_vectors_count'] = int(zero_vectors.sum())
    
    # Check for identical vectors
    unique_vectors = np.unique(embeddings, axis=0)
    results['unique_vectors_count'] = len(unique_vectors)
    results['duplicate_vectors_count'] = len(embeddings) - len(unique_vectors)
    
    return results 