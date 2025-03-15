from typing import Dict, List, Optional, Union
import json
import os
from datetime import datetime
from pathlib import Path
import logging
import shutil
import yaml

logger = logging.getLogger(__name__)

class ModelVersion:
    """Represents a specific version of a model."""
    
    def __init__(
        self,
        model_id: str,
        version: str,
        model_type: str,
        created_at: str,
        metrics: Dict[str, float],
        parameters: Dict[str, Union[str, float, int, bool]],
        artifacts_path: Path,
        description: Optional[str] = None
    ) -> None:
        self.model_id = model_id
        self.version = version
        self.model_type = model_type
        self.created_at = created_at
        self.metrics = metrics
        self.parameters = parameters
        self.artifacts_path = artifacts_path
        self.description = description

    def to_dict(self) -> Dict:
        """Convert model version to dictionary."""
        return {
            'model_id': self.model_id,
            'version': self.version,
            'model_type': self.model_type,
            'created_at': self.created_at,
            'metrics': self.metrics,
            'parameters': self.parameters,
            'artifacts_path': str(self.artifacts_path),
            'description': self.description
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelVersion':
        """Create ModelVersion instance from dictionary."""
        return cls(
            model_id=data['model_id'],
            version=data['version'],
            model_type=data['model_type'],
            created_at=data['created_at'],
            metrics=data['metrics'],
            parameters=data['parameters'],
            artifacts_path=Path(data['artifacts_path']),
            description=data.get('description')
        )

class ModelRegistry:
    """Registry for managing model versions and their artifacts."""
    
    def __init__(self, registry_path: Union[str, Path]) -> None:
        """Initialize model registry.
        
        Args:
            registry_path: Path to registry directory
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.registry_path / 'registry_metadata.json'
        self.models: Dict[str, List[ModelVersion]] = {}
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load registry metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                
                for model_id, versions in data.items():
                    self.models[model_id] = [
                        ModelVersion.from_dict(v) for v in versions
                    ]
                logger.info(f"Loaded metadata for {len(self.models)} models")
            except Exception as e:
                logger.error(f"Error loading registry metadata: {e}")
                self.models = {}

    def _save_metadata(self) -> None:
        """Save registry metadata to disk."""
        try:
            data = {
                model_id: [v.to_dict() for v in versions]
                for model_id, versions in self.models.items()
            }
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info("Saved registry metadata")
        except Exception as e:
            logger.error(f"Error saving registry metadata: {e}")
            raise

    def register_model(
        self,
        model_id: str,
        model_type: str,
        metrics: Dict[str, float],
        parameters: Dict[str, Union[str, float, int, bool]],
        artifacts_dir: Union[str, Path],
        description: Optional[str] = None
    ) -> ModelVersion:
        """Register a new model version.
        
        Args:
            model_id: Unique identifier for the model
            model_type: Type of model (e.g., 'embedding', 'anomaly_detection')
            metrics: Performance metrics
            parameters: Model parameters
            artifacts_dir: Directory containing model artifacts
            description: Optional description
            
        Returns:
            New ModelVersion instance
        """
        # Generate version number
        existing_versions = self.models.get(model_id, [])
        version = f"v{len(existing_versions) + 1}"
        
        # Create version directory
        version_dir = self.registry_path / model_id / version
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy artifacts
        artifacts_path = version_dir / 'artifacts'
        if os.path.exists(artifacts_dir):
            shutil.copytree(artifacts_dir, artifacts_path, dirs_exist_ok=True)
        
        # Save parameters
        with open(version_dir / 'parameters.yaml', 'w') as f:
            yaml.dump(parameters, f)
        
        # Create model version
        model_version = ModelVersion(
            model_id=model_id,
            version=version,
            model_type=model_type,
            created_at=datetime.now().isoformat(),
            metrics=metrics,
            parameters=parameters,
            artifacts_path=artifacts_path,
            description=description
        )
        
        # Update registry
        if model_id not in self.models:
            self.models[model_id] = []
        self.models[model_id].append(model_version)
        
        # Save metadata
        self._save_metadata()
        
        logger.info(f"Registered {model_id} {version}")
        return model_version

    def get_latest_version(self, model_id: str) -> Optional[ModelVersion]:
        """Get the latest version of a model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Latest ModelVersion or None if not found
        """
        versions = self.models.get(model_id, [])
        return versions[-1] if versions else None

    def get_version(self, model_id: str, version: str) -> Optional[ModelVersion]:
        """Get a specific version of a model.
        
        Args:
            model_id: Model identifier
            version: Version string
            
        Returns:
            ModelVersion or None if not found
        """
        versions = self.models.get(model_id, [])
        for v in versions:
            if v.version == version:
                return v
        return None

    def get_best_model(
        self,
        model_id: str,
        metric: str,
        higher_is_better: bool = True
    ) -> Optional[ModelVersion]:
        """Get the best performing model version based on a metric.
        
        Args:
            model_id: Model identifier
            metric: Metric to compare
            higher_is_better: Whether higher metric values are better
            
        Returns:
            Best performing ModelVersion or None if not found
        """
        versions = self.models.get(model_id, [])
        if not versions:
            return None
            
        return max(
            versions,
            key=lambda v: v.metrics.get(metric, float('-inf')) if higher_is_better
                        else -v.metrics.get(metric, float('inf'))
        )

    def list_models(self) -> List[str]:
        """Get list of all registered models.
        
        Returns:
            List of model identifiers
        """
        return list(self.models.keys())

    def list_versions(self, model_id: str) -> List[ModelVersion]:
        """Get all versions of a model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            List of ModelVersion instances
        """
        return self.models.get(model_id, []) 