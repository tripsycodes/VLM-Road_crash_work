"""Configuration management utilities."""
import yaml
import os
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration manager for the project."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to config.yaml file. If None, uses default.
        """
        if config_path is None:
            # Get project root directory
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._update_paths()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _update_paths(self):
        """Update relative paths to absolute paths."""
        project_root = Path(__file__).parent.parent.parent
        
        # Update dataset paths
        if 'dataset' in self.config:
            dataset_config = self.config['dataset']
            if 'root_dir' in dataset_config:
                root = Path(dataset_config['root_dir'])
                if not root.is_absolute():
                    dataset_config['root_dir'] = str(project_root / root)
                else:
                    dataset_config['root_dir'] = str(root)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'dataset.videos_dir')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-like access."""
        return self.get(key)
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in config."""
        return self.get(key) is not None


# Global config instance
_config = None


def get_config(config_path: str = None) -> Config:
    """
    Get global configuration instance.
    
    Args:
        config_path: Path to config file (optional)
        
    Returns:
        Config instance
    """
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config

