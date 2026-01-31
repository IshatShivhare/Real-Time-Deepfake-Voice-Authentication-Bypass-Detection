import yaml
import os
from pathlib import Path

_CONFIG = None

def load_config(config_path="config.yaml"):
    """
    Load the global configuration from a YAML file.
    """
    global _CONFIG
    
    # If path is relative, try to find it relative to project root
    if not os.path.isabs(config_path):
        # Assuming we are running from project root or src is one level deep
        # Ideally, we find the root containing config.yaml
        current_dir = Path.cwd()
        potential_path = current_dir / config_path
        
        if not potential_path.exists():
            # Try looking up one level (if running from src/)
            potential_path = current_dir.parent / config_path
            
        config_path = str(potential_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, 'r') as f:
        _CONFIG = yaml.safe_load(f)
        
    return _CONFIG

def get_config():
    """
    Get the loaded configuration. Loads default if not initialized.
    """
    global _CONFIG
    if _CONFIG is None:
        return load_config()
    return _CONFIG
