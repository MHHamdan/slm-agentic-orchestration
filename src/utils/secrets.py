"""
Secure secret management utilities for SLM Agentic Orchestration.
"""

import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def get_secret(secret_name: str, required: bool = True) -> Optional[str]:
    """Get a secret from environment variable or file.
    
    Args:
        secret_name: Name of the secret (environment variable name)
        required: Whether the secret is required
        
    Returns:
        The secret value or None if not found and not required
    """
    # Try environment variable first
    env_value = os.getenv(secret_name)
    if env_value:
        return env_value
    
    # Try secret file in secrets directory
    secrets_dir = Path("secrets")
    for subdir in ["api_keys", "tokens", "database", "certificates", ""]:
        secret_file = secrets_dir / subdir / f"{secret_name.lower()}.key"
        if secret_file.exists():
            try:
                return secret_file.read_text().strip()
            except Exception as e:
                logger.error(f"Error reading secret file {secret_file}: {e}")
    
    if required:
        raise ValueError(f"Secret '{secret_name}' not found")
    
    return None


def get_api_key(provider: str) -> str:
    """Get API key for a specific provider."""
    key_name = f"{provider.upper()}_API_KEY"
    return get_secret(key_name)


def load_env_file(env_file: Path = None):
    """Load environment variables from a .env file."""
    env_file = env_file or Path(".env")
    
    if not env_file.exists():
        return
    
    try:
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
    except Exception as e:
        logger.error(f"Error loading {env_file}: {e}")


# Auto-load .env file if it exists
if Path(".env").exists():
    load_env_file()