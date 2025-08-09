# File: config.py
"""
Configuration management for SLM Agentic Orchestration
Priority P0 - Week 1 implementation with YAML integration
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str
    provider: str
    cost_per_1k: float
    size: str = None

@dataclass
class RoutingConfig:
    """Configuration for task routing."""
    complexity_threshold_slm: float = 0.3
    complexity_threshold_llm: float = 0.7
    default_model: str = "phi-3-mini"
    fallback_model: str = "gpt-4o-mini"

class Config:
    """Enhanced configuration with YAML support."""
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent  # Go up to project root
    DATA_DIR = PROJECT_ROOT / "data"
    CACHE_DIR = PROJECT_ROOT / "cache"
    RESULTS_DIR = PROJECT_ROOT / "results"
    MODELS_DIR = PROJECT_ROOT / "models"
    LOGS_DIR = PROJECT_ROOT / "logs"
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
    
    # Model settings
    DEFAULT_TEMPERATURE = 0.3
    MAX_TOKENS = 500
    
    # Evaluation settings (Phase 1 minimums)
    DEFAULT_N_TASKS = 300  # Per domain
    BATCH_SIZE = 10
    DOMAINS_COUNT = 3      # Focus on 3 domains for Phase 1
    
    # Performance thresholds (Phase 1 targets)
    MAX_LATENCY_MS = 1000
    MAX_ERROR_RATE = 0.1
    TARGET_SLM_COVERAGE = 0.65  # Minimum viable (not 0.7 target)
    TARGET_COST_REDUCTION = 5.0  # Minimum viable (not 8-10x target)
    
    def __init__(self):
        self.setup_directories()
        self.load_yaml_config()
        
    def load_yaml_config(self):
        """Load configuration from default.yaml."""
        config_path = Path(__file__).parent / "default.yaml"
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
                
            # Parse models
            self.slm_models = [
                ModelConfig(**model) for model in yaml_config.get('models', {}).get('slms', [])
            ]
            self.llm_models = [
                ModelConfig(**model) for model in yaml_config.get('models', {}).get('llms', [])
            ]
            
            # Parse routing
            routing_data = yaml_config.get('routing', {})
            self.routing = RoutingConfig(**routing_data)
            
            # Parse domains
            self.evaluation_domains = yaml_config.get('evaluation', {}).get('domains', [])
    
    def get_api_key(self, provider: str) -> str:
        """Get API key for provider."""
        key_mapping = {
            'openai': self.OPENAI_API_KEY,
            'anthropic': self.ANTHROPIC_API_KEY,
            'google': self.GEMINI_API_KEY,
            'microsoft': self.HUGGINGFACE_API_KEY,  # For Phi-3 via HF
            'meta': self.HUGGINGFACE_API_KEY,       # For Llama via HF
        }
        
        api_key = key_mapping.get(provider)
        if not api_key:
            raise ValueError(f"Missing API key for provider: {provider}")
        
        return api_key
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories."""
        dirs = [cls.DATA_DIR, cls.CACHE_DIR, cls.RESULTS_DIR, cls.MODELS_DIR, cls.LOGS_DIR]
        for dir_path in dirs:
            dir_path.mkdir(exist_ok=True, parents=True)

# Global config instance
config = Config()