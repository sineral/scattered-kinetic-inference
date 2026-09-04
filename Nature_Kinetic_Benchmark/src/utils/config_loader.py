"""
Configuration loader for training neural networks.

This module provides utilities to load training configurations from YAML files.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file
        
    Returns
    -------
    Dict[str, Any]
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_model_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract model parameters from configuration.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary
        
    Returns
    -------
    Dict[str, Any]
        Model parameters dictionary
    """
    model_name = config['model']['name'].lower()
    
    if model_name == 'deepsets':
        return config['model']['deepsets']
    if model_name == 'settransformer':
        return config['model']['settransformer']
    raise ValueError(f"Unknown model type: {model_name}")


def print_config(config: Dict[str, Any]) -> None:
    """Print configuration in a readable format.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary
    """
    print("=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    
    # Data configuration
    print("Data:")
    for key, value in config['data'].items():
        print(f"  {key}: {value}")
    
    # Model configuration
    print(f"\nModel: {config['model']['name']}")
    model_params = get_model_params(config)
    for key, value in model_params.items():
        print(f"  {key}: {value}")
    
    # Training configuration
    print("\nTraining:")
    for key, value in config['training'].items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value}")
        else:
            print(f"  {key}: {value}")
    
    print("=" * 60)
