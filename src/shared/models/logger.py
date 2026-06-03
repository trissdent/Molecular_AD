# logger.py
# Experiment logging utilities.

import os
import json
import yaml
import shutil
from datetime import datetime
from pathlib import Path


class ExperimentLogger:
    """
    Log experiment config, results, and model info.
    """
    def __init__(self, log_dir="./logs/", experiment_name=None):
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        self.experiment_dir = Path(log_dir) / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
    
    def get_experiment_dir(self):
        """Return experiment directory path (for Lightning logger)."""
        return str(self.experiment_dir)
    
    def log_config(self, config_path):
        """Copy config file to experiment folder."""
        shutil.copy(config_path, self.experiment_dir / "config.yml")
    
    def log_config_dict(self, config_dict):
        """Save config dict to experiment folder."""
        with open(self.experiment_dir / "config.yml", 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def log_model_info(self, model):
        """Save model info."""
        info = {
            "architecture": model.__class__.__name__,
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
        with open(self.experiment_dir / "model_info.json", 'w') as f:
            json.dump(info, f, indent=2)
    
    def log_results(self, results: dict):
        """Save final results."""
        with open(self.experiment_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2)
    
    def log_message(self, message: str):
        """Append message to log file."""
        with open(self.experiment_dir / "log.txt", 'a') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {message}\n")