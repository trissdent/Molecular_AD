# config.py
import os
import yaml


class ConfigException(Exception):
    def __init__(self, message="Config error"):
        self.message = message
        super().__init__(self.message)


class ConfigReader:
    def __init__(self, sections: dict):
        for section, content in sections.items():
            if isinstance(content, dict):
                setattr(self, section, ConfigReader(content))
            else:
                setattr(self, section, content)

    @classmethod
    def load(cls, config_path: str):
        if not config_path.endswith((".yml", ".yaml")):
            raise ConfigException("Only support yaml or yml file.")
        
        with open(config_path, 'r') as f:
            return cls(yaml.safe_load(f))
    
    @classmethod
    def merge_dict(cls, default_path: str, experiment_path: str = None) -> dict:  # ty:ignore[invalid-parameter-default]
        """defaults.yaml with an optional override file applied on top, as a plain dict."""
        with open(default_path, 'r') as f:
            merged = yaml.safe_load(f)

        if experiment_path:
            if not os.path.exists(experiment_path):
                raise ConfigException(f"Override file not found: {experiment_path}")
            with open(experiment_path, 'r') as f:
                merged = cls._deep_merge(merged, yaml.safe_load(f) or {})

        return merged

    @classmethod
    def merge(cls, default_path: str, experiment_path: str = None):  # ty:ignore[invalid-parameter-default]
        return cls(cls.merge_dict(default_path, experiment_path))
    
    @staticmethod
    def _deep_merge(default: dict, override: dict) -> dict:
        result = default.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigReader._deep_merge(result[key], value)
            else:
                result[key] = value
        return result