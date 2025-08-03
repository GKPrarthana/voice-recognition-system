import yaml
import os
from pathlib import Path

class ConfigurationManager:
    def __init__(self, config_path="config/config.yaml", params_path="config/params.yaml"):
        self.config = self._load_yaml(config_path)
        self.params = self._load_yaml(params_path)

    def _load_yaml(self, path):
        try:
            with open(path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found at {path}")

    def get_data_config(self):
        return self.config['data']

    def get_model_config(self):
        return self.config['model'], self.params['model']

    def get_training_config(self):
        return self.config['training'], self.params['preprocessing']