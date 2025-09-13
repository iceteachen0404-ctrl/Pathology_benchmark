import yaml

class ModelConfigManager:
    def __init__(self, config_file: str):
        with open(config_file, "r") as f:
            self.model_configs = yaml.safe_load(f)

    def get_model_config(self, model_name: str):
        if model_name not in self.model_configs:
            raise ValueError(f"Model '{model_name}' not found in models.yaml. "
                             f"Available models: {list(self.model_configs.keys())}")
        return self.model_configs[model_name]