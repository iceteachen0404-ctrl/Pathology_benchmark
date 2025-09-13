import json
import os
import pandas as pd

class DatasetManager:
    def __init__(self, index_file):
        with open(index_file, "r") as f:
            self.index_data = json.load(f)

    def get_dataset_paths(self, dataset_name):
        if dataset_name not in self.index_data:
            raise ValueError(f"Dataset {dataset_name} not found in index file.")
        return self.index_data[dataset_name]

    def load_label(self, dataset_name):
        paths = self.get_dataset_paths(dataset_name)
        if "labels" in paths and os.path.exists(paths["labels"]):
            return pd.read_csv(paths["labels"])
        return None