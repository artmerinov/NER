import yaml
import torch
import random
import json
import numpy as np
from typing import Dict


class Config:
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        for key, value in config_dict.items():
            setattr(self, key.upper(), value)


def set_random_seed(seed: int) -> None:
    """
    Fixes random state for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_json(file_path: str) -> Dict[str, int]:
    """
    Loads JSON file.
    """
    with open(file_path, 'r') as file:
        ner_config = json.load(file)
    return ner_config