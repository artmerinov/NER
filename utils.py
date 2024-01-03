import yaml
import torch
import random
import json
import numpy as np
from typing import Dict, List


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
        json_file = json.load(file)
    return json_file


def io2bio(tags: List[str]) -> List[str]:
    """
    Convert list of tags in IO format into BIO format 
    (the format expected by seqeval library).
    """
    converted_sequence = []
    current_entity = None

    for tag in tags:
        if tag != 'O':
            if current_entity is None:
                converted_sequence.append('B-' + tag)
                current_entity = tag
            elif current_entity == tag:
                converted_sequence.append('I-' + tag)
            else:
                converted_sequence.append('B-' + tag)
                current_entity = tag
        else:
            converted_sequence.append('O')
            current_entity = None
            
    return converted_sequence


def make_padding(sequence: List[int], max_len: int = 100) -> List[int]:
    """
    Makes sequence to be fixed size using padding.
    """
    if len(sequence) >= max_len:
        return sequence[:max_len]
    sequence.extend([0]*(max_len - len(sequence)))
    return sequence
