import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from typing import List, Dict


class NERDataset(Dataset):
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'tokens_ids': torch.tensor(self.data['tokens_ids'][idx]),
            'tags_ids': torch.tensor(self.data['tags_ids'][idx]),
        }
    
    def __len__(self) -> int:
        return len(self.data)


def io2df(filepath: str) -> pd.DataFrame:
    """
    Reads file in IO format and transform it into pandas dataframe.
    """
    num_lines = sum(1 for _ in open(filepath, encoding="utf-8"))
    id = 0

    with open(filepath, "r", encoding="utf-8") as f:
        
        tokens = []
        tags_fine_grained = []
        tags_coarse_grained = []

        records = []

        for line in tqdm(f, total=num_lines):
            line = line.strip().split()

            if line:
                token, tag_fine_grained = line
                tag_coarse_grained = tag_fine_grained.split("-")[0]

                tokens.append(token)
                tags_fine_grained.append(tag_fine_grained)
                tags_coarse_grained.append(tag_coarse_grained)

            # end of sentence
            elif tokens:
                record = {
                    "id": id, 
                    "tokens": tokens, 
                    "tags_fine_grained": tags_fine_grained, 
                    'tags_coarse_grained': tags_coarse_grained
                }
                records.append(record)
                tokens = []
                tags_fine_grained = []
                tags_coarse_grained = []
                id += 1
        
        # take the last sentence
        if tokens:
            record = {
                "id": id, 
                "tokens": tokens, 
                "tags_fine_grained": tags_fine_grained, 
                'tags_coarse_grained': tags_coarse_grained
            }
            records.append(record)

    titles = pd.DataFrame(records)
    return titles


def io2bio(tags: List[str]) -> List[str]:
    """
    Convert list of tags in IO format into BIO format 
    (the format expected by seqeval).
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


def padding(ids: List[int], max_len: int = 100) -> List[int]:
    """
    Makes sequence of ids to be fixed size using padding.
    """
    if len(ids) >= max_len:
        return ids[:max_len]
    ids.extend([0]*(max_len - len(ids)))
    return ids