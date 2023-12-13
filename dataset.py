import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from typing import List, Dict


class NERDataset(Dataset):
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        return {
            'tokens_ids': np.array(self.data['tokens_ids'][idx]),
            'tags_ids': np.array(self.data['tags_ids'][idx]),
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
        
        tokens, tags = [], []
        records = []

        for line in tqdm(f, total=num_lines):
            line = line.strip().split()

            if line:
                token, fine_tag = line
                tag = fine_tag.split("-")[0]

                tokens.append(token)
                tags.append(tag)

            # end of sentence
            elif tokens:
                record = {"id": id, "tokens": tokens, "tags": tags}
                records.append(record)
                tokens, tags = [], []
                id += 1

        # take the last sentence
        if tokens:
            record = {"id": id, "tokens": tokens, "tags": tags,}
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