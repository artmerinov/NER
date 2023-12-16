import pandas as pd
from tqdm import tqdm
from collections import Counter
import torch
from torch.utils.data import Dataset
from typing import List, Dict


class NERDataset(Dataset):
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'word_ids': torch.tensor(self.data['word_ids'][idx]),
            'tag_ids': torch.tensor(self.data['tag_ids'][idx]),
        }
    
    def __len__(self) -> int:
        return len(self.data)


def io2df(filepath: str) -> pd.DataFrame:
    """
    Reads file in IO format and transform it into pandas dataframe.
    """
    id = 0

    with open(filepath, "r", encoding="utf-8") as f:
        
        words = []
        tags_fine_grained = []
        tags_coarse_grained = []

        records = []

        for line in f:
            line = line.strip().split()

            if line:
                word, tag_fine_grained = line
                tag_coarse_grained = tag_fine_grained.split("-")[0]

                words.append(word)
                tags_fine_grained.append(tag_fine_grained)
                tags_coarse_grained.append(tag_coarse_grained)

            # end of sentence
            elif words:
                record = {
                    "id": id, 
                    "words": words, 
                    "tags_fine_grained": tags_fine_grained, 
                    'tags_coarse_grained': tags_coarse_grained
                }
                records.append(record)
                words = []
                tags_fine_grained = []
                tags_coarse_grained = []
                id += 1
        
        # take the last sentence
        if words:
            record = {
                "id": id, 
                "words": words, 
                "tags_fine_grained": tags_fine_grained, 
                'tags_coarse_grained': tags_coarse_grained
            }
            records.append(record)

    titles = pd.DataFrame(records)
    return titles


def make_word_vocab(filepath: str, support: int = 5) -> Dict[str, int]:
    """
    Creates word-level vocabulary.
    """
    word_cntr = Counter()

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().split()
            if line:
                word, _ = line
                word_cntr[word] += 1

    top_words = [w for w, cnt in word_cntr.most_common() if cnt >= support]

    word2idx = {token: i + 2 for i, token in enumerate(top_words)}
    word2idx['PAD'] = 0
    word2idx['UKN'] = 1
    
    return word2idx


def make_char_vocab(filepath: str, support: int = 100) -> Dict[str, int]:
    """
    Creates character-level vocabulary.
    """
    char_cntr = Counter()

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().split()
            if line:
                word, _ = line
                for ch in word:
                    char_cntr[ch] += 1

    top_chars = [ch for ch, cnt in char_cntr.most_common() if cnt >= support]

    char2idx = {ch: i + 2 for i, ch in enumerate(top_chars)}
    char2idx['PAD'] = 0
    char2idx['UKN'] = 1

    return char2idx


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