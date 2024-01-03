import pandas as pd
import argparse
from typing import List, Dict

from utils import make_padding


def preprocess_raw_data(config: argparse.Namespace, 
                        tag2idx: Dict[str, int],
                        word2idx: Dict[str, int],
                        ) -> List[pd.DataFrame]:
    """
    Load and preprocesss raw data.
    """
    # load raw data
    tr_titles = read_raw_data(filepath=config.tr_path)
    va_titles = read_raw_data(filepath=config.va_path)

    # apply tag2idx
    tr_titles['tag_ids'] = tr_titles['tags_fine_grained'].transform(lambda x: [tag2idx[tag] for tag in x])
    va_titles['tag_ids'] = va_titles['tags_fine_grained'].transform(lambda x: [tag2idx[tag] for tag in x])

    # apply word2idx
    tr_titles['word_ids'] = tr_titles['words'].transform(lambda x: [word2idx[w] if w in word2idx else word2idx['UKN'] for w in x])
    va_titles['word_ids'] = va_titles['words'].transform(lambda x: [word2idx[w] if w in word2idx else word2idx['UKN'] for w in x])

    # apply padding
    tr_titles['word_ids'] = tr_titles['word_ids'].transform(make_padding, max_len=config.max_seq_len)
    tr_titles['tag_ids'] = tr_titles['tag_ids'].transform(make_padding, max_len=config.max_seq_len)

    va_titles['word_ids'] = va_titles['word_ids'].transform(make_padding, max_len=config.max_seq_len)
    va_titles['tag_ids'] = va_titles['tag_ids'].transform(make_padding, max_len=config.max_seq_len)

    return tr_titles, va_titles


def read_raw_data(filepath: str) -> pd.DataFrame:
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
