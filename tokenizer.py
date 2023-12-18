from collections import Counter
from typing import Dict
import json


def make_word_vocab(filepath: str, 
                    support: int,
                    save_path: str = None
                    ) -> Dict[str, int]:
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

    top_words = [w for w, cnt in word_cntr.most_common() 
                 if cnt >= support]

    # create vocab
    word2idx = {}
    word2idx['PAD'] = 0
    word2idx['UKN'] = 1

    for i, token in enumerate(top_words):
        word2idx[token] = i + 2

    # save vocab if save_path is provided
    if save_path is not None:
        with open(save_path, 'w') as f:
            json.dump(word2idx, f, indent=4)
    
    return word2idx


def make_char_vocab(filepath: str, 
                    support: int, 
                    save_path: str = None
                    ) -> Dict[str, int]:
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

    top_chars = [ch for ch, cnt in char_cntr.most_common() 
                 if cnt >= support]

    # create vocab
    char2idx = {}
    char2idx['PAD'] = 0
    char2idx['UKN'] = 1

    for i, ch in enumerate(top_chars):
        char2idx[ch] = i + 2

    # save vocab if save_path is provided
    if save_path is not None:
        with open(save_path, 'w') as f:
            json.dump(char2idx, f, indent=4)

    return char2idx
