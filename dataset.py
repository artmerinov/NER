import torch
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm
from typing import Dict

from prepare_data import make_padding


class NERDataset(Dataset):
    def __init__(
            self, 
            data: pd.DataFrame,
            idx2word: Dict[int, str],
            char2idx: Dict[str, int],
            max_word_len: int
        ) -> None:

        self.data = data
        self.idx2word = idx2word
        self.char2idx = char2idx
        self.max_word_len = max_word_len

        self.word_ids_data = self._get_word_ids_data()
        self.char_ids_data = self._get_char_ids_data()
        self.tag_ids_data = self._get_tag_ids_data()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'word_ids': self.word_ids_data[idx],
            'char_ids': self.char_ids_data[idx],
            'tag_ids': self.tag_ids_data[idx],
        }
    
    def __len__(self) -> int:
        return len(self.data)
    
    def _get_word_ids_data(self) -> torch.Tensor:
        """
        Creates word ids tensor of size [max_seq_len].
        """
        return torch.tensor(self.data['word_ids'])

    def _get_char_ids_data(self) -> torch.Tensor:
        """
        Creates char ids tensor of size [max_seq_len, max_word_len].
        """
        char_tensor = []
        for word_ids in tqdm(self.data['word_ids']):
            char_matrix = []
            for word_id in word_ids:
                word = self.idx2word[word_id]
                char_seq = [self.char2idx[ch] if ch in self.char2idx 
                            else self.char2idx['UKN'] for ch in word]
                char_seq = make_padding(char_seq, max_len=self.max_word_len)
                char_matrix.append(char_seq)
            char_matrix = torch.tensor(char_matrix).unsqueeze(0)
            char_tensor.append(char_matrix) 
        return torch.cat(char_tensor, dim=0)
    
    def _get_tag_ids_data(self) -> torch.Tensor:
        """
        Creates tag ids tensor of size [max_seq_len].
        """
        return torch.tensor(self.data['tag_ids'])
