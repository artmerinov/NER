import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset


def IO2df(filepath):
    """
    Trannsforms IO format into dataframe.
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


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return {
            'tokens_ids': np.array(self.data['tokens_ids'][idx]),
            'tags_ids': np.array(self.data['tags_ids'][idx]),
        }

    def __len__(self):
        return len(self.data)
