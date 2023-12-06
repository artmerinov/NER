import torch
import torch.nn as nn
from torchcrf import CRF


class BiLSTM_CRF(nn.Module):
    def __init__(
            self, 
            embed_size: int, 
            hidden_size: int, 
            dropout: float, 
            token_voc_size: int, 
            tag_voc_size: int
    ) -> None:
        super(BiLSTM_CRF, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.token_voc_size = token_voc_size
        self.tag_voc_size = tag_voc_size
        
        self.token_embedding = nn.Embedding(self.token_voc_size, self.embed_size)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.tag_voc_size)
        self.crf = CRF(self.tag_voc_size, batch_first=True)
        
        self.init_weights()
        
    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor, tags: torch.Tensor) -> torch.Tensor:
        embedding = self.token_embedding(x)
        outputs, hidden = self.lstm(embedding)
        outputs = self.dropout(outputs)
        outputs = self.fc(outputs)
        outputs = self.crf.decode(outputs)
        return outputs

    def neg_log_likelihood(self, x: torch.Tensor, tags: torch.Tensor) -> torch.Tensor:
        embedding = self.token_embedding(x)
        outputs, hidden = self.lstm(embedding)
        outputs = self.dropout(outputs)
        outputs = self.fc(outputs)
        # find the mask for ignoring padding (index 0)
        mask = (tags > 0).bool()
        return - self.crf(outputs, tags, mask=mask)
