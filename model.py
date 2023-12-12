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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predicts label sequence.
        """
        embedding = self.token_embedding(x)
        outputs, hidden = self.lstm(embedding)
        outputs = self.dropout(outputs)
        outputs = self.fc(outputs)
        outputs = self.crf.decode(outputs)
        return torch.tensor(outputs)

    def loss_fn(self, 
                x: torch.Tensor, 
                tags: torch.Tensor, 
                mask: torch.Tensor) -> torch.Tensor:
        """
        Calculates negative log-likelihood loss (NLL).
        """
        embedding = self.token_embedding(x)
        outputs, hidden = self.lstm(embedding)
        outputs = self.dropout(outputs)
        outputs = self.fc(outputs)
        return -self.crf(outputs, tags, mask=mask)
    
    def regularization_loss_fn(self, lam=1e-3, alpha=0.5):
        """
        Calculates regularization loss function.
        """
        weigths = [
            self.token_embedding.weight,
            self.lstm.weight_ih_l0,
            self.lstm.weight_hh_l0,
            self.lstm.weight_ih_l0_reverse,
            self.lstm.weight_hh_l0_reverse,
            self.fc.weight,
            self.crf.transitions
        ]
        elastic = torch.tensor([alpha * self._l1_penalty(w) + (1 - alpha) * self._l2_penalty(w) for w in weigths])
        loss = lam * torch.sum(elastic)
        return loss
    
    @staticmethod
    def _l1_penalty(v):
        return torch.sum(torch.abs(v))
    
    @staticmethod
    def _l2_penalty(v):
        return torch.sum(v**2)
        
    def init_weights(self):
        """
        Xavier/Glorot initialization for weights and 
        zero initialization for biases.
        """
        learnable_named_parameters = [(name, p) for name, p in self.named_parameters() if p.requires_grad]
        
        for name, p in learnable_named_parameters:
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                print(f'{name:<30} initialized w with Xavier {" "*10} parameters #: {p.numel()}', flush=True)
            else:
                nn.init.zeros_(p)
                print(f'{name:<30} initialized b with zero   {" "*10} parameters #: {p.numel()}', flush=True)