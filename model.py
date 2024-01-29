import torch
import torch.nn as nn
from torchcrf import CRF
from typing import List, Dict
import torch.nn.functional as F


class CharCNN(nn.Module):
    def __init__(self,
                 char_vocab_size: int,
                 char_embed_size: int,
                 kernel_size: int,
                 ) -> None:
        super(CharCNN, self).__init__()

        self.char_embedding = nn.Embedding(
            num_embeddings=char_vocab_size, 
            embedding_dim=char_embed_size, 
            padding_idx=0
        )
        self.char_conv = nn.Conv1d(
            in_channels=char_embed_size, 
            out_channels=char_embed_size, 
            kernel_size=kernel_size, 
            stride=1,
            padding=kernel_size // 2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :x: [batch_size, max_seq_len, max_word_size]
        :return: [batch_size, max_seq_len, char_embed_size]
        """
        batch_size = x.size(0)
        max_seq_len = x.size(1)
        max_word_len = x.size(2)

        x = self.char_embedding(x) # out: [batch_size, max_seq_len, max_word_len, char_embed_size]
        x = x.view(batch_size * max_seq_len, max_word_len, -1) # out: [batch_size * max_seq_len, max_word_len, char_embed_size]
        # Conv1d takes in [batch, dim, seq_len]
        x = x.transpose(2, 1) # out: [batch_size * max_seq_len, char_embed_size, max_word_len]
        
        output = self.char_conv(x) # out: [batch_size * max_seq_len, char_embed_size, max_word_len]
        output = torch.max(output, dim=-1, keepdim=True)[0] # out: [batch_size * max_seq_len, char_embed_size, 1]
        output = output.view(batch_size, max_seq_len, -1) # out: [batch_size, max_seq_len, char_embed_size]
        return output
    

class CNN_BiRNN_CRF(nn.Module):
    def __init__(self, 
                 word_embed_size: int, 
                 char_embed_size: int,
                 kernel_size: int,
                 lstm_hidden_size: int, 
                 dropout: float, 
                 num_layers: int,
                 word_voc_size: int, 
                 char_voc_size: int,
                 tag_voc_size: int,
                 ) -> None:
        super(CNN_BiRNN_CRF, self).__init__()

        self.word_embedding = nn.Embedding(
            num_embeddings=word_voc_size, 
            embedding_dim=word_embed_size,
            padding_idx=0
        )
        self.char_embedding = CharCNN(
            char_vocab_size=char_voc_size,
            char_embed_size=char_embed_size,
            kernel_size=kernel_size,
        )
        # self.lstm = nn.LSTM(
        #     input_size=word_embed_size+char_embed_size,
        #     hidden_size=lstm_hidden_size // 2, 
        #     num_layers=num_layers, 
        #     bidirectional=True, 
        #     batch_first=True,
        #     bias=False
        # )
        self.gru = nn.GRU(
            input_size=word_embed_size+char_embed_size,
            hidden_size=lstm_hidden_size // 2, 
            num_layers=num_layers, 
            bidirectional=True, 
            batch_first=True,
            bias=False
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(
            in_features=lstm_hidden_size, 
            out_features=tag_voc_size
        )
        self.crf = CRF(
            num_tags=tag_voc_size, 
            batch_first=True
        )
        self.init_weights()

    def forward(self, word_ids: torch.Tensor, char_ids: torch.Tensor) -> torch.Tensor:
        """
        Predicts emission scores.
        
        word_ids: [batch_size, max_seq_len]
        char_ids: [batch_size, max_seq_len, max_word_len]
        """
        word1_embeddings = self.word_embedding(word_ids) # out: [batch_size, max_seq_len, word_embed_size]
        word2_embeddings = self.char_embedding(char_ids) # out: [batch_size, max_seq_len, char_embed_size]
        embeddings = torch.cat([word1_embeddings, word2_embeddings], dim=-1) # out: [batch_size, max_seq_len, word_embed_size + char_embed_size]
        # outputs, hidden = self.lstm(embeddings)
        rnn_outputs, hidden = self.gru(embeddings)
        rnn_outputs = self.dropout(rnn_outputs)
        # outputs = self.fc(outputs)
        emission_scores = self.fc(rnn_outputs + word2_embeddings) # skip-connection: they have the same dim
        return emission_scores
    
    def decode(self, emission_scores: torch.Tensor) -> List[float]:
        """
        Returns sequence of labels.
        """
        return self.crf.decode(emission_scores)
        
    def loss_fn(self, emission_scores: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Calculates negative log-likelihood loss (NLL).
        """
        return -self.crf(emission_scores, tags, mask=mask, reduction='mean')
    
    def regularization_loss_fn(self, lam: float) -> float:
        """
        Calculates regularization loss function.
        """
        learnable_weights = [p for p in self.parameters() if p.requires_grad and p.dim() > 1]
        loss = lam * torch.sum(torch.tensor([torch.sum(w**2) for w in learnable_weights]))
        return loss
        
    def init_weights(self) -> None:
        """
        Xavier/Glorot initialization for weights and 
        zero initialization for biases.
        """
        learnable_named_parameters = [(name, p) for name, p in self.named_parameters() if p.requires_grad]
        
        for name, p in learnable_named_parameters:
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                print(f'{name:<40} initialized w with Xavier {" "*10} parameters #: {p.numel()}', flush=True)
            else:
                nn.init.zeros_(p)
                print(f'{name:<40} initialized b with zero   {" "*10} parameters #: {p.numel()}', flush=True)
