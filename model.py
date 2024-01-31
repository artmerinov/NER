import torch
import torch.nn as nn
from torchcrf import CRF
from typing import List, Dict
import torch.nn.functional as F


class CharRNN(nn.Module):
    def __init__(self,
                 char_vocab_size: int,
                 char_embed_size: int,
                 ) -> None:
        super(CharRNN, self).__init__()

        self.char_embedding = nn.Embedding(
            num_embeddings=char_vocab_size, 
            embedding_dim=char_embed_size, 
            padding_idx=0 # padding doesn't contribute to the gradient
        )
        self.char_rnn = nn.GRU(
                input_size=char_embed_size,
                hidden_size=char_embed_size // 2, # same
                num_layers=1, 
                bidirectional=True, 
                batch_first=True,
                bias=True
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        batch_size = x.size(0)
        max_seq_len = x.size(1)
        max_word_len = x.size(2)

        x = self.char_embedding(x) 
        # [batch_size, max_seq_len, max_word_len, char_embed_size]
        # print(x.size()) # [128, 64, 16, 128]
        
        x = x.view(batch_size * max_seq_len, max_word_len, -1) 
        # [batch_size * max_seq_len, max_word_len, char_embed_size]
        # print(x.size()) # [128*64, 16, 128]
        
        rnn_outputs, hidden = self.char_rnn(x) 
        # [batch_size * max_seq_len, max_word_len, char_embed_size]
        # print(rnn_outputs.size()) # [128*64, 16, 128]
        
        rnn_outputs = rnn_outputs.transpose(2, 1) 
        # [batch_size * max_seq_len, char_embed_size, max_word_len]
        # print(rnn_outputs.size()) # [128*64, 128, 16]
        
        output = torch.max(rnn_outputs, dim=-1, keepdim=True)[0] 
        # max pooling among max_word_len dimention
        # (collapse char dimentions to represent a word)
        # [batch_size * max_seq_len, char_embed_size, 1]
        # print(output.size()) # [128*64, 128, 1]
        
        output = output.view(batch_size, max_seq_len, -1) 
        # [batch_size, max_seq_len, char_embed_size]
        # print(output.size()) # [128, 64, 128]
        
        return output


class CharCNN(nn.Module):
    def __init__(self,
                 char_vocab_size: int,
                 char_embed_size: int,
                 char_kernel_size: int,
                 ) -> None:
        super(CharCNN, self).__init__()

        self.char_embedding = nn.Embedding(
            num_embeddings=char_vocab_size, 
            embedding_dim=char_embed_size, 
            padding_idx=0 # padding doesn't contribute to the gradient
        )
        self.char_conv = nn.Conv1d(
            in_channels=char_embed_size, 
            out_channels=char_embed_size, # same
            kernel_size=char_kernel_size, 
            stride=1,
            padding=char_kernel_size // 2,
            # bias=False,
            bias=True,
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :x: [batch_size, max_seq_len, max_word_size]
        :return: [batch_size, max_seq_len, char_embed_size]
        """
        batch_size = x.size(0)
        max_seq_len = x.size(1)
        max_word_len = x.size(2)

        x = self.char_embedding(x) 
        # out: [batch_size, max_seq_len, max_word_len, char_embed_size]
        # print(x.size()) # [128, 64, 16, 128]
        
        x = x.view(batch_size * max_seq_len, max_word_len, -1) 
        # out: [batch_size * max_seq_len, max_word_len, char_embed_size]
        # print(x.size()) # [128*64, 16, 128]
        
        x = x.transpose(2, 1) 
        # Conv1d takes in [batch, dim, seq_len]
        # out: [batch_size * max_seq_len, char_embed_size, max_word_len]
        # print(x.size()) # [128*64, 128, 16]
        
        output = self.char_conv(x) 
        # out: [batch_size * max_seq_len, char_embed_size, max_word_len]
        # print(output.size()) # [128*64, 128, 16]
        
        output = torch.max(output, dim=-1, keepdim=True)[0] 
        # max pooling among max_word_len dimention
        # (collapse char dimentions to represent a word)
        # out: [batch_size * max_seq_len, char_embed_size, 1]
        # print(output.size()) # [128*64, 128, 1]
        
        output = output.view(batch_size, max_seq_len, -1) 
        # out: [batch_size, max_seq_len, char_embed_size]
        # print(output.size()) # [128, 64, 128]

        return output
    

class CNN_BiRNN_CRF(nn.Module):
    def __init__(self, 
                 word_voc_size: int, 
                 char_voc_size: int,
                 tag_voc_size: int,
                 word_embed_size: int, 
                 char_embed_size: int,
                 char_kernel_size: int,
                 rnn_cell: str,
                 rnn_hidden_size: int, 
                 dropout: float, 
                 num_layers: int,
                 skip_connection: bool,
                 ) -> None:
        super(CNN_BiRNN_CRF, self).__init__()
        self.skip_connection = skip_connection

        self.word_embedding = nn.Embedding(
            num_embeddings=word_voc_size, 
            embedding_dim=word_embed_size,
            padding_idx=0 # padding doesn't contribute to the gradient
        )
        self.char_embedding = CharCNN(
            char_vocab_size=char_voc_size,
            char_embed_size=char_embed_size,
            char_kernel_size=char_kernel_size,
        )
        # self.char_embedding = CharRNN(
        #     char_vocab_size=char_voc_size,
        #     char_embed_size=char_embed_size,
        # )
        # self.batch_norm = nn.BatchNorm1d(
        #     num_features=word_embed_size + char_embed_size
        # )
        if rnn_cell == "LSTM":
            self.rnn = nn.LSTM(
                input_size=word_embed_size+char_embed_size,
                hidden_size=rnn_hidden_size // 2, 
                num_layers=num_layers, 
                bidirectional=True, 
                batch_first=True,
                # bias=False,
                bias=True,
            )
        elif rnn_cell == "GRU":
            self.rnn = nn.GRU(
                input_size=word_embed_size+char_embed_size,
                hidden_size=rnn_hidden_size // 2, 
                num_layers=num_layers, 
                bidirectional=True, 
                batch_first=True,
                # bias=False,
                bias=True,
            )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(
            in_features=rnn_hidden_size, 
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
        word1_embeddings = self.word_embedding(word_ids)
        # out: [batch_size, max_seq_len, word_embed_size]
        word2_embeddings = self.char_embedding(char_ids)
        # out: [batch_size, max_seq_len, char_embed_size]
        embeddings = torch.cat([word1_embeddings, word2_embeddings], dim=-1) 
        # out: [batch_size, max_seq_len, word_embed_size + char_embed_size]
        # embeddings = self.batch_norm(embeddings.permute(0, 2, 1)).permute(0, 2, 1)

        rnn_outputs, hidden = self.rnn(embeddings)
        rnn_outputs = self.dropout(rnn_outputs)

        if self.skip_connection:
            assert rnn_outputs.size() == word2_embeddings.size(), "Input sizes must match for skip connection."
            emission_scores = self.fc(rnn_outputs + word2_embeddings) # skip-connection: they have the same dim
        else:
            emission_scores = self.fc(rnn_outputs)
            
        return emission_scores
    
    def decode(self, emission_scores: torch.Tensor) -> List[float]:
        """
        Returns the most likely tag sequence using Viterbi algorithm.
        """
        return self.crf.decode(emission_scores)
        
    def loss_fn(self, emission_scores: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Calculates negative log-likelihood loss (NLL).
        
        The forward computation of CRF class computes the log likelihood of the 
        given sequence of tags and emission score tensor. Therefore, we need to 
        make this value negative as it is a loss. 
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
                print(f'{name:<40} init W with Xavier {" "*10} # params: {p.numel()}', flush=True)
            else:
                nn.init.zeros_(p)
                print(f'{name:<40} init b with zero   {" "*10} # params: {p.numel()}', flush=True)

        # Set forget gate bias high to improve gradients flow
        # https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745/3
        for names in self.rnn._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = getattr(self.rnn, name)
                n = bias.size(0)
                bias.data[n//4:n//2].fill_(1.0)
