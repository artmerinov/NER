import os
import json
import argparse
from tqdm import tqdm
import numpy as np
from datetime import datetime

import torch
from torch.utils.data import DataLoader
import torch.utils.tensorboard as tensorboard
from seqeval.metrics import classification_report

from tokenizer import make_word_vocab, make_char_vocab
from prepare_data import preprocess_raw_data
from dataset import NERDataset
from model import CNN_BiRNN_CRF
from utils import set_random_seed, Config, load_json, io2bio


def train():

    # load default parameters from yaml file
    config = Config('config.yaml')

    parser = argparse.ArgumentParser()
    parser.add_argument('--tr_path', type=str, default=config.TR_PATH, help='Path to the train data')
    parser.add_argument('--va_path', type=str, default=config.VA_PATH, help='Path to the validation data')
    parser.add_argument('--te_path', type=str, default=config.TE_PATH, help='Path to the test data')

    parser.add_argument('--max_seq_len', type=int, default=config.MAX_SEQ_LEN, help='Max number of words in the sentence')
    parser.add_argument('--max_word_len', type=int, default=config.MAX_WORD_LEN, help='Max number of characters in the word')
    parser.add_argument('--word_support', type=int, default=config.WORD_SUPPORT, help='Min frequency of word to include in vocabulary')
    parser.add_argument('--char_support', type=int, default=config.CHAR_SUPPORT, help='Min frequency of char to include in vocabulary')
    
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE, help='Number of sentences in the batch')
    parser.add_argument('--num_epochs', type=int, default=config.NUM_EPOCHS, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=config.LR, help='Learning rate')
    parser.add_argument('--reg_lambda', type=float, default=config.REG_LAMBDA, help='Regularization coefficient for the weights')
    parser.add_argument('--max_grad_norm', type=int, default=config.MAX_GRAD_NORM, help='Max norm of gradients in order to clip them')
    
    parser.add_argument('--word_embed_size', type=int, default=config.WORD_EMBED_SIZE, help='Embedding size of the word')
    parser.add_argument('--char_embed_size', type=int, default=config.CHAR_EMBED_SIZE, help='Embedding size of the char')
    parser.add_argument('--char_kernel_size', type=int, default=config.CHAR_KERNEL_SIZE, help='Kernel size of CNN for char representation')
    parser.add_argument('--rnn_cell', type=str, default=config.RNN_CELL, help='RNN cell: LSTM or GRU')
    parser.add_argument('--rnn_hidden_size', type=int, default=config.RNN_HIDDEN_SIZE, help='Total embedding size of BiRNN output')
    parser.add_argument('--dropout', type=int, default=config.DROPOUT, help='Dropout probability')
    parser.add_argument('--num_layers', type=int, default=config.NUM_LAYERS, help='Number of reccurent layers')
    parser.add_argument('--skip_connection', type=bool, default=False, help="Whether use or not skip connection trick")
    
    # rewrite config
    config = parser.parse_args()
    print(config)

    TAG2IDX = load_json('ner_tags/ner_fine_grained.json')

    # load/create vocabulary
    WORD2IDX = make_word_vocab(filepath=config.tr_path, support=config.word_support, save_path='tokenizers/word2idx.json')
    CHAR2IDX = make_char_vocab(filepath=config.tr_path, support=config.char_support, save_path='tokenizers/char2idx.json')

    IDX2TAG = {i: t for t, i in TAG2IDX.items()}
    IDX2WORD = {i: w for w, i in WORD2IDX.items()}

    tr_titles, va_titles = preprocess_raw_data(config=config, tag2idx=TAG2IDX, word2idx=WORD2IDX)

    tr_dataset = NERDataset(data=tr_titles, idx2word=IDX2WORD, char2idx=CHAR2IDX, max_word_len=config.max_word_len)
    va_dataset = NERDataset(data=va_titles, idx2word=IDX2WORD, char2idx=CHAR2IDX, max_word_len=config.max_word_len)

    tr_dataloader = DataLoader(dataset=tr_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    va_dataloader = DataLoader(dataset=va_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = CNN_BiRNN_CRF(
        word_embed_size  = config.word_embed_size,
        char_embed_size  = config.char_embed_size,
        char_kernel_size = config.char_kernel_size,
        rnn_cell         = config.rnn_cell,
        rnn_hidden_size  = config.rnn_hidden_size,
        dropout          = config.dropout,
        num_layers       = config.num_layers,
        skip_connection  = config.skip_connection,
        word_voc_size    = len(WORD2IDX),
        char_voc_size    = len(CHAR2IDX),
        tag_voc_size     = len(TAG2IDX),
    ).to(device)
    print(model)

    weights_folder = 'weights'
    if not os.path.exists(weights_folder):
        os.makedirs(weights_folder)
        
    runs_folder = 'runs'
    if not os.path.exists(runs_folder):
        os.makedirs(runs_folder)

    experiment_folder = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not os.path.exists(f'{runs_folder}/{experiment_folder}'):
        os.makedirs(f'{runs_folder}/{experiment_folder}')

    # Make optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Make tensorboard writer
    writer = tensorboard.SummaryWriter(log_dir=f'{runs_folder}/{experiment_folder}')
    writer.add_text('Run parameters', " ".join([f"{k}={v}" for k,v in config.__dict__.items()]))

    for epoch in range(config.num_epochs):

        # TRAINING PHASE
        
        tr_losses = []
        
        model.train()
        for batch_id, tr_batch in enumerate(tr_dataloader):
            optimizer.zero_grad()
            
            tr_xs = tr_batch['word_ids'].to(device) # [batch_size, max_seq_len]
            tr_cs = tr_batch['char_ids'].to(device) # [batch_size, max_seq_len, max_word_len]
            tr_ys = tr_batch['tag_ids'].to(device) # [batch_size, max_seq_len]
            tr_mask = (tr_ys > 0).bool()
            
            tr_emission_scores = model(word_ids=tr_xs, char_ids=tr_cs).to(device) # [batch_size, max_seq_len, max_word_len]
            tr_loss = model.loss_fn(emission_scores=tr_emission_scores, tags=tr_ys, mask=tr_mask)
            reg_loss = model.regularization_loss_fn(lam=config.reg_lambda)
            total_loss = tr_loss + reg_loss
            tr_losses.append(tr_loss.item())
            
            if batch_id % 100 == 0:
                print(
                    f"epoch={epoch:02d}",
                    f"batch_id={batch_id:04d}/{len(tr_dataloader)}",
                    f"crf_loss={tr_loss:.2f}",
                    f"reg_loss={reg_loss:.2f}",
                    f"total_loss={total_loss:.2f}"
                )

            # Backward pass: compute gradient of the loss w.r.t. all learnable parameters
            total_loss.backward()
            
            # Clip computed gradients
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.max_grad_norm)
            
            # Optimize: update the weights using Adam optimizer
            optimizer.step()

        # VALIDATION PHASE
        
        va_losses = []
        
        batch_preds = []
        batch_trues = []

        model.eval()
        with torch.no_grad():
            for va_batch in tqdm(va_dataloader, total=len(va_dataloader)):
                
                va_xs = va_batch['word_ids'].to(device) # [batch_size, max_seq_len]
                va_cs = va_batch['char_ids'].to(device) # [batch_size, max_seq_len, max_word_len]
                va_ys = va_batch['tag_ids'].to(device) # [batch_size, max_seq_len]
                va_mask = (va_ys > 0).bool()

                # Forward pass: compute predicted output by passing input to the model
                va_emission_scores = model(word_ids=va_xs, char_ids=va_cs).to(device) # [batch_size, max_seq_len]
                va_loss = model.loss_fn(emission_scores=va_emission_scores, tags=va_ys, mask=va_mask)
                va_losses.append(va_loss.item())

                for row_id, true in enumerate(va_ys):
                    # do not count padding
                    true_tags = true[va_mask[row_id]]
                    # idx2tag
                    true_tags = [IDX2TAG[idx] for idx in true_tags.tolist()]
                    # convert to the format expected by seqeval
                    true_tags = io2bio(true_tags)
                    batch_trues.append(true_tags)

                va_preds = torch.tensor(model.decode(va_emission_scores)).to(device)
                for row_id, pred in enumerate(va_preds):
                    # do not count padding
                    pred_tags = pred[va_mask[row_id]]
                    # idx2tag
                    pred_tags = [IDX2TAG[idx] for idx in pred_tags.tolist()]
                    # convert to the format expected by seqeval
                    pred_tags = io2bio(pred_tags)
                    batch_preds.append(pred_tags)
            
    #         for i in range(5):
    #             print('pred:', batch_preds[i])
    #             print('true:', batch_trues[i])
    #             print()

        writer.add_scalar('tr/'+'loss', np.mean(tr_losses), global_step=epoch)
        writer.add_scalar('va/'+'loss', np.mean(va_losses), global_step=epoch)
        for name, param in model.named_parameters():
            writer.add_histogram('tr/' + name + '_weight', param.data, global_step=epoch)
            writer.add_histogram('tr/' + name + '_grad', param.grad, global_step=epoch)

        print(
            f"tr_avg_loss={np.mean(tr_losses)}", 
            f"va_avg_loss={np.mean(va_losses)}"
        )
        
        report = classification_report(y_true=batch_trues, y_pred=batch_preds, zero_division=0)
        print(report)

        torch.save(model.state_dict(), f"weights/model_epoch_{epoch:02d}.pt")

    writer.close()
        

if __name__ == "__main__":
    set_random_seed(seed=0)
    train()

# python3 train.py