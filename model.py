import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_module, dropout, maxlen=5000):  # d_modele = 512 in paper
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, d_module, 2) * math.log(1000) / d_module) # 512/2
        pos = torch.arange(0, maxlen).reshape(maxlen, 1) # 5000x1
        pos_embedding = torch.zeros((maxlen, d_module)) # 5000x512
        pos_embedding[:, 0::2] = torch.sin(pos * den)  # 5000x(512/2)
        pos_embedding[:, 1::2] = torch.cos(pos * den)  # 5000x(512/2)
        pos_embedding = pos_embedding.unsqueeze(-2) # 5000x1x512

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        # TxBxEz: T is length of sequence, B is batch_size, Ez is Embedding size
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size):  # emb_size = 512 in paper
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class MyTransformer(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers, emb_size, nhead,
                 src_vocab_size, tgt_vocab_size, dim_feedforward=512, dropout=0.1):
        super(MyTransformer, self).__init__()
        self.src_token = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_token = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
        # transformer layer which includes n_encoder layer and n_decoder layer
        self.transformer = nn.Transformer(d_model=emb_size,
                                          nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)  # last layer show probability of token_tgt in vocab_tgt

    def forward(self, src, tgt, src_mask, tgt_mask, src_padding_mask,
               tgt_padding_mask, memory_key_padding_mask):
        """
        Args:
            src: Batch of Tensor - batch input sequences were tokenized, vocab, add <bos> <eos> and pad_sequence
            tgt: Batch of Tensor - batch target sequences were tokenized, vocab, add <bos> <eos> and pad_sequence
            src_mask: Tensor T_src x T_src with full 1, T_src is the longest length of sequence in src batch
            tgt_mask: Tensor T_tgt x T_tgt
            src_padding_mask
            tgt_padding_mask
            memory_key_padding_mask
        """
        src_emb = self.positional_encoding(self.src_token(src))
        tgt_emb = self.positional_encoding(self.tgt_token(tgt))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None, src_padding_mask,
                                tgt_padding_mask, memory_key_padding_mask)  # TxBxE T is the longest length of sequence,
                                                                            # B is batch_size, E is Embedding size.
        return self.generator(outs)

    def encode(self, src, src_mask):
        return self.transformer.encoder(self.positional_encoding(self.src_token(src)), src_mask)

    def decode(self, tgt, memory, tgt_mask):
        return self.transformer.decoder(self.positional_encoding(self.tgt_token(tgt)), memory, tgt_mask)


