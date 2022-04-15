import os
import argparse
import spacy
import urllib.request as request

import torch
import torch.nn as nn
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from datasets_tn import MyDatasets
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from setup import train_model
from utils import*
from model import*

def Train(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    SRC_LANGUAE = 'en'
    TGT_LANGUAE = 'vi'

    token_transform = {}
    vocab_transform = {}

    # create data and Download directory
    data = './data'
    download = './Download'
    create_dir(data)
    create_dir(download)

    # dowload data to Download directory
    urls = {}
    target = {}
    urls['train'] = 'https://github.com/stefan-it/nmt-en-vi/raw/master/data/train-en-vi.tgz'
    urls['vaild'] = 'https://github.com/stefan-it/nmt-en-vi/raw/master/data/test-2013-en-vi.tgz'
    target['train'] = os.path.join(download, 'train-en-vi.tgz')
    target['vaild'] = os.path.join(download, 'test-2013-en-vi.tgz')
    for url in urls:
        if not os.path.exists(target[url]):
            request.urlretrieve(urls[url], target[url])


    # extract data tu data directory
    for id in target:
        extract_tgz(target[id], os.path.join(data, id))

    vi_train = []
    en_train = []
    vi_val = []
    en_val = []

    create_raw_data('./data/train/train.vi', vi_train)
    create_raw_data('./data/train/train.en', en_train)
    create_raw_data('./data/vaild/tst2013.vi', vi_val)
    create_raw_data('./data/vaild/tst2013.en', en_val)

    x = [i for i, e in enumerate(vi_train) if e == '.\n' or e == '\n']
    y = [i for i, e in enumerate(en_train) if e == '.\n' or e == '\n']

    if len(y) >= len(x):
        vi_train = [vi_train[idx] for idx, el in enumerate(vi_train) if idx not in y]
        en_train = [en_train[idx] for idx, el in enumerate(en_train) if idx not in y]
    else:
        i_train = [vi_train[idx] for idx, el in enumerate(vi_train) if idx not in x]
        en_train = [en_train[idx] for idx, el in enumerate(en_train) if idx not in x]

    vi_train = vi_train[1: len(vi_train)-3]
    en_train = en_train[1: len(en_train)-3]
    data_dict_train = {SRC_LANGUAE: en_train, TGT_LANGUAE: vi_train}
    data_dict_val = {SRC_LANGUAE: en_val, TGT_LANGUAE: vi_val}

    # **************************** build vocab ******************************

    token_transform[SRC_LANGUAE] = get_tokenizer('spacy', language='en_core_web_sm')
    token_transform[TGT_LANGUAE] = get_tokenizer('spacy', language='vi_core_news_lg')

    def yield_tokens(data_dict, language):
        for data in data_dict[language]:
            yield token_transform[language](data)

    UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX  = 0, 1, 2, 3
    special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

    for ln in [SRC_LANGUAE, TGT_LANGUAE]:
        vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(data_dict_train, ln),
                                                        min_freq=1,
                                                        specials=special_symbols,
                                                        special_first=True)

    vocab_transform[SRC_LANGUAE].set_default_index(UNK_IDX)
    vocab_transform[TGT_LANGUAE].set_default_index(UNK_IDX)

    # ********************* init model object *****************

    torch.manual_seed(1)

    SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAE])
    TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAE])

    transformer = MyTransformer(args.num_encoder_layers, args.num_decoder_layers, args.emb_size,
                                args.nhead, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, args.ffn_hid_dim)

    # init parameter with xavier init
    for param in transformer.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)

    # ******************************* create mask ****************************************

    def generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def create_mask(src, tgt):
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

        tgt_mask = generate_square_subsequent_mask(tgt_seq_len)  # T_src x T_src with lower left and diagonal is 0.0, upper right is -inf
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)  # T_tgt x T_tgt full of 0.0

        src_padding_mask = (src == PAD_IDX).transpose(0, 1)
        tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    # **************************** transforms data **************************

    def sequential_transforms(*transforms):
        def func(txt_input):
            for transform in transforms:
                txt_input = transform(txt_input)
            return txt_input

        return func

    def bos_eos_add(token_ids):
        return torch.cat((torch.tensor([BOS_IDX]),
                          torch.tensor(token_ids),
                          torch.tensor([EOS_IDX])))

    text_transform = {}
    for ln in [SRC_LANGUAE, TGT_LANGUAE]:
        text_transform[ln] = sequential_transforms(token_transform[ln],
                                                   vocab_transform[ln],
                                                   bos_eos_add)

    def collate_fn(batch):
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(text_transform[SRC_LANGUAE](src_sample.rstrip("\n")))
            tgt_batch.append(text_transform[TGT_LANGUAE](tgt_sample.rstrip("\n")))

        src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)  # TxB, T is the longest length of sequences, B is b_size
        tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)  # TxB
        return src_batch, tgt_batch


    Dataset_train = MyDatasets(data_dict_train[SRC_LANGUAE], data_dict_train[TGT_LANGUAE])
    Dataset_vaild = MyDatasets(data_dict_val[SRC_LANGUAE], data_dict_val[TGT_LANGUAE])

    trainLoader = DataLoader(Dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers, collate_fn=collate_fn)
    vaildLoader = DataLoader(Dataset_vaild, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers, collate_fn=collate_fn)


    dataLoader = {'train': trainLoader, 'val': vaildLoader}
    data_size = {'train': len(trainLoader), 'val': len(vaildLoader)}

    # ************************************* traslation *****************************************
    def greedy_decode(model, src, src_mask, max_len, start_symbol, device):
        src = src.to(device)
        src_mask = src_mask.to(device)

        memory = model.encode(src, src_mask)
        ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
        for i in range(max_len - 1):
            memory = memory.to(device)
            tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                        .type(torch.bool)).to(device)
            out = model.decode(ys, memory, tgt_mask)
            out = out.transpose(0, 1)
            prob = model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()

            ys = torch.cat([ys,
                            torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
            if next_word == EOS_IDX:
                break
        return ys

    def translation(model, src_sentence):
        model.eval()
        src = text_transform[SRC_LANGUAE](src_sentence).view(-1, 1)
        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        tgt_tokens = greedy_decode(model, src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX, device=device).flatten()
        return " ".join(vocab_transform[TGT_LANGUAE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")

    # ********************************* process ***************************************
    if args.is_train:
        if args.use_weights:
            print('Using weight pretrain to init model')
            print('--' * 20)
            transformer_weight = torch.load(args.weights)
            transformer.load_state_dict(transformer_weight)
        model = train_model(transformer, dataLoader, criterion, optimizer, args.n_epochs, data_size, device=device,  create_mask=create_mask)
    else:
        if not os.path.exists(args.weights):
            print('Run with random weight, result:')
            print(translation(transformer, args.sentence_src))
        else:
            transformer_weight = torch.load(args.weights, map_location=device)
            transformer.load_state_dict(transformer_weight)
            print(translation(transformer, args.sentence_src))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train money model')
    parser.add_argument('--use-weights', action='store_true', help='check use weights')
    parser.add_argument('--is-train', action='store_true', help='activate train mode')
    parser.add_argument('--weights', type=str, default='./weight/translation.pth')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--n-epochs', type=int, default=5)
    parser.add_argument('--n-workers', type=int, default=8)
    parser.add_argument('--emb-size', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--ffn-hid-dim', type=int, default=512)
    parser.add_argument('--num-encoder-layers', type=int, default=3)
    parser.add_argument('--num-decoder-layers', type=int, default=3)
    parser.add_argument('--sentence-src', type=str, default='hello')
    args = parser.parse_args()
    Train(args)

















