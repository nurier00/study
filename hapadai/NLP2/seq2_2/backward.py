import os
import numpy as np
from torchtext import data, datasets
import torch
import torch.nn as nn
import seq2seq
import data_loader as data_loader

import torch
from torch import optim
import torch_optimizer as custom_optim

from utils import get_grad_norm, get_parameter_norm

def get_model(input_size, output_size):
    model = seq2seq.Seq2Seq(
        input_size,
        word_vec_size,           # Word embedding vector size
        hidden_size,             # LSTM's hidden vector size
        output_size,
        n_layers=n_layers,       # number of layers in LSTM
        dropout_p=dropout        # dropout-rate in LSTM
    )

    return model


def get_crit(output_size, pad_index):
    # Default weight for loss equals to 1, but we don't need to get loss for PAD token.
    # Thus, set a weight for PAD to zero.
    loss_weight = torch.ones(output_size)   # torch.ones(x) : 파라미터로 받은 변수 x의 크기만큼의 1로 채워진 tensor 생성
    loss_weight[pad_index] = 0.             # <PAD> 토큰이 채워진부분의 weight 값을 모두 0으로 변환
    # Instead of using Cross-Entropy loss,
    # we can use Negative Log-Likelihood(NLL) loss with log-probability.
    crit = nn.NLLLoss(
        weight=loss_weight,
        reduction='sum'                     # 'none' : 감소 적용 X | 'mean' : 출력의 가중평균 사용 | 'sum' : 출력 합산
    )

    return crit


def get_scheduler(optimizer, config):
    if config.lr_step > 0:
        lr_scheduler = optim.lr_scheduler.MultiStepLR(      # milestones 로 지정한 스텝마다 학습률에 감마를 곱해줌
            optimizer,
            milestones=[i for i in range(
                max(0, config.lr_decay_start - 1),
                (config.init_epoch - 1) + config.n_epochs,
                config.lr_step
            )],
            gamma=config.lr_gamma,
            last_epoch=config.init_epoch - 1 if config.init_epoch > 1 else -1,
        )
    else:
        lr_scheduler = None

    return lr_scheduler


def to_text(indice, vocab):
    # This method converts index to word to show the translation result.
    lines = []

    for i in range(len(indice)):
        line = []
        for j in range(len(indice[i])):
            index = indice[i][j]

            if index == data_loader.EOS:
                line += ['<EOS>']
                break
            else:
                # print(index)
                line += [vocab.itos[index]]
                # if index < 300:
                #     line += [vocab.itos[index]]

        line = ' '.join(line)
        lines += [line]
    return lines



def bsToTextTgt(tildes, type):
    print('###############')
    print(len(tildes))

    hats = []
    indices = []

    for it in range(len(tildes)):
        tilde = tildes[it]

        hat = generator(tilde)
        hats += [hat]
        indice = hat.argmax(dim=-1)
        indices += [indice]
        # print(indice)
    hats = torch.cat(hats, dim=1)
    indices = torch.cat(indices, dim=1)

    if type == 'src':
        output = to_text(indices, loader.src.vocab)
    else:
        output = to_text(indices, loader.tgt.vocab)
    listPrint(output)

def bsToTextSrc(tildes, type):
    print('###############')
    print(len(tildes))

    hats = []
    indices = []

    for it in range(len(tildes)):
        tilde = tildes[it]

        hat = generator(tilde)
        hats += [hat]
        indice = hat.argmax(dim=-1)
        indices += [indice]

    if type == 'src':
        output = to_text(indices, loader.src.vocab)
    else:
        output = to_text(indices, loader.tgt.vocab)
    listPrint(output)

def listPrint(tmp):
    print(len(tmp))
    for it in range(len(tmp)):
        print(tmp[it])


if __name__ == '__main__':

    batch_size = 100
    n_epochs = 100
    max_length = 100
    dropout = .2
    word_vec_size = 512
    hidden_size = 768
    n_layers = 1
    max_grad_norm = 1e+8
    iteration_per_update = 2
    lr = 1e-3
    lr_step = 0

    loader = data_loader.DataLoader(
        # train_fn='./mini10/corpus.news.shuf.train.10.tok.bpe',
        # valid_fn='./mini10/corpus.news.shuf.valid.10.tok.bpe',
        train_fn='./data/corpus.news.shuf.train.1000.tok.bpe',
        valid_fn='./data/corpus.news.shuf.valid.1000.tok.bpe',
        exts=('en', 'ko'),
        batch_size=batch_size,
        device=-1,                              # Lazy loading
        max_length=max_length  # 164
    )

    input_size = len(loader.src.vocab)
    output_size =  len(loader.tgt.vocab)
    print('input_size : ', input_size) # 304
    print('output_size : ', output_size) # 328

    generator = seq2seq.Generator(hidden_size, output_size)

    model = get_model(input_size, output_size)
    crit = get_crit(output_size, data_loader.PAD)
    # optimizer = get_optimizer(model)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):

        batch_index = None
        batch = None
        optimizer.zero_grad()

        for batch_index, batch in enumerate(loader.train_iter):
            pass
            # print('# : ', batch_index)
            # print('batch_index :', batch_index)
            # print('batch :', batch)

            batch.src = (batch.src[0], batch.src[1])
            batch.tgt = (batch.tgt[0], batch.tgt[1])

            x, y = batch.src, batch.tgt[0][:, 1:]


            y_hat, output_enc, test_out_dec, test_context_vector, h_tilde = model(x, batch.tgt[0][:,:-1])

            print('src, tgt ##################')
            x_enc = to_text(batch.src[0], loader.src.vocab)
            listPrint(x_enc)

            tgt_dec = to_text(batch.tgt[0], loader.tgt.vocab)
            listPrint(tgt_dec)

            print('enc ##################')
            bsToTextSrc(output_enc, 'src')

            print('dec ##################')
            print('test_out_dec')
            bsToTextTgt(test_out_dec, 'tgt')

            print('test_context_vector #####')
            bsToTextTgt(test_context_vector, 'src')

            print('result #######')
            bsToTextTgt(h_tilde, 'src')


            loss = crit(
                y_hat.contiguous().view(-1, y_hat.size(-1)),
                y.contiguous().view(-1)
            )
            iteration_per_update = 2
            backward_target = loss.div(y.size(0)).div(iteration_per_update)

            backward_target.backward()

            word_count = int(batch.tgt[1].sum())
            p_norm = float(get_parameter_norm(model.parameters()))
            g_norm = float(get_grad_norm(model.parameters()))


            loss = float(loss / word_count)
            ppl = np.exp(loss)

            print('loss: ', loss)
            print('ppl: ', ppl)

            optimizer.step()
