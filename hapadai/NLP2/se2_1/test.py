import torch
from torch import optim
import torch.nn as nn

from data_loader import DataLoader
import data_loader as data_loader

from seq2seq import Seq2Seq

from hapadai.NLP2.seq2.trainer import SingleTrainer
from hapadai.NLP2.seq2.trainer import MaximumLikelihoodEstimationEngine

batch_size = 10
n_epochs = 1
max_length = 64
dropout = .2
word_vec_size = 512
hidden_size = 768
n_layers = 1
max_grad_norm = 1e+8
iteration_per_update = 2
lr = 1e-3
lr_step = 0

def get_model(input_size, output_size):
    model = Seq2Seq(
        input_size,
        word_vec_size,
        hidden_size,
        output_size,
        n_layers=n_layers,
        dropout_p=dropout
    )
    return model

def get_crit(output_size, pad_index):
    loss_weight = torch.ones(output_size)
    loss_weight[pad_index] = 0.

    crit = nn.NLLLoss(
        weight=loss_weight,
        reduction='sum'
    )
    return crit

def get_optimizer(model):
    # adam
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return optimizer


def get_scheduler(optimizer):
    if lr_step > 0:
        pass
        # lr_scheduler = optim.lr_scheduler.MultiStepLR(
        #     optimizer,
        #     milestones=[i for i in range(
        #         max(0, lr_decay_start - 1),
        #         (init_epoch - 1) + n_epochs,
        #         lr_step
        #     )],
        #     gamma=lr_gamma,
        #     last_epoch=init_epoch - 1 if config.init_epoch > 1 else -1,
        # )
    else:
        lr_scheduler = None

    return lr_scheduler


if __name__ == '__main__':


    loader = DataLoader(
        train_fn='./mini10/corpus.news.shuf.train.10.tok.bpe',
        valid_fn='./mini10/corpus.news.shuf.valid.10.tok.bpe',
        exts=('en', 'ko'),
        batch_size=10,
        device=-1,                              # Lazy loading
        max_length=164  # 164
    )

    input_size = len(loader.src.vocab)
    output_size =  len(loader.tgt.vocab)
    print('input_size : ', input_size) # 304
    print('output_size : ', output_size) # 328

    batch_index = None
    batch = None
    for batch_index, batch in enumerate(loader.train_iter):
        pass

    model = get_model(input_size, output_size)
    crit = get_crit(output_size, data_loader.PAD)
    optimizer = get_optimizer(model)

    lr_scheduler = get_scheduler(optimizer)

    mle_trainer = SingleTrainer(MaximumLikelihoodEstimationEngine)
    mle_trainer.train(
        model,
        crit,
        optimizer,
        train_loader=loader.train_iter,
        valid_loader=loader.valid_iter,
        src_vocab=loader.src.vocab,
        tgt_vocab=loader.tgt.vocab,
        n_epochs=n_epochs,
        lr_scheduler=lr_scheduler
    )
