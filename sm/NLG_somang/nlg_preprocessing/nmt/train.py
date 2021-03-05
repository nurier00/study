import argparse
import pprint

import torch
from torch import optim
import torch.nn as nn

import torch_optimizer as custom_optim

from simple_nmt.data_loader import DataLoader
import simple_nmt.data_loader as data_loader

from simple_nmt.models.seq2seq import Seq2Seq
from simple_nmt.models.transformer import Transformer
from simple_nmt.models.rnnlm import LanguageModel

from simple_nmt.trainer import SingleTrainer
from simple_nmt.rl_trainer import MinimumRiskTrainingEngine
from simple_nmt.trainer import MaximumLikelihoodEstimationEngine


def define_argparser(is_continue=False):
    p = argparse.ArgumentParser()

    if is_continue:
        p.add_argument(
            '--load_fn',            # continue_train.py 로 중단된 학습을 재시작할 때 설정할 옵션, 이전 모델 파일
            required=True,
            help='Model file name to continue.'
        )

    p.add_argument(
        '--model_fn',               # 학습 시 생성될 모델 파일명
        required=not is_continue,
        help='Model file name to save. Additional information would be annotated to the file name.'
    )
    p.add_argument(
        '--train',                  # train 데이터 ( bpe 적용 데이터를 넣되 확장자(en / ko) 표시 X )
        required=not is_continue,
        help='Training set file name except the extention. (ex: train.en --> train)'
    )
    p.add_argument(
        '--valid',                  # valid 데이터 ( bpe 적용 데이터를 넣되 확장자(en / ko) 표시 X )
        required=not is_continue,
        help='Validation set file name except the extention. (ex: valid.en --> valid)'
    )
    p.add_argument(
        '--lang',                   # 번역 설정 ( 영어 > 한국어 : enko / 한국어 > 영어 : koen )
        required=not is_continue,
        help='Set of extention represents language pair. (ex: en + ko --> enko)'
    )
    p.add_argument(
        '--gpu_id',                 # gpu 사용시
        type=int,
        default=-1,
        help='GPU ID to train. Currently, GPU parallel is not supported. -1 for CPU. Default=%(default)s'
    )
    p.add_argument(
        '--off_autocast',           # AMP(FP16 + FP32) 미사용
        action='store_true',
        help='Turn-off Automatic Mixed Precision (AMP), which speed-up training.',
    )

    p.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Mini batch size for gradient descent. Default=%(default)s'
    )
    p.add_argument(
        '--n_epochs',
        type=int,
        default=20,
        help='Number of epochs to train. Default=%(default)s'
    )
    p.add_argument(
        '--verbose',                # 출력 레벨 설정
        type=int,
        default=2,
        help='VERBOSE_SILENT, VERBOSE_EPOCH_WISE, VERBOSE_BATCH_WISE = 0, 1, 2. Default=%(default)s'
    )
    p.add_argument(
        '--init_epoch',             # continue_train.py 를 사용하여 재시작 시 설정할 옵션, 시작 epoch 설정
        required=is_continue,
        type=int,
        default=1,
        help='Set initial epoch number, which can be useful in continue training. Default=%(default)s'
    )

    p.add_argument(
        '--max_length',             # 최대 sequence 길이
        type=int,
        default=100,
        help='Maximum length of the training sequence. Default=%(default)s'
    )
    p.add_argument(
        '--dropout',                # dropout 설정률
        type=float,
        default=.2,
        help='Dropout rate. Default=%(default)s'
    )
    p.add_argument(
        '--word_vec_size',
        type=int,
        default=512,
        help='Word embedding vector dimension. Default=%(default)s'
    )
    p.add_argument(
        '--hidden_size',
        type=int,
        default=768,
        help='Hidden size of LSTM. Default=%(default)s'
    )
    p.add_argument(
        '--n_layers',               # 레이어 수
        type=int,
        default=4,
        help='Number of layers in LSTM. Default=%(default)s'
    )
    p.add_argument(
        '--max_grad_norm',          # (SGD optimizer 사용시) 기울기가 발산되는 것을 막기 위해 임계값을 설정하여 학습 안정화 도모
        type=float,
        default=5.,
        help='Threshold for gradient clipping. Default=%(default)s'
    )
    p.add_argument(
        '--iteration_per_update',   # Gradient Accumulation
        type=int,
        default=1,
        help='Number of feed-forward iterations for one parameter update. Default=%(default)s'
    )

    p.add_argument(
        '--lr',                     # learning rate
        type=float,
        default=1.,
        help='Initial learning rate. Default=%(default)s',
    )

    p.add_argument(
        '--lr_step',                # learning rate decay 사용 시 epochs 를 돌 때, 증가될 스텝(걸음폭)
        type=int,
        default=1,
        help='Number of epochs for each learning rate decay. Default=%(default)s',
    )
    p.add_argument(
        '--lr_gamma',               # learning rate 에 곱해줄 gamma 값 (감소값)
        type=float,
        default=.5,
        help='Learning rate decay rate. Default=%(default)s',
    )
    p.add_argument(
        '--lr_decay_start',         # 시작 학습률
        type=int,
        default=10,
        help='Learning rate decay start at. Default=%(default)s',
    )

    p.add_argument(
        '--use_adam',               # adam optimizer 사용 (SGD 사용시 옵션 미입력)
        action='store_true',
        help='Use Adam as optimizer instead of SGD. Other lr arguments should be changed.',
    )
    p.add_argument(
        '--use_radam',              # Radam optimizer 사용
        action='store_true',
        help='Use rectified Adam as optimizer. Other lr arguments should be changed.',
    )

    p.add_argument(
        '--rl_lr',                  # 강화학습 시 leraning rate
        type=float,
        default=.01,
        help='Learning rate for reinforcement learning. Default=%(default)s'
    )
    p.add_argument(
        '--rl_n_samples',           # 강화학습 샘플 수
        type=int,
        default=1,
        help='Number of samples to get baseline. Default=%(default)s'
    )
    p.add_argument(
        '--rl_n_epochs',            # 강화학습 epochs
        type=int,
        default=0,
        help='Number of epochs for reinforcement learning. Default=%(default)s'
    )
    p.add_argument(
        '--rl_n_gram',              # BLEU 평가 시 사용될 n_gram 수 지정
        type=int,
        default=6,
        help='Maximum number of tokens to calculate BLEU for reinforcement learning. Default=%(default)s'
    )
    p.add_argument(
        '--rl_reward',              # 강화학습 reward function 으로 사용할 함수명
        type=str,
        default='gleu',
        help='Method name to use as reward function for RL training. Default=%(default)s'
    )

    p.add_argument(
        '--use_transformer',        # transformer 사용
        action='store_true',
        help='Set model architecture as Transformer.',
    )
    p.add_argument(
        '--n_splits',               # transformer 사용시 multi-head attention 의 head 수
        type=int,
        default=8,
        help='Number of heads in multi-head attention in Transformer. Default=%(default)s',
    )

    config = p.parse_args()

    return config


def get_model(input_size, output_size, config):
    if config.use_transformer:
        model = Transformer(
            input_size,                     # Source vocabulary size
            config.hidden_size,             # Transformer doesn't need word_vec_size.
            output_size,                    # Target vocabulary size
            n_splits=config.n_splits,       # Number of head in Multi-head Attention.
            n_enc_blocks=config.n_layers,   # Number of encoder blocks
            n_dec_blocks=config.n_layers,   # Number of decoder blocks
            dropout_p=config.dropout,       # Dropout rate on each block
        )
    else:
        model = Seq2Seq(
            input_size,
            config.word_vec_size,           # Word embedding vector size
            config.hidden_size,             # LSTM's hidden vector size
            output_size,
            n_layers=config.n_layers,       # number of layers in LSTM
            dropout_p=config.dropout        # dropout-rate in LSTM
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


def get_optimizer(model, config):
    if config.use_adam:
        if config.use_transformer:
            optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(.9, .98))
            # betas : 그래디언트와 그 제곱의 지수평균을 계산하는데 사용 (기본값 : 0.9, 0.999)

        else:   # case of rnn based seq2seq.
            optimizer = optim.Adam(model.parameters(), lr=config.lr)
    elif config.use_radam:
        optimizer = custom_optim.RAdam(model.parameters(), lr=config.lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=config.lr)

    return optimizer


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


def main(config, model_weight=None, opt_weight=None):
    def print_config(config):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(vars(config))
    print_config(config)

    loader = DataLoader(
        config.train,                           # Train file name except extention, which is language.
        config.valid,                           # Validation file name except extension.
        (config.lang[:2], config.lang[-2:]),    # Source and target language.
        batch_size=config.batch_size,
        device=-1,                              # Lazy loading
        max_length=config.max_length,           # Loger sequence will be excluded.
        dsl=False,                              # Turn-off Dual-supervised Learning mode.
    )

    input_size, output_size = len(loader.src.vocab), len(loader.tgt.vocab)

    # print(f"input_size : {input_size}, output_size : {output_size}")
    # input_size : 8099 (src.vocab_length) , output_size : 6249 (tgt.vocab_length)

    model = get_model(input_size, output_size, config)
    crit = get_crit(output_size, data_loader.PAD)

    # continue_train 시 사용
    if model_weight is not None:
        model.load_state_dict(model_weight)

    # Pass models to GPU device if it is necessary.
    if config.gpu_id >= 0:
        model.cuda(config.gpu_id)
        crit.cuda(config.gpu_id)

    optimizer = get_optimizer(model, config)

    # continue_train 시 사용
    if opt_weight is not None and (config.use_adam or config.use_radam):
        optimizer.load_state_dict(opt_weight)

    lr_scheduler = get_scheduler(optimizer, config)

    if config.verbose >= 2:
        print(model)
        print(crit)
        print(optimizer)

    # Start training. This function maybe equivalent to 'fit' function in Keras.
    mle_trainer = SingleTrainer(MaximumLikelihoodEstimationEngine, config)
    mle_trainer.train(
        model,
        crit,
        optimizer,
        train_loader=loader.train_iter,
        valid_loader=loader.valid_iter,
        src_vocab=loader.src.vocab,
        tgt_vocab=loader.tgt.vocab,
        n_epochs=config.n_epochs,
        lr_scheduler=lr_scheduler,
    )

    if config.rl_n_epochs > 0:
        optimizer = optim.SGD(model.parameters(), lr=config.rl_lr)
        mrt_trainer = SingleTrainer(MinimumRiskTrainingEngine, config)

        mrt_trainer.train(
            model,
            None,    # We don't need criterion for MRT.
            optimizer,
            train_loader=loader.train_iter,
            valid_loader=loader.valid_iter,
            src_vocab=loader.src.vocab,
            tgt_vocab=loader.tgt.vocab,
            n_epochs=config.rl_n_epochs,
        )


if __name__ == '__main__':
    config = define_argparser()
    main(config)
