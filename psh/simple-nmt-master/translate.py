import argparse
import sys
import codecs
from operator import itemgetter

import torch

from simple_nmt.data_loader import DataLoader
import simple_nmt.data_loader as data_loader
from simple_nmt.models.seq2seq import Seq2Seq
from simple_nmt.models.transformer import Transformer


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument(
        '--model_fn',
        required=True,
        help='Model file name to use'
    )
    p.add_argument(
        '--gpu_id',
        type=int,
        default=-1,
        help='GPU ID to use. -1 for CPU. Default=%(default)s'
    )

    p.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Mini batch size for parallel inference. Default=%(default)s'
    )
    p.add_argument(
        '--max_length',
        type=int,
        default=255,
        help='Maximum sequence length for inference. Default=%(default)s'
    )
    p.add_argument(
        '--n_best',
        type=int,
        default=1,
        help='Number of best inference result per sample. Default=%(default)s'
    )
    p.add_argument(
        '--beam_size',
        type=int,
        default=5,
        help='Beam size for beam search. Default=%(default)s'
    )
    p.add_argument(
        '--lang',
        type=str,
        default=None,
        help='Source language and target language. Example: enko'
    )
    p.add_argument(
        '--length_penalty',
        type=float,
        default=1.2,
        help='Length penalty parameter that higher value produce shorter results. Default=%(default)s',
    )

    config = p.parse_args()

    return config


# 여러줄의 문장을 표준입력으로 받아서 처리
def read_text(batch_size=128):
    # This method gets sentences from standard input and tokenize those.
    lines = []

    sys.stdin = codecs.getreader("utf-8")(sys.stdin.detach())

    # 배치 사이즈만큼의 문장들만 처리하기 위한 로직
    for line in sys.stdin:
        if line.strip() != '':
            lines += [line.strip().split(' ')]  # 문장에서 앞,뒤 공백 제거 후 공백기준으로 단어단위 분리

        if len(lines) >= batch_size:
            yield lines
            lines = []

    if len(lines) > 0:
        yield lines


def to_text(indice, vocab):
    # This method converts index to word to show the translation result.
    lines = []

    for i in range(len(indice)):    # 배치사이즈 내에서 한 문장씩
        line = []
        for j in range(len(indice[i])): # 한 문장내에서 한 단어씩
            index = indice[i][j]
            if index == data_loader.EOS:
                # line += ['<EOS>']
                break
            else:
                line += [vocab.itos[index]]
        line = ' '.join(line)
        lines += [line]

    return lines


def is_dsl(train_config):
    # return 'dsl_lambda' in vars(train_config).keys()
    return not ('rl_n_epochs' in vars(train_config).keys())


# input, output data load
def get_vocabs(train_config, config, saved_data):
    if is_dsl(train_config):    # DSL
        assert config.lang is not None

        if config.lang == train_config.lang:
            is_reverse = False
        else:
            is_reverse = True

        if not is_reverse:
            # Load vocabularies from the model.
            src_vocab = saved_data['src_vocab']
            tgt_vocab = saved_data['tgt_vocab']
        else:
            src_vocab = saved_data['tgt_vocab']
            tgt_vocab = saved_data['src_vocab']

        return src_vocab, tgt_vocab, is_reverse
    else:
        # Load vocabularies from the model.
        src_vocab = saved_data['src_vocab']
        tgt_vocab = saved_data['tgt_vocab']

    return src_vocab, tgt_vocab, False


def get_model(input_size, output_size, train_config, is_reverse=False):
    # Declare sequence-to-sequence model.
    if 'use_transformer' in vars(train_config).keys() and train_config.use_transformer:
        model = Transformer(
            input_size,
            train_config.hidden_size,
            output_size,
            n_splits=train_config.n_splits,
            n_enc_blocks=train_config.n_layers,
            n_dec_blocks=train_config.n_layers,
            dropout_p=train_config.dropout,
        )
    else:
        model = Seq2Seq(
            input_size,
            train_config.word_vec_size,
            train_config.hidden_size,
            output_size,
            n_layers=train_config.n_layers,
            dropout_p=train_config.dropout,
        )

    # 저장한 모델에서 사용 한 기울기정보, 하이퍼파라메터 정보 가져옴
    if is_dsl(train_config):
        if not is_reverse:
            model.load_state_dict(saved_data['model'][0])
        else:
            model.load_state_dict(saved_data['model'][1])
    else:
        model.load_state_dict(saved_data['model'])  # Load weight parameters from the trained model.
    model.eval()  # We need to turn-on the evaluation mode, which turns off all drop-outs.

    return model


if __name__ == '__main__':
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    config = define_argparser()

    # Load saved model.
    saved_data = torch.load(
        config.model_fn,
        map_location='cpu',
    )

    # Load configuration setting in training.
    train_config = saved_data['config']

    # 학습 때 저장했던 모델에서 src,tgt 문장들 가져옴
    src_vocab, tgt_vocab, is_reverse = get_vocabs(train_config, config, saved_data)

    # Initialize dataloader, but we don't need to read training & test corpus.
    # What we need is just load vocabularies from the previously trained model.
    # src,tgt 문장들로 DataLoader 객체 생성
    loader = DataLoader()
    loader.load_vocab(src_vocab, tgt_vocab)

    # 이전 모델 데이터로 모델 객체 생성
    input_size, output_size = len(loader.src.vocab), len(loader.tgt.vocab)
    model = get_model(input_size, output_size, train_config, is_reverse)

    # Put models to device if it is necessary.
    if config.gpu_id >= 0:
        model.cuda(config.gpu_id)

    # 학습 필요 없으니까
    with torch.no_grad():
        # Get sentences from standard input.
        # 배치사이즈 만큼의 문장들이 lines
        for lines in read_text(batch_size=config.batch_size):
            # Since packed_sequence must be sorted by decreasing order of length,
            # sorting by length in mini-batch should be restored by original order.
            # Therefore, we need to memorize the original index of the sentence.
            lengths         = [len(line) for line in lines]
            # lines        [['나는', '학교에', '간다'],
            #              ['나는', '밥을', '먹었다'],
            #              ['안녕', '하세요'],
            #              ['안녕']]
            # lengths [3, 3, 2, 1]
            # original_indice [0, 1, 2, 3]
            # sorted_tuples [(['나는', '학교에', '간다'], 3, 0), (['나는', '밥을', '먹었다'], 3, 1), (['안녕', '하세요'], 2, 2), (['안녕'], 1, 3)]

            original_indice = [i for i in range(len(lines))]

            sorted_tuples = sorted(
                zip(lines, lengths, original_indice),
                key=itemgetter(1),
                reverse=True,
            )

            sorted_lines    = [sorted_tuples[i][0] for i in range(len(sorted_tuples))]
            lengths         = [sorted_tuples[i][1] for i in range(len(sorted_tuples))]
            original_indice = [sorted_tuples[i][2] for i in range(len(sorted_tuples))]

            # Converts string to list of index.
            x = loader.src.numericalize(        # string 으로 이뤄진 tensor객체들을 단어사전의 index로 표현
                loader.src.pad(sorted_lines),   # 배치사이즈 내 문장 최대길이만큼 패드 추가 <pad>
                device='cuda:%d' % config.gpu_id if config.gpu_id >= 0 else 'cpu'
            )

            # |x| = (batch_size, length)
            print("src.vocab len : ", len(loader.src.vocab))
            print("tgt.vocab len : ", len(loader.tgt.vocab))
            if config.beam_size == 1:
                y_hats, indice = model.search(x)

                # |y_hats| = (batch_size, length, output_size) emb된 값이 들어있음
                # |indice| = (batch_size, length)   y_hat의 index가 들어있음

                output = to_text(indice, loader.tgt.vocab)
                sorted_tuples = sorted(zip(output, original_indice), key=itemgetter(1))
                output = [sorted_tuples[i][0] for i in range(len(sorted_tuples))]

                sys.stdout.write('\n'.join(output) + '\n')
            else:
                # Take mini-batch parallelized beam search.
                batch_indice, _ = model.batch_beam_search(
                    x,
                    beam_size=config.beam_size,
                    max_length=config.max_length,
                    n_best=config.n_best,
                    length_penalty=config.length_penalty,
                )

                # Restore the original_indice.
                output = []
                for i in range(len(batch_indice)):
                    output += [to_text(batch_indice[i], loader.tgt.vocab)]
                sorted_tuples = sorted(zip(output, original_indice), key=itemgetter(1))
                output = [sorted_tuples[i][0] for i in range(len(sorted_tuples))]

                for i in range(len(output)):
                    sys.stdout.write('\n'.join(output[i]) + '\n')
