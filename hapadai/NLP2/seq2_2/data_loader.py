import os
from torchtext import data, datasets
import torch
import torch.nn as nn
import seq2seq

from torch import optim
import torch_optimizer as custom_optim

# PAD, BOS, EOS = 100, 200, 300
PAD, BOS, EOS = 1, 2, 3

class DataLoader():

    def __init__(self,
                 train_fn=None,
                 valid_fn=None,
                 exts=None,
                 batch_size=64,
                 device='cpu',
                 max_vocab=99999999,
                 max_length=255,
                 fix_length=None,
                 use_bos=True,
                 use_eos=True,
                 shuffle=True,
                 dsl=False
                 ):

        super(DataLoader, self).__init__()



        self.src = data.Field(
            sequential=True,
            use_vocab=True,
            batch_first=True,
            include_lengths=True,
            fix_length=fix_length,
            init_token='<BOS>' if dsl else None,
            eos_token='<EOS>' if dsl else None,
        )

        self.tgt = data.Field(
            sequential=True,
            use_vocab=True,
            batch_first=True,
            include_lengths=True,
            fix_length=fix_length,
            init_token='<BOS>' if use_bos else None,
            eos_token='<EOS>' if use_eos else None,
        )
        '''
            https://torchtext.readthedocs.io/en/latest/data.html?highlight=data%20field
            
            sequential: 시퀀스 데이터 여부. (True가 기본값)
                        text는 sequential 데이터이므로 인자를 True 로 두고, LABEL 데이터는 순서가 필요없기 때문에 False 로 둔다. 
            use_vocab : 단어 집합을 만들 것인지 여부. (True가 기본값)
              tokenize : 어떤 토큰화 함수를 사용할 것인지 지정. (string.split이 기본값)
              lower : 영어 데이터를 전부 소문자화한다. (False가 기본값)
            batch_first : 미니 배치 차원을 맨 앞으로 하여 데이터를 불러올 것인지 여부. (False가 기본값)
              is_target : 레이블 데이터 여부. (False가 기본값)
            fix_length : 최대 허용 길이. 이 길이에 맞춰서 패딩 작업(Padding)이 진행된다.
            include_lengths : 패딩 된 미니 배치의 튜플과 각 예제의 길이가 포함 된 목록을 반환할지 아니면 패딩 된 미니 배치 만 반환할지 여부입니다. 기본값 : False.
            
            init_token – 이것을 사용하는 모든 예제 앞에 추가 될 토큰 들, 또는 초기 토큰이없는 경우 없음. 기본값 : 없음.
            eos_token – 이것을 사용하는 모든 예제에 추가 될 토큰 들, 또는 문장 끝 토큰이없는 경우 없음. 기본값 : 없음.

            pad_token – The string token used as padding. Default: “<pad>”.
        '''

        if train_fn is not None and valid_fn is not None and exts is not None:

            '''
                경로 : 두 언어 모두에 대한 데이터 파일 경로의 공통 접두어입니다.
                exts : 각 언어의 경로에 대한 확장을 포함하는 튜플.
                fields : 각 언어의 데이터에 사용될 필드가 포함 된 튜플입니다.
                나머지 키워드 인수 : data.Dataset의 생성자에 전달됩니다.
            '''
            train = TranslationDataset(
                path=train_fn,
                exts=exts,
                fields=[('src', self.src), ('tgt', self.tgt)],
                max_length=max_length
            )
            valid = TranslationDataset(
                path=valid_fn,
                exts=exts,
                fields=[('src', self.src), ('tgt', self.tgt)],
                max_length=max_length,
            )

            '''
                BuckerIterator로, 이는 TranslationDataset을 첫번째 인자로 받아 사용하기 쉽습니다. 
                비슷한 길이를 갖는 데이터를 함께 묶는(batch) Iterator를 정의합니다. 
                매 새로운 epoch에서 랜덤한 batch를 생성하는 과정에서 padding을 최소화합니다.
            '''
            # print('train len : ', len(train))
            # print('valid len : ', len(valid))
            #
            # print('train : ', train)

            self.train_iter = data.BucketIterator(
                train,
                batch_size=batch_size,
                device='cuda:%d' % device if device >= 0 else 'cpu',
                shuffle=shuffle,
                sort_key=lambda x: len(x.tgt) + (max_length * len(x.src)),
                sort_within_batch=True,
            )
            self.valid_iter = data.BucketIterator(
                valid,
                batch_size=batch_size,
                device='cuda:%d' % device if device >= 0 else 'cpu',
                shuffle=False,
                sort_key=lambda x: len(x.tgt) + (max_length * len(x.src)),
                sort_within_batch=True,
            )

            '''
                build_vocab은 Positional, keyward argument 두 개를 받는데, 
                positional argument의 경우, Dataset 오브젝트나 iterable한 데이터를 받아 Vocab객체를 생성합니다. 
                keyward argument의 경우 Vocab의 생성자로 전달할 인자를 받습니다.

            '''
            self.src.build_vocab(train, max_size=max_vocab)
            self.tgt.build_vocab(train, max_size=max_vocab)

            # SRC.build_vocab(train_data, min_freq=2)

    def load_vocab(self, src_vocab, tgt_vocab):
        self.src.vocab = src_vocab
        self.tgt.vocab = tgt_vocab


class TranslationDataset(data.Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.src), len(ex.trg))

    '''
                path=train_fn,
                exts=exts,
                fields=[('src', self.src), ('tgt', self.tgt)],
                max_length=max_length
    '''
    def __init__(self, path, exts, fields, max_length=None, **kwargs):
        """Create a TranslationDataset given paths and fields.

        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.

        """
        # 아닌 경우 fields 생성
        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1])]

        # print('fields : ', fields)
        # print('fields : ', fields[0])

        if not path.endswith('.'):
            path += '.'

        # path 끝에 en, ko 붙이기
        src_path, trg_path = tuple(os.path.expanduser(path + x) for x in exts)

        examples = []
        # file open
        with open(src_path, encoding='utf-8') as src_file, open(trg_path, encoding='utf-8') as trg_file:
            # line 씩 읽기
            for src_line, trg_line in zip(src_file, trg_file):
                src_line, trg_line = src_line.strip(), trg_line.strip()
                # print('src_line : ', src_line)
                # print('trg_line : ', trg_line)

                # max len 보다 크면 skip
                if max_length and max_length < max(len(src_line.split()), len(trg_line.split())):
                    continue

                #
                if src_line != '' and trg_line != '':
                    examples += [data.Example.fromlist([src_line, trg_line], fields)]
            # print('examples : ', len(examples))
            # for it in examples:
            #     print('examples : ', it)

            # print('##############')

        super().__init__(examples, fields, **kwargs)


