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


################################################
# mask 생성 : max 이하 길이 문장
# 값이 있는 부분 : 0
# 값이 없는 부분 : 1
def generate_mask(x, length):
    mask = []

    max_length = max(length)
    print(max_length)
    for l in length:
        if max_length - l > 0:
            # 0000000 + 1111
            mask += [torch.cat([x.new_ones(1, l).zero_(),
                                x.new_ones(1, (max_length - l))
                               ], dim=-1)]
        else:
            mask += [x.new_ones(1, l).zero_()]

    # x 전체 모양
    mask = torch.cat(mask, dim=0).bool()
    return mask

# Ner
def fast_merge_encoderr_hiddens(encoder_hiddens, hidden_size):
    h_0_tgt, c_0_tgt = encoder_hiddens
    batch_size = h_0_tgt.size(1)

    h_0_tgt = h_0_tgt.transpose(0, 1).contiguous().view(batch_size,
                                                        -1,
                                                        hidden_size
                                                        ).transpose(0, 1).contiguous()
    c_0_tgt = c_0_tgt.transpose(0, 1).contiguous().view(batch_size,
                                                        -1,
                                                        hidden_size
                                                        ).transpose(0, 1).contiguous()
    return h_0_tgt, c_0_tgt


def to_text(indice, vocab):
    # This method converts index to word to show the translation result.
    lines = []

    for i in range(len(indice)):
        line = []
        for j in range(len(indice[i])):
            index = indice[i][j]

            if index == EOS:
                line += ['<EOS>']
                break
            else:
                # print(index)
                if index < 300:
                    line += [vocab.itos[index]]

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


if __name__ == '__main__':
    loader = DataLoader(
        train_fn='./mini10/corpus.news.shuf.train.10.tok.bpe',
        valid_fn='./mini10/corpus.news.shuf.valid.10.tok.bpe',
        exts=('en', 'ko'),
        batch_size=10,
        device=-1,                              # Lazy loading
        max_length=164  # 164
    )

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

    input_size = len(loader.src.vocab)
    output_size =  len(loader.tgt.vocab)
    print('input_size : ', input_size) # 304
    print('output_size : ', output_size) # 328

    batch_index = None
    batch = None
    for batch_index, batch in enumerate(loader.train_iter):
        pass
        # print('# : ', batch_index)
        # print('batch_index :', batch_index)
        # print('batch :', batch)

    # print(loader.src.vocab..__dict__)

    print('\n ## Seq2Seq ##')
    # Embed
    embed_enc = nn.Embedding(input_size, word_vec_size)
    embed_dec = nn.Embedding(output_size, word_vec_size)

    # Encoder
    encoder = seq2seq.Encoder(word_vec_size, hidden_size, n_layers, dropout)
    # decoder
    decoder = seq2seq.Decoder(word_vec_size, hidden_size, n_layers, dropout)

    # Attention
    attn = seq2seq.Attention(hidden_size)
    linear = nn.Linear(hidden_size*2, hidden_size)
    linear_enc = nn.Linear(hidden_size, hidden_size)
    tanh = nn.Tanh()
    generator = seq2seq.Generator(hidden_size, output_size)


    # todo : Encoder
    print('\n ## Encoder ##')
    # Encoder 준비
    x = None
    x_length = None
    mask = None
    if isinstance(batch.src, tuple):
        x, x_length = batch.src
        print(x.size())
        print(x_length)
    # attention 계산 손실을 막아준다
    mask = generate_mask(x, x_length)
    print('mask : ', mask.size())
    # |mask| = (batch_size, length) mask :  torch.Size([10, 71])

    x_enc = to_text(x, loader.src.vocab)
    listPrint(x_enc)

    # Encoder
    # embed
    emb_enc = embed_enc(x)
    print('emb_enc.size : ', emb_enc.size())
    # emb_enc.size :  torch.Size([10, 71, 5])
    # print(emb_enc)

    output_enc, hidden_enc = encoder(emb_enc)

    # print('output_enc : ', output_enc)
    # print('h_s_enc : ', h_s_enc)
    print('output_enc : ', output_enc.size())  # output_enc: torch.Size([10, 71, 100])
    print('h_s_enc[0] : ', hidden_enc[0].size())  # h_s_enc: torch.Size([2, 10, 384])
    print('h_s_enc[1] : ', hidden_enc[1].size())  # h_s_enc: torch.Size([2, 10, 384])
    # print(output_enc)

    # tilde_enc = tanh(linear_enc(output_enc))
    # bsToTextSrc(tilde_enc, 'src')
    bsToTextSrc(output_enc, 'src')
    # bsToTextTgt(tilde_enc)
    ################################################

    print('\n ## Decoder ##')
    # Decoder 준비
    # y
    tgt = None
    tgt_length = None
    if isinstance(batch.tgt, tuple):
        tgt = batch.tgt[0]
        tgt_length = batch.tgt[1]


    tgt_dec = to_text(tgt, loader.tgt.vocab)
    listPrint(tgt_dec)

    # emb_dec
    emb_dec = embed_dec(tgt)
    print('emb_dec.size : ', emb_dec.size())

    # h_tilde
    h_tilde = []
    h_t_tilde = None


    # h_0_tgt
    h_0_tgt = fast_merge_encoderr_hiddens(hidden_enc, hidden_size)
    print('h_0_tgt : ', h_0_tgt[0].size()) # h_0_tgt :  torch.Size([1, 10, 768])
    decoder_hidden = h_0_tgt


    # Decoder : teach_forching
    print('tgt_length : ', tgt_length)
    # tgt_length :  tensor([64, 60, 52, 41, 43, 38, 44, 35, 45, 22])

    # test
    test_out_dec = []
    test_context_vector = []





    # 첫번째가 mini batch의 max length : 0 ~ 63
    for t in range(tgt.size(1)):

        emb_t = emb_dec[:, t, :].unsqueeze(1)
        output_dec, hidden_dec = decoder(emb_t, h_t_tilde, decoder_hidden)
        context_vector = attn(output_enc, output_dec, mask)
        h_t_tilde = tanh(linear(torch.cat([output_dec, context_vector], dim=-1)))
        # |h_t_tilde| = (batch_size, 1, hidden_size)   torch.Size([10, 1, 100])

        test_out_dec += [output_dec]
        test_context_vector += [context_vector]
        # 총 63번째 time step의 bs 10개씩의
        h_tilde += [h_t_tilde]

        if t == 63:
            print('emb_t : ', emb_t.size())                    # emb_t :  torch.Size([10, 1, 5])
            print('output_dec : ', output_dec.size())          # output_dec :  torch.Size([10, 1, 100])
            print('hidden_dec : ', hidden_dec[0].size())       # torch.Size([1, 10, 100])
            print('context_vector : ', context_vector.size())  # context_vector :  torch.Size([10, 1, 100])
            print('h_t_tilde : ', h_t_tilde.size())            # h_t_tilde :  torch.Size([10, 1, 100])
            print('h_tilde : ', len(h_tilde))

    print('##################')
    print('test_out_dec')
    bsToTextTgt(test_out_dec, 'tgt')

    print('test_context_vector')
    bsToTextTgt(test_context_vector, 'src')

    print('result')
    bsToTextTgt(h_tilde, 'src')


