import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

import data_loader as data_loader
# from search import SingleBeamSearchBoard

class Encoder(nn.Module):
    def __init__(self, word_vec_size, hidden_size, n_layers=4, dropout_p=.2):
        super(Encoder, self).__init__()

        # nn.LSTM (Input dim, output dim)
        self.rnn = nn.LSTM(
            word_vec_size,
            int(hidden_size/2),
            num_layers=n_layers,
            dropout=dropout_p,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, emb):
        # |emb| = (bs, length, word_vec_size)
        if isinstance(emb, tuple):
            x, lengths = emb
            x = pack(x, lengths.toList(), batch_first=True)
        else:
            x = emb

        y, h = self.rnn(x)
        # |y| (bs, length, hs)
        # |h[0]| (n_layer * 2, bs, hs/2)

        # ?? 왜 unpack ??
        if isinstance(emb, tuple):
            y, _ = unpack(y, batch_first=True)

        return y, h


class Decoder(nn.Module):
    def __init__(self, word_vec_size, hidden_size, n_layers=4, dropout_p=.2):
        super(Decoder, self).__init__()

        # encoder 의 hidden state 값 사이즈 : |h_t_1[0]| : (n_layers, bs, hs)
        self.rnn = nn.LSTM(
            word_vec_size + hidden_size,
            hidden_size,
            num_layers=n_layers,
            dropout=dropout_p,
            bidirectional=False,
            batch_first=True,
        )

    def forward(self, emb_t, h_t_1_tilde, h_t_1):
        # |emb_t| : (bs, 1, word_vec_size)
        # |h_t_1_tilde| : (bs, 1, hs)
        # |h_t_1[0]| : (n_layers, bs, hs)
        batch_size = emb_t.size(0)
        hidden_size = h_t_1[0].size(-1)

        # BOS의 h_t_1 : # If this is the first time-step
        if h_t_1_tilde is None:
            h_t_1_tilde = emb_t.new(batch_size, 1, hidden_size).zero_()

        # Input Feeding
        x = torch.cat([emb_t, h_t_1_tilde], dim=-1)

        y, h = self.rnn(x, h_t_1)

        return y, h


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        # x, y
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, h_src, h_t_tgt, mask=None):
        # |h_src| = (bs, length, hs)
        # |h_t_tgt| = (bs, 1, hs)
        # |mask| = (bs, length)

        # Q : query = H(dec)_t * Wa
        query = self.linear(h_t_tgt)
        # |query| = (bs, 1, hs)

        # Q * K : query * H(enc)_all-T
        weight = torch.bmm(query, h_src.transpose(1, 2))
        # (bs, 1, hs) (bs, hs, length)
        # |weight| = (bs, 1, length)

        # softmax 전에 mask 작업
        if mask is not None:
            weight.masked_fill_(mask.unsqueeze(1), -float('inf'))

        # Q, K의 유사도 결과 weight
        weight = self.softmax(weight)

        # V : QK weight * H(end)_all
        context_vector = torch.bmm(weight, h_src)
        # |context_vector| = (bs, 1, hs)

        return context_vector

class Generator(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Generator, self).__init__()

        self.output = nn.Linear(hidden_size, output_size)
        self.sotfmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # |x| : (bs, length, hs)

        y = self.sotfmax(self.output(x))
        # |y| : (bs, length, output_size)

        # log-probability
        return y


class Seq2Seq(nn.Module):
    def __init__(
            self,
            input_size,
            word_vec_size,
            hidden_size,
            output_size,
            n_layers=4,
            dropout_p=.2
    ):
        self.input_size = input_size
        self.word_vec_size = word_vec_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        super(Seq2Seq, self).__init__()

        self.emb_src = nn.Embedding(input_size, word_vec_size)
        self.emb_dec = nn.Embedding(output_size, word_vec_size)

        self.encoder = Encoder(
            word_vec_size, hidden_size,
            n_layers=n_layers, dropout_p=dropout_p,
        )
        self.decoder = Decoder(
            word_vec_size, hidden_size,
            n_layers=n_layers, dropout_p=dropout_p,
        )
        self.attn = Attention(hidden_size)

        self.concat = nn.Linear(hidden_size*2, hidden_size)
        self.tanh = nn.Tanh()
        self.generator = Generator(hidden_size, output_size)

    # mask 생성 : max 이하 길이 문장
    # 값이 있는 부분 : 0
    # 값이 없는 부분 : 1
    def generate_mask(self, x, length):
        mask = []

        max_length = max(length)
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

    # old : encoder 의 bidirectional hiddens 들을 한개로 만들기 : h/2 -> h 로 만들기
    def merge_encoder_hidens(self, encoder_hiddens):
        new_hiddens = []
        new_cells = []

        hiddens, cells = encoder_hiddens

        # i-th and (i+1)-th layer is opposite direction.
        # Also, each direction of layer is half hidden size.
        # Therefore, we concatenate both directions to 1 hidden size layer.
        for i in range(0, hiddens.size(0), 2):
            new_hiddens += [torch.cat([hiddens[i], hiddens[i + 1]], dim=-1)]
            new_cells += [torch.cat([cells[i], cells[i + 1]], dim=-1)]

        new_hiddens, new_cells = torch.stack(new_hiddens), torch.stack(new_cells)

        return (new_hiddens, new_cells)

    # Ner
    def fast_merge_encoderr_hiddens(self, encoder_hiddens):
        h_0_tgt, c_0_tgt = encoder_hiddens
        batch_size = h_0_tgt.size(1)

        h_0_tgt = h_0_tgt.transpose(0, 1).contiguous().view(batch_size,
                                                            -1,
                                                            self.hidden_size
                                                            ).transpose(0, 1).contiguous()
        c_0_tgt = c_0_tgt.transpose(0, 1).contiguous().view(batch_size,
                                                            -1,
                                                            self.hidden_size
                                                            ).transpose(0, 1).contiguous()
        return h_0_tgt, c_0_tgt

    def forward(self, src, tgt):
        batch_size = tgt.size(0)
        mask = None
        x_length = None
        if isinstance(src, tuple):
            x, x_length = src
            # attention 계산 손실을 막아준다
            mask = self.generate_mask(x, x_length)
        else:
            x = src
        if isinstance(tgt, tuple):
            tgt = tgt[0]

        # Encoder
        emb_src = self.emb_src(x)
        h_src, h_0_tgt = self.encoder((emb_src, x_length))
        h_0_tgt = self.fast_merge_encoderr_hiddens(h_0_tgt)

        # Decoder
        emb_tgt = self.emb_dec(tgt)

        h_tilde = []
        h_t_tilde = None
        decoder_hidden = h_0_tgt

    def search(self, src, is_greedy=True, max_length=255):
        # Encoder 준비
        if isinstance(src, tuple):
            x, x_length = src
            # padding
            mask = self.generate_mask(x, x_length)
        else:
            x, x_length = src, None
            mask = None
        batch_size = x.size(0)

        # Encoder
        emb_src = self.emb_src(x)
        h_src, h_0_tgt = self.encoder(emb_src)
        # Decoder 입력 size로 변환
        decoder_hidden = self.fast_merge_encoderr_hiddens(h_0_tgt)

        # Decoder 준비
        # : 최초 입력은 BOS
        y = x.new(batch_size, 1).zero_()+data_loader.BOS

        # 연산 종료 확인 : 모두 true
        is_decoding = x.new_ones(batch_size, 1).bool()

        # 중간 데이터
        h_t_tilde, y_hats, indice = None, [], []


        # 문장 생성
        while is_decoding.sum > 0 and len(indice) < max_length:
            # time step 별로 진행
            emb_t = self.emb_dec(y)
            decoder_output, decoder_hidden = self.decoder(emb_t, h_t_tilde, decoder_hidden)
            context_vector = self.attn(h_src, decoder_output, mask)

            h_t_tilde = self.hanh(self.concat(torch.cat([decoder_output,
                                                        context_vector], dim=-1)))
            y_hat = self.generator(h_t_tilde)
            # logSoftmax 결과
            # |y_hat| = (batch_size, 1, output_size)

            # bs 전체 y_hat
            y_hats += [y_hat]
            # bs의 time step별 모음

            # argmax
            if is_greedy:
                y = y_hat.argmax(dim=-1)
                # |y| = (batch_size, 1)
            else:
                y = torch.multinomial(y_hat.exp().view(batch_size, -1), 1)

            # 끝난 문장뒤에 pad 추가
            y = y.masked_fill_(~is_decoding, data_loader.PAD)
            # and 연산 : true * 같으면 False = false
            is_decoding = is_decoding * torch.ne(y, data_loader.EOS)

            # 추론 결과값
            indice += [y]

        # seq 순서로 bs cat
        y_hats = torch.cat(y_hats, dim=1)
        indice = torch.cat(indice, dim=1)
        # |y_hats| = (batch_size, length, output_size)
        # |indice| = (batch_size, length)

        return y_hats, indice