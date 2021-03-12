import os
from torchtext import data, datasets
import torch
import torch.nn as nn
import seq2seq
import data_loader as data_loader


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

            if index == data_loader.EOS:
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



if __name__ == '__main__':
    loader = data_loader.DataLoader(
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

    # x
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

