import random
import torch
import torch.nn as nn
import torch.optim as optim

import seq2seq as seq

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

raw = ["I feel hungry.	나는 배가 고프다.",
       "Pytorch is very easy.	파이토치는 매우 쉽다.",
       "Pytorch is a framework for deep learning.	파이토치는 딥러닝을 위한 프레임워크이다.",
       "Pytorch is very clear to use.	파이토치는 사용하기 매우 직관적이다."]

SOS_token = 0
EOS_token = 1

# 단어 사전 : word 사전, index 사전, 출현빈도 사전
# 신규 단어 : 사전 추가
# 기존 단어 : 출현 count +1
class Vocab:
    def __init__(self):
        self.vocab2index = {"<SOS>": SOS_token, "<EOS>":EOS_token}
        self.index2vocab = {SOS_token: "<SOS>", EOS_token: "<EOS>"}
        self.vocab_count = {}
        self.n_vocab = len(self.vocab2index)

    def add_vocab(self, sentence):
        for word in sentence.split(" "):
            if word not in self.vocab2index:
                self.vocab2index[word] = self.n_vocab
                self.vocab_count[word] = 1
                self.index2vocab[self.n_vocab] = word
                self.n_vocab += 1
            else:
                self.vocab_count[word] += 1


# 문장이 max보다 길면 pairs 입력하지 않는다.
def filter_pair(pair, source_max_length, target_max_length):
    return len(pair[0].split(" ")) < source_max_length and len(pair[1].split(" ")) < target_max_length

# 라인별로 읽어서 한글:영어 pairs를 만든다
# pair[0] : 영어 - source_vocab
# pair[1] : 한글 - target_vocab
def preprocess(corpus, source_max_length, target_max_length):
    print("reading corpus...")

    pairs = []
    for line in corpus:
        print(line.strip().lower().split("\t"))
        pairs.append([s for s in line.strip().lower().split("\t")])
    print(pairs)
    print("Read {} sentence pairs".format(len(pairs)))

    # 문장 사이즈가 max보다 크면 제외
    pairs = [pair for pair in pairs if filter_pair(pair, source_max_length, target_max_length)]
    print("Trimmed to {} sentence pairs".format(len(pairs)))

    # 문장들의 단어 사전 생성
    source_vocab = Vocab()
    target_vocab = Vocab()

    print("Counting words...")
    for pair in pairs:
        source_vocab.add_vocab(pair[0])
        target_vocab.add_vocab(pair[1])
    print("source voacb size = ", source_vocab.n_vocab)
    print("target vocab size = ", target_vocab.n_vocab)

    print("source_vocab.vocab_count", source_vocab.vocab_count)
    print("target_vocab.vocab_count", target_vocab.vocab_count)

    return pairs, source_vocab, target_vocab

if __name__ == '__main__':
    SOURCE_MAX_LENGTH = 10
    TARGET_MAX_LENGTH = 12

    enc_hidden_size = 16
    dec_hidden_size = enc_hidden_size


    load_pairs, load_source_vocab, load_target_vocab = preprocess(raw, SOURCE_MAX_LENGTH, TARGET_MAX_LENGTH)
    print('###')
    print(random.choice(load_pairs))
    print('###')

    # rparis = random.choice(load_pairs)
    rpairs = load_pairs[0]
    print(rpairs)
    print(rpairs[0])
    print(rpairs[1])

    src_count = load_source_vocab.n_vocab
    print('src_count :', src_count)
