import data_loader


if __name__ == '__main__':

    loaders = data_loader.DataLoader(
        train_fn='./review.sorted.uniq.refined.tok.shuf.train.tsv',
        batch_size=256,
        valid_ratio=.2,
        device=-1,
        max_vocab=999999,
        min_freq=5,
    )

    print("# Check loader")
    print("| train | = %d" % len(loaders.train_loader.dataset))
    print("| valid | = %d" % len(loaders.valid_loader.dataset))
    print("| vocab | = %d" % len(loaders.text.vocab))
    print("| label | = %d" % len(loaders.label.vocab))

    print("\n# Get mini-batch tensors")
    data = next(iter(loaders.train_loader))

    print("text shape : {}".format(data.text.shape))
    print("label shape : {}".format(data.label.shape))

    print("\n# Use vocab")
    word = '구매'
    print("vocab stoi '{}' : {}".format(word, loaders.text.vocab.stoi[word]))

    num = 7
    print("vocab itos '{}' : {}".format(num, loaders.text.vocab.itos[num]))

    print("\n# Check most frequent words")
    for i in range(30):
        word = loaders.text.vocab.itos[i]
        print('%5d: %s\t%d' % (i, word, loaders.text.vocab.freqs[word]))

    print("\n# Restore text from tensor")
    x = data.text[-1]
    print(x)

    line = []
    for x_i in x:
        line += [loaders.text.vocab.itos[x_i]]

    print(' '.join(line))
