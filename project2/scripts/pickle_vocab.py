#!/usr/bin/env python3
import pickle


def main():
    vocab = dict()
    with open('../outputs/vocab_cut.txt') as f:
        for idx, line in enumerate(f):
            vocab[line.strip()] = idx

    with open('../outputs/vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
