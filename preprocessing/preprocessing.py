
# Pre-processing module which handles: 1) Stop-word removal 2) Vocab builder 3)
# Vocab Pruner 4) Text to int. sequence generator 5) Seq. padding

import os, glob, json
from gensim.parsing.preprocessing import STOPWORDS
from gensim import utils
from collections import defaultdict
import pdb

class Preprocess(object):

    def __init__(self, path):
        self.path = path


    def filter_text(self, text):
        # Filters stop words from text
        return [token for token in utils.simple_preprocess(text) if token \
                not in STOPWORDS]


    def gen_vocab(self):
        self.vocab_freq = defaultdict(int)
        self.vocab = {}
        f_json = open('vocab.json', 'w')
        # Check if the file is file or folder
        # Starting index from 1 for 0 padding
        ind = 1
        if os.path.isdir(self.path):
            # Find all '*.txt' files in the folder
            os.chdir(self.path)
            for File in glob.glob("*.txt"):
                f = open(File, 'r')
                for line in f:
                    for token in self.filter_text(line.rstrip()):
                        if token not in self.vocab:
                            self.vocab[token] = ind
                            ind += 1
                        self.vocab_freq[token] += 1
            self.vocab['UNK'] = ind
            json.dump(self.vocab, f_json)

        elif os.path.isfile(self.path):
            f = open(self.path, 'r')
            ind = 1
            for line in f:
                for token in self.filter_text(line.rstrip()):
                    if token not in self.vocab:
                        self.vocab[token] = ind
                        ind += 1
                    self.vocab_freq[token] += 1
            self.vocab['UNK'] = ind
            json.dump(self.vocab, f_json)


    def prune_vocab(self, k):
        f = open('./vocab_filter', 'w')
        f.write('dfdfd')
        print f
        # Prune the vocab to include top-k frequent tokens
        # Sort vocab items in decreasing order of freq.
        self.vocab_filter = sorted(self.vocab_freq, key=self.vocab_freq.get)
        # Take top k entries
        self.filter_tokens = self.vocab_filter[:k]
        pdb.set_trace()
        vocab = dict((k, v) for k, v in zip(self.filter_tokens, \
                                                 range(1, \
                                                    len(self.filter_tokens) +1)))
        vocab['UNK'] = len(vocab) + 1
        print("Pruning vocab")
        json.dump(vocab, f)
        print("Dumped")

    def str2int(self, text):
        # Convert a new phrase into int seq corresponding to dict generated
        if os.path.isfile('vocab.json'):
            f = open('vocab.json', 'r')
            vocab = json.load(f)
            return [vocab[token] for token in utils.simple_preprocess(text) \
                    if token not in STOPWORDS]

        else:
            print("ValueError : Vocab not present, please run vocab gen function\
                  first")
            exit(0)


    def padding(self, text, k):
        # Convert a given input string to (padded) seq of ints
        if os.path.isfile('vocab.json'):
            f = open('vocab.json', 'r')
            vocab = json.load(f)

            tokens = self.str2int(' '.join([vocab[token] for token in utils.simple_preprocess(text) \
                        if token not in STOPWORDS]))
            if len(tokens) == k:
                return tokens

            elif len(tokens) > k:
                return tokens[:k]

            elif len(tokens) < k:
                tokens1 = [0] * k
                tokens1[:len(tokens)] = tokens
                tokens = tokens1

            return tokens1

        else:
            print("ValueError : Vocab not present, please run vocab gen function\
                  first")
            exit(0)


def main(path):
    preprocess = Preprocess(path)
    preprocess.gen_vocab()
    preprocess.prune_vocab(50000)


if __name__ == "__main__":
    main('../corpus')
