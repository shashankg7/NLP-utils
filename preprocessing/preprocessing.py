
# Pre-processing module which handles: 1) Stop-word removal 2) Vocab builder 3)
# Vocab Pruner 4) Text to int. sequence generator 5) Seq. padding

from __future__ import print_function
import itertools
import os, glob, json, string
import numpy as np
from scipy.sparse import csc_matrix, coo_matrix
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import gensim
from gensim import utils
from collections import defaultdict, Counter
from batch_generator import batch_gen
from gensim.matutils import Sparse2Corpus
import pdb


class Preprocess(object):
    '''
    Class to wrap basic NLP functions (tokenization, bow representaion etc
    '''
    def __init__(self, path):
        '''
        constructor
        '''
        self.path = path
        self.vocab = defaultdict(int)
        self.vocab_freq = defaultdict(int)


    def filter_text(self, text):
        '''
        Filters stop words from text
        '''
        text = text.rstrip().lower()
        punctuations = set(punctuation)
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in punctuations]
        tokens = [token.translate(None, string.punctuation) for token in tokens if token not in stopwords.words('english')]
        tokens = [token for token in tokens if len(token) > 1]
        return tokens


    def gen_vocab(self):
        '''
        Generate vocabulary from 
        '''
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

        else:
            print("No text file or directory of text present")
            exit(0)
        pdb.set_trace()
        f_json.close()


    def prune_vocab(self, k):
        '''
        Prune terms from vocab with freq. < k
        '''
        f = open('./vocab.json', 'w')
        # Prune the vocab to include top-k frequent tokens
        # Sort vocab items in decreasing order of freq.
        self.vocab_filter = sorted(self.vocab_freq, key=self.vocab_freq.get)
        # Take top k entries
        self.filter_tokens = self.vocab_filter[:k]
        vocab = dict((k, v) for k, v in zip(self.filter_tokens, \
                                                 range(1, \
                                                    len(self.filter_tokens) +1)))
        vocab['UNK'] = len(vocab) + 1
        print("Pruning vocab")
        json.dump(vocab, f)
        print("Dumped")


    def str2int(self, text):
        '''
        Convert a new phrase into int seq corresponding to dict generated
        '''
        if os.path.isfile('./vocab.json'):
            f = open('./vocab.json', 'r')
            vocab = json.load(f)
            tokens = self.filter_text(text)
            return [vocab.get(token, vocab['UNK']) for token in tokens]

        else:
            print("ValueError : Vocab not present, please run vocab gen function\
                  first")
            exit(0)


    def padding(self, text, k):
        '''
        Convert a given input string to (padded) seq of ints
        '''
        if os.path.isfile('vocab.json'):
            f = open('vocab.json', 'r')
            vocab = json.load(f)

            tokens = self.str2int(text)
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


    def textseq_batch(self, k, batch_size=32):
        '''
        Generates batch of token sequence from text corpus
        '''
        data = []
        if os.path.isdir(self.path):
            # Find all '*.txt' files in the folder
            os.chdir(self.path)
            for File in glob.glob("*.txt"):
                f = open(File, 'r')
                for line in f:
                    data.append(self.padding(line, k))       
            

        elif os.path.isfile(self.path):
            f = open(self.path, 'r')
            ind = 1
            for line in f:
                data.append(self.padding(line, k))       

        else:
            print("No text file or directory of text present")
            exit(0)

        data = np.array(data)
        return batch_gen(data, batch_size)


    def termfreq_gen(self):
        '''
        Generates term-frequence matrix from the corpus
        '''
        # Initialize variables for row, col and data for converting to sparse matrix
        row, col, data = [], [], []
        rowind = 0
        if os.path.isdir(self.path):
            # Find all '*.txt' files in the folder
            os.chdir(self.path)
            for File in glob.glob("*.txt"):
                f = open(File, 'r')
                file_tokens = []
                for line in f:
                    file_tokens.extend(self.filter_text(line))

                term_freqs = Counter(filter_tokens)
                terms = map(lambda x:self.vocab.get(x, vocab['UNK']), term_freqs.keys())
                freq = term_freqs.values()
                row.extend([rowind] * len(terms))
                col.extend(terms)
                data.extend(freq)
                rowind += 1


        elif os.path.isfile(self.path):
            f = open(self.path, 'r')
            
            for line in f:
                file_tokens = self.filter_text(line)
                term_freqs = Counter(file_tokens)
                terms = map(lambda x:self.vocab.get(x, self.vocab['UNK']), term_freqs.keys())
                freq = term_freqs.values()
                row.extend([rowind] * len(terms))
                col.extend(terms)
                data.extend(freq)
                rowind += 1

        else:
            print("No text file or directory of text present")
            exit(0)


        tf_matrix = coo_matrix((data, (row, col)))
        return tf_matrix.tocsc()


    def gensim_corpusgen(self):
        '''
        Converts sparse term-doc matrix to gensim corpus format
        '''
        termfreq_matrix = self.termfreq_gen()
        return Sparse2Corpus(termfreq_matrix)


    def lsa(self, k):
        '''
        Performs LSA using gensim on the corpus
        '''
        pass

    def lda(self, k):
        '''
        Performs LDA on the corpus using gensim
        '''
        


def main(path):
    preprocess = Preprocess(path)
    preprocess.gen_vocab()
    #preprocess.prune_vocab(50000)
    string = preprocess.str2int('I am a student, first and second')
    print(string)
    print(preprocess.padding('I am a student, first and second', 20))
    textseq_batch = preprocess.textseq_batch(4, 2)
    for batch in textseq_batch:
        print(batch.shape)
    x = preprocess.termfreq_gen()
    pdb.set_trace()


if __name__ == "__main__":
    main('./corpus.txt')
