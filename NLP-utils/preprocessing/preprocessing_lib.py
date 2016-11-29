
# Pre-processing module which handles: 1) Stop-word removal 2) Vocab builder 3)
# Vocab Pruner 4) Text to int. sequence generator 5) Seq. padding

from __future__ import print_function
import itertools
import os, glob, json, string
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import gensim
from gensim import utils
from collections import defaultdict
import pdb



class Preprocess(object):
    '''
    Class to wrap basic NLP functions (tokenization, bow representaion etc
    '''
    def __init__(self, path):
        self.path = path


    def filter_text_gensim(self, text):
        '''
        Filters stop words from text (DEPRECATED)
        '''
        return [token for token in utils.simple_preprocess(text) if token \
                not in STOPWORDS]

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
        print(tokens)
        return tokens


    def doc_stream_gen(self):
        '''
        Generates a document stream, later to be consumed by other functions
        '''
        if os.path.isdir(self.path):
            # Find all '*.txt' files in the folder
            os.chdir(self.path)
            for File in glob.glob("*.txt"):
                f = open(File, 'r')
                for line in f:
                    tokens = self.filter_text(line.rstrip())
                    yield tokens
                            
            
        elif os.path.isfile(self.path):
            f = open(self.path, 'r')
            ind = 1
            for line in f:
                tokens = self.filter_text(line.rstrip())
                yield tokens

        
    def gen_vocab(self):
        '''
        Generate vocabulary from 
        '''
        doc_stream = (tokens for tokens in self.doc_stream_gen())
        self.id2word = gensim.corpora.Dictionary(doc_stream)
        pdb.set_trace()


    def gen_bow(self):
        '''
        Creates a bow stream of the corpus
        '''
        if os.path.isdir(self.path):
            # Find all '*.txt' files in the folder
            os.chdir(self.path)
            for File in glob.glob("*.txt"):
                f = open(File, 'r')
                for line in f:
                    tokens = self.filter_text(line.rstrip())
                    yield self.id2word.doc2bow(tokens)
                            
            
        elif os.path.isfile(self.path):
            f = open(self.path, 'r')
            ind = 1
            for line in f:
                tokens = self.filter_text(line.rstrip())
                yield self.id2word.doc2bow(tokens)


    def str2int(self, text):
        '''
        Returns bow representation of the string passed.
        '''
        return self.id2word.doc2bow(self.filter_text(text))


    def str2seq(self, text):
        '''
        Returns integer sequence corresponding to dictionary values of tokens
        '''
        tokens = self.filter_text(text.rstrip())
        vals = self.id2word.values()
        vals.append('UNK')
        keys = self.id2word.keys()
        keys.append(len(vals))
        vocab = dict(zip(keys, vals))
        seq = [vocab.get(token, vocab['UNK']) for token in tokens]
        

    def gen_tfidf(self):
        '''
        Initializes tf-idf transformer from data.
        '''
        bow = self.gen_bow()
        self.tfidf = gensim.models.TfidfModel(gen_bow, id2word=id2word)


    def get_tfidf(self, text):
        '''
        Get the tf-df representation of text
        '''
        tokens = self.filter_text(text.rstrip())
        return self.tfidf[self.id2word.doc2bow(tokens)]


    def padding(self, text, k):
        '''
        Convert a given input string to (padded) seq of ints
        '''
        pass


def main(path):
    preprocess = Preprocess(path)
    preprocess.filter_text("I am a student's studying at thez International Institute of Information Technology first line")
    pdb.set_trace()
    preprocess.gen_vocab()
    string = preprocess.str2int('I am a student studying at International Institute of Information Technology first line')
    print(string)
    #print preprocess.padding('I am a student studying at International Institute of Information Technology', 10)
    #print string
    pdb.set_trace()


if __name__ == "__main__":
    main('./corpus.txt')
