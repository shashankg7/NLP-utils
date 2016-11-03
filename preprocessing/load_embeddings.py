# Module to generate embeddings for text, through 1) tf-idf 2) word2vec 3) Glove

from gensim.models import Word2Vec
from gensim.parsing.preprocessing import STOPWORDS
import numpy as np
from scipy.sparse import lil_matrix, coo_matrix
from sklearn.feature_extraction import TfidfTransformer
from collections import Counter
import json


class Embeddings(object):
    # Embeds a document stream using various embedding schemes :
    # 1)    TF-IDF
    # 2)    Word2vec
    # 3)    Doc2vec
    def __init__(self, path):
        # path of the vocab
        self.vocab_path = path
        f = open(path, 'r')
        self.vocab = json.load(f)
        self.terms = self.vocab.keys()


    def filter_text(self, text):
        # Filters stop words from text
        return [token for token in utils.simple_preprocess(text) if token \
                not in STOPWORDS]

    def tf(self):
        row_ind = 0
        row = []
        col = []
        data = []
        # Check if the file is file or folder
        if os.path.isdir(self.path):
            # Find all '*.txt' files in the folder
            os.chdir(self.path)
            for File in glob.glob("*.txt"):
                f = open(File, 'r')
                for line in f:
                    ids = [self.vocab.get(token, 'UNK') in filter_text(line.rstrip())]
                    freq = Counter(ids)
                    col.extend(freq.keys())
                    rows.extend([row_ind] * len(col))
                    data.extend(freq.values())

            count_mat = coo_matrix((data, row, col))
            count_mat = count_mat.tocsc()
            return count_mat

        elif os.path.isfile(self.path):
            print("ValueError: Pre-process the text to one document per file")
            exit(0)


    def tf_idf(self):
        tf_mat = self.tf()
        return TfidfTransformer.fit_transform(tf_mat)


    def load_word2vec(self, w2v_path, w2v_dim):
        word2vec = Word2Vec.load_word2vec_format(w2v_path, binary=True)
        embeddings = np.zeros((len(self.vocab), w2v_dim))
        # Sample a uniform random vector for initializing unseen words
        np.random.seed(42)
        rand_vec = np.random.rand(w2v_dim)

        for k, v in self.vocab.iteritems():
            try:
                embeddings[v] = word2vec[k]
            except:
                embeddings[v] = rand_vec

        return embeddings


