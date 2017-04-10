import directories
import ipdb
import pickle
from tqdm import tqdm

class WordVector:
    def __init__(self):
        self.vocabulary = {}
        self.vectors = []
        self.d = directories.VEC_SIZE
        id = 0 - directories.VEC_SIZE
        with open(directories.WORD_VECTOR) as f:
            for line in f:
                split = line.split()
                word = split[0]
                if not word in self.vocabulary:
                    vec = [float(i) for i in split[id:]]
                    self.vocabulary[word] = len(self.vectors)
                    self.vectors.append(vec)

    def get(self, w):
        if w in self.vocabulary:
            return self.vectors[self.vocabulary[w]]
        return self.vectors[self.add_vector(w)]

    def add_vector(self, w):
        if w not in self.vocabulary:
            self.vocabulary[w] = len(self.vectors)
            self.vectors.append([float(0) for i in range(50)])
            return self.vocabulary[w]
