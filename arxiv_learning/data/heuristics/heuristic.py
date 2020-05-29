import io
import os
import pickle
import gzip
import zipfile
import arxiv_learning.data.load_mathml as load_mathml
import torch_geometric.data
from random import seed
import numpy.random as npr
MEMORY_BUFFERED = False
GZIP = False



class Heuristic(object):
    def __init__(self, basefile="/data/s1/pfahler/arxiv_v2/json_db.zip", test=False):
    # def __init__(self, basefile="/home/pfahler/arxiv_learning/subset_ml_train.zip", test=False):
        self.basefile = basefile
        # self.basefile = "/data/s1/pfahler/arxiv_processed/json_db.zip"
        self.alphabet = load_mathml.load_alphabet(os.path.abspath(
            os.path.join(os.path.split(self.basefile)[0], "vocab.pickle")))
        self.test = test
        if test and "train" in self.basefile:
            self.basefile = self.basefile.replace("train", "test")
        if MEMORY_BUFFERED:
            buffer = io.BytesIO(open(self.basefile, "rb").read())
            self.archive = zipfile.ZipFile(buffer, "r")
        else:
            self.archive = zipfile.ZipFile(self.basefile, "r")
        self.data = self.archive.namelist()
        if (not test and "train" not in self.basefile) or (test and "test" not in self.basefile):
            # TODO: Group according to paper, shuffle papers and take 1/5th of papers
            self.data = sorted(self.data)
            cutoff = len(self.data) // 5
            if test:
                self.data = list([x for i, x in enumerate(self.data) if not i % 5])
            else:
                self.data = list([x for i, x in enumerate(self.data) if i % 5])
        if len(self.data) > 100000:
            self.data = npr.choice(self.data, size=100000, replace=False)
        # print(npr.randint(1000))
        self.item = None
        self.setup_iterator()

    def setup_iterator(self):
        if GZIP:
            self.generator = iter(self.batch_and_pickle(iter(torch_geometric.data.DataLoader(self, batch_size=3*128)), batch_size=1))
        else:
            self.generator = iter(torch_geometric.data.DataLoader(self, batch_size=3*128))

    def batch_and_pickle(self, generator, batch_size=128):
        import pickle, gzip, io
        buffer = []
        for x in iter(generator):
            buffer.append(x)
            if len(buffer) == batch_size:
                with io.BytesIO() as io_buffer:
                    with gzip.open(io_buffer, "wb", compresslevel=1) as gzip_obj:
                        pickle.dump(buffer, gzip_obj)
                    io_buffer.seek(0)
                    yield io_buffer
                buffer = []

    def generate(self):
        try:
            self.item = next(self.generator)
            return self.item
        except StopIteration as stop:
            # print("STOP ITERATION", self)
            self.item = None
            return self.item

    def seed(self, i):
        seed(i * 12345)
        npr.seed(i * 12345)
        self.setup_iterator()
