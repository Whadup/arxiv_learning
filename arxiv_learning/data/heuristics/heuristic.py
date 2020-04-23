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
    def __init__(self, test=False):
        self.basefile = "/data/s1/pfahler/arxiv_processed/subset_ml/train/mathml.zip"
        self.alphabet = load_mathml.load_alphabet(
            os.path.join(os.path.split(self.basefile)[0], "vocab.pickle"))
        if test and "train" in self.basefile:
            self.basefile = self.basefile.replace("train", "test")
        if MEMORY_BUFFERED:
            buffer = io.BytesIO(open(self.basefile, "rb").read())
            self.archive = zipfile.ZipFile(buffer, "r")
        else:
            self.archive = zipfile.ZipFile(self.basefile, "r")
        self.data = self.archive.namelist()
        if (not test and "train" not in self.basefile) or (test and "test" not in self.basefile):
            self.data = sorted(self.data)
            cutoff = len(self.data) // 5
            if test:
                self.data = self.data[-cutoff:]
            else:
                self.data = self.data[:-cutoff]
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