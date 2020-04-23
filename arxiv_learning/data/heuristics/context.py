import ray
import os
import torch
import numpy as np
from random import shuffle, seed, randint
from copy import deepcopy
import arxiv_learning.data.heuristics.heuristic
import arxiv_learning.data.load_mathml as load_mathml

def load_xml(archive, alphabet, file):
    xml = archive.open(file, "r").read()
    if not xml:
        raise FileNotFoundError(file)
    return load_mathml.load_pytorch(None, alphabet, string=xml)

@ray.remote
class SamePaper(arxiv_learning.data.heuristics.heuristic.Heuristic, torch.utils.data.IterableDataset):
    def __init__(self, test=False):
        super().__init__(test=test)
        from itertools import groupby
        self.papers = []
        self.paper_ids = []
        for p, e in groupby(self.data, key=lambda x:x.split("/")[1]):
            self.papers.append(list([x for x in e if x.endswith("kmathml")]))
            self.paper_ids.append(p)
        # self.generator = iter(self.batch_and_pickle())

    def __iter__(self):
        while True:
            i = np.random.choice(len(self.papers))
            eqs = self.papers[i]
            if len(eqs)<2:
                # del self.papers[i]
                continue
            other_paper = (i + randint(0,len(self.papers)-1)) % len(self.papers)
            neg_eqs = self.papers[other_paper]
            if not neg_eqs:
                # del self.papers[other_paper]
                continue
            x, y = np.random.choice(eqs, size=2)
            z = np.random.choice(neg_eqs)
            try:
                x = load_xml(self.archive, self.alphabet, x)
                y = load_xml(self.archive, self.alphabet, y)
                z = load_xml(self.archive, self.alphabet, z)
                yield x
                yield y
                yield z
                # self.item = (x,y,z)
                # return self.item
            except Exception as identifier:
                print(type(identifier), identifier)
                pass

@ray.remote
class SameSection(arxiv_learning.data.heuristics.heuristic.Heuristic, torch.utils.data.IterableDataset):
    def __init__(self, test=False):
        from arxiv_learning.data.heuristics.equations import group_sections
        from itertools import groupby
        super().__init__(test=test)
        self.papers = []
        self.paper_ids = []
        for p, e in groupby(self.data, key=group_sections):
            self.papers.append(list([x for x in e if x.endswith("kmathml")]))
            self.paper_ids.append(p)
        # self.generator = iter(self.batch_and_pickle())

    #EXACT COPY FROM ABOVE!
    def __iter__(self):
        while True:
            i = np.random.choice(len(self.papers))
            eqs = self.papers[i]
            if len(eqs)<2:
                # del self.papers[i]
                continue
            other_paper = (i + randint(0,len(self.papers)-1)) % len(self.papers)
            neg_eqs = self.papers[other_paper]
            if not neg_eqs:
                # del self.papers[other_paper]
                continue
            x, y = np.random.choice(eqs, size=2)
            z = np.random.choice(neg_eqs)
            try:
                x = load_xml(self.archive, self.alphabet, x)
                y = load_xml(self.archive, self.alphabet, y)
                z = load_xml(self.archive, self.alphabet, z)
                yield x
                yield y
                yield z
                # self.item = (x,y,z)
                # return self.item
            except Exception as identifier:
                print(type(identifier), identifier)
                pass

