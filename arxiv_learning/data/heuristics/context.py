import ray
import os
import json
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

def load_json(archive, file):
    try:
        return json.load(archive.open(file, "r"))
    except json.decoder.JSONDecodeError as e:
        return None

def sample_equation(paper, size=None):
    all_eqs = sum([
        [eq["mathml"] for eq in section["equations"] if "mathml" in eq] for section in paper["sections"]]
        , [])
    if all_eqs:
        try:
            return np.random.choice(all_eqs, size=size, replace=False)
        except:
            return None
    return None

@ray.remote
class SamePaper(arxiv_learning.data.heuristics.heuristic.Heuristic, torch.utils.data.IterableDataset):
    def __init__(self, test=False):
        super().__init__(test=test)


    def __iter__(self):
        while True:
            i = np.random.choice(len(self.data))
            paper = load_json(self.archive, self.data[i])
            if paper is None:
                continue
            pair = sample_equation(paper, size=2)
            if pair is None:
                # del self.papers[i]
                continue
            x, y = pair
            # print(x)
            j = (i + randint(0, len(self.data)-1)) % len(self.data)
            other_paper = load_json(self.archive, self.data[j])
            if other_paper is None:
                continue
            z = sample_equation(other_paper)
            if z is None:
                continue
            # print(z)
            x = load_mathml.load_pytorch(x, self.alphabet)
            y = load_mathml.load_pytorch(y, self.alphabet)
            z = load_mathml.load_pytorch(z, self.alphabet)
            yield x
            yield y
            yield z



@ray.remote
class SameSection(arxiv_learning.data.heuristics.heuristic.Heuristic, torch.utils.data.IterableDataset):
    #EXACT COPY FROM ABOVE!
    def __iter__(self):
        while True:
            i = np.random.choice(len(self.data))
            paper = load_json(self.archive, self.data[i])
            if paper is None:
                continue
            try:
                section = np.random.choice(paper["sections"])
            except Exception as e:
                print(type(e))
                continue
            equations = [eq["mathml"] for eq in section["equations"] if "mathml" in eq]
            try:
                pair = np.random.choice(equations, size=2, replace=False)
            except:
                continue
            x, y = pair
            # print(x)
            j = (i + randint(0, len(self.data)-1)) % len(self.data)
            other_paper = load_json(self.archive, self.data[j])
            if other_paper is None:
                continue
            z = sample_equation(other_paper)
            if z is None:
                continue
            # print(z)
            x = load_mathml.load_pytorch(x, self.alphabet)
            y = load_mathml.load_pytorch(y, self.alphabet)
            z = load_mathml.load_pytorch(z, self.alphabet)
            yield x
            yield y
            yield z

