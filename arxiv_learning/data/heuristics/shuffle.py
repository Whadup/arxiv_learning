"""Split Expressions into LHS and RHS of equalities and inequalities"""
from copy import deepcopy
import numpy as np
import torch
from random import randint, seed
import ray
import xml.etree.ElementTree as ET
import arxiv_learning.data.heuristics.heuristic
import arxiv_learning.data.load_mathml as load_mathml
from arxiv_learning.data.heuristics.context import sample_equation, load_json
from arxiv_learning.data.augmentation import break_order
@ray.remote
class ShuffleHeuristic(arxiv_learning.data.heuristics.heuristic.Heuristic, torch.utils.data.IterableDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_seed = None

    def __iter__(self):
        while True:
            try:
                i = np.random.choice(len(self.data))
                paper = load_json(self.archive, self.data[i])
                if paper is None:
                    # print("continue1")

                    continue
                eq = sample_equation(paper)
                x = load_mathml.load_pytorch(eq, self.alphabet)
                x.y = (np.random.rand() < 0.5) + 0
                if x.y:
                    x = break_order(x, proba=0.05)
                yield x
            except GeneratorExit:
                break
            except:
                pass
    def seed(self, i):
        if self.custom_seed == None:
            self.custom_seed = i
        seed(self.custom_seed * 12345)
        np.random.seed(self.custom_seed * 12345)
        self.custom_seed = randint(0, 100000)
        self.setup_iterator()