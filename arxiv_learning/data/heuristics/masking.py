import numpy as np
import torch
import ray
import arxiv_learning.data.heuristics.heuristic
import arxiv_learning.data.load_mathml as load_mathml
from arxiv_learning.data.load_mathml import VOCAB_SYMBOLS
from arxiv_learning.data.heuristics.context import sample_equation, load_json
from arxiv_learning.data.augmentation import permute, prepare_permutations


@ray.remote
class MaskingHeuristic(arxiv_learning.data.heuristics.heuristic.Heuristic, torch.utils.data.IterableDataset):
    def __init__(self, data_augmentation=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.permute = False # only permute during training
        if data_augmentation and not self.test:
            self.permute = True
            self.perms = prepare_permutations(self.alphabet)
        self.custom_seed = None
    def __iter__(self):
        while True:
            try:
                i = np.random.choice(len(self.data))
                paper = load_json(self.archive, self.data[i])
                if paper is None:
                    continue
                eq = sample_equation(paper)
                if eq is None:
                    continue
                x = load_mathml.load_pytorch(eq, self.alphabet)
                mask = torch.rand((x.x.shape[0], 1)) < 0.15
                unk = torch.ones(1, dtype=torch.int64) * (VOCAB_SYMBOLS - 1)
                ign = torch.ones(1, dtype=torch.int64) * -100
                x.y = torch.where(mask, x.x, ign)
                x.x = torch.where(mask, unk, x.x)
                # print(x)
                yield x
            except GeneratorExit:
                return
            except:
                continue


    def seed(self, i):
        from random import seed, randint
        if self.custom_seed == None:
            self.custom_seed = i
        seed(self.custom_seed * 12345)
        np.random.seed(self.custom_seed * 12345)
        self.custom_seed = randint(0, 100000)
        self.setup_iterator()
