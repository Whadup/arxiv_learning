import ray
import json
import os
import torch
import numpy as np
from random import shuffle, seed, randint
from copy import deepcopy
import os.path as osp
import arxiv_learning.data.heuristics.heuristic
import arxiv_learning.data.load_mathml as load_mathml

SAME_SECTION = 1
SAME_PAPER = 2
ALONG_CITATION = 3
DIFFERENT_PAPER = 4

@ray.remote
class JsonDataset(arxiv_learning.data.heuristics.heuristic.Heuristic, torch.utils.data.IterableDataset):
    """
    For legacy reasons we also allow streaming predefined json datasets.
    Do not use Blowout other than 5 as it will replicate the data weirdly
    """
    def __init__(self, test=False):
        super().__init__(test=test)
        self.basedir = "/data/s1/pfahler/arxiv_processed/subset_ml/train/pickle"
        if test:
            self.basedir = self.basedir.replace("train", "test")
        self.files = os.listdir(self.basedir)
        self.files = sorted([x for x in self.files if x.endswith(".json")])
        self.workload = self.files
        
    def seed(self, i):
        if i>9:
            raise ValueError("BLOWOUT LARGER THAN 10 for {}".format(type(self)))
        self.workload = self.files[i::10]
        self.setup_iterator()

    def __iter__(self):
        # print("THRESHOLD", self.threshold)
        # print(self.workload)
        # shuffle(self.workload)
        for batch in self.workload:
            data = json.load(open(os.path.join(self.basedir, batch), "r"))
            for triple in data:
                y1 = triple["first_action"]
                y2 = triple["second_action"]
                try:
                    in1 = osp.join("mathml", triple["paper_a"], triple["equation_a"])
                    in2 = osp.join("mathml", triple["paper_b"], triple["equation_b"])
                    in3 = osp.join("mathml", triple["paper_c"], triple["equation_c"])
                    in1 = self.archive.open(in1).read()
                    in2 = self.archive.open(in2).read()
                    in3 = self.archive.open(in3).read()
                    in1 = load_mathml.load_pytorch(None, self.alphabet, string=in1)
                    in2 = load_mathml.load_pytorch(None, self.alphabet, string=in2)
                    in3 = load_mathml.load_pytorch(None, self.alphabet, string=in3)
                except Exception as exp:
                    print(exp)
                    continue
                margin = 0
                if y2 == DIFFERENT_PAPER:
                    margin = 1.0
                elif y2 == ALONG_CITATION:
                    margin = 0.5
                elif y2 == SAME_PAPER:
                    margin = 0.25
                if y1 == ALONG_CITATION:
                    margin -= 0.5
                elif y1 == SAME_PAPER:
                    margin -= 0.25
                if margin < 0:
                    continue
                margin = 1
                anchor_swap = 1.0
                if y1 == ALONG_CITATION:
                    anchor_swap = 0.0
                yield in1
                yield in2
                yield in3
