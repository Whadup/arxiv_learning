"""Split Expressions into LHS and RHS of equalities and inequalities"""
from copy import deepcopy
import numpy as np
import torch
from random import randint
import ray
import xml.etree.ElementTree as ET
import arxiv_learning.data.heuristics.heuristic
import arxiv_learning.data.load_mathml as load_mathml

def group_sections(x):
    parts = x.split("/")
    if len(parts)<3:
        return parts[1]
    f = "_".join(parts[2].split("_")[:-1])
    return parts[1] + "/" + f

NAMESPACE = {"mathml":"http://www.w3.org/1998/Math/MathML"}
ET.register_namespace("", NAMESPACE["mathml"])

def subtree(obj, current):
    newroot = deepcopy(obj)
    newroot.find("mathml:math/mathml:semantics/mathml:mrow",
        namespaces=NAMESPACE).clear()
    newroot.find("mathml:math/mathml:semantics/mathml:mrow",
        namespaces=NAMESPACE).extend(current)
    return ET.ElementTree(element=newroot)

def split(path=None, string=None, fail=True):
    if string is not None:
        obj = ET.fromstring(string)
    else:
        obj = ET.parse(path).getroot()
    operators = "=≤≥<>"
    splits = []
    # print(obj[0][0])
    for operator in operators:
        splits.extend(
            obj.findall("mathml:math/mathml:semantics/mathml:mrow/mathml:mo[.='{}']".format(operator),
                namespaces=NAMESPACE)
        )
    if not splits and fail:
        return
    split_layer = obj.find("mathml:math/mathml:semantics/mathml:mrow",
        namespaces=NAMESPACE)
    if split_layer is None:
        return None
    current = []
    count = 1
    results = []
    for elem in split_layer:
        if elem in splits:
            # print(current)
            newroot = subtree(obj, current)
            # newroot.write("tmp{}.mathml".format(count))
            results.append(ET.tostring(newroot.getroot(), encoding="unicode"))
            count += 1
            current = []
        else:
            current.append(elem)
    newroot = subtree(obj, current)
    # newroot.write("tmp{}.mathml".format(count))
    results.append(ET.tostring(newroot.getroot(), encoding="unicode"))
    if not fail or len(results)>1:
        return results
    return None

@ray.remote
class EqualityHeuristic(arxiv_learning.data.heuristics.heuristic.Heuristic, torch.utils.data.IterableDataset):
    def __init__(self, test=False):
        super().__init__(test=test)
        from itertools import groupby
        self.papers = []
        self.paper_ids = []
        for p, e in groupby(self.data, key=group_sections):
            self.papers.append(list([x for x in e if x.endswith("kmathml")]))
            self.paper_ids.append(p)
        # self.papers_all = deepcopy(self.papers)
        # self.generator = iter(self.batch_and_pickle())
        # self.generator = iter(self.batch_and_pickle(iter(torch_geometric.data.DataLoader(self, batch_size=3*128)), batch_size=1))

    def __iter__(self):
        while True:
            i = np.random.choice(len(self.papers))
            eqs = self.papers[i]
            if len(eqs)<2:
                continue
            j = np.random.choice(len(eqs))
            eq = eqs[j]
            try:
                parts = split(string=self.archive.open(eq, "r").read())
                if parts is None:
                    # del eqs[j]
                    continue
                other_eq = (j + randint(0, len(eqs)-1)) % len(eqs)
                z = self.archive.open(eqs[other_eq], "r").read()
                z = split(string=z, fail=False)
                if z is None:
                    # del eqs[other_eq]
                    continue
                part_a, part_b = np.random.choice(parts, size=2)
                part_c = np.random.choice(z)
                try:
                    x = load_mathml.load_pytorch(None, self.alphabet, string=part_a)
                    y = load_mathml.load_pytorch(None, self.alphabet, string=part_b)
                    z = load_mathml.load_pytorch(None, self.alphabet, string=part_c)
                    yield x
                    yield y
                    yield z
                    # self.item = (x,y,z)
                    # return self.item
                except Exception as identifier:
                    print(type(identifier), identifier)
                    raise identifier
            except Exception as identifier:
                print(type(identifier), identifier)
                raise identifier
