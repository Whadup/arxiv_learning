"""Split Expressions into LHS and RHS of equalities and inequalities"""
from copy import deepcopy
import numpy as np
import torch
from random import randint
import ray
import xml.etree.ElementTree as ET
import arxiv_learning.data.heuristics.heuristic
import arxiv_learning.data.load_mathml as load_mathml
from arxiv_learning.data.heuristics.context import sample_equation, load_json
# def group_sections(x):
#     parts = x.split("/")
#     if len(parts)<3:
#         return parts[1]
#     f = "_".join(parts[2].split("_")[:-1])
#     return parts[1] + "/" + f

NAMESPACE = {"mathml":"http://www.w3.org/1998/Math/MathML"}
ET.register_namespace("", NAMESPACE["mathml"])
NEW_ROW = "mathml:math/mathml:semantics/mathml:mtable/mathml:mtr[1]/mathml:mtd[1]/mathml:mstyle/mathml:mrow"
FIRST_ROW = "mathml:math/mathml:semantics/mathml:mtable/mathml:mtr[1]"
FIRST_COL = "mathml:math/mathml:semantics/mathml:mtable/mathml:mtr[1]/mathml:mtd[1]"

def subtree(obj, current):
    newroot = deepcopy(obj)
    first_row = obj.find(FIRST_ROW, namespaces=NAMESPACE)
    first_col = obj.find(FIRST_COL, namespaces=NAMESPACE)

    # Remove all but the first colum from the first row
    first_row.clear()
    first_row.append(first_col)
    # Remove all buut the first row
    newroot.find("mathml:math/mathml:semantics/mathml:mtable",
        namespaces=NAMESPACE).clear()
    newroot.find("mathml:math/mathml:semantics/mathml:mtable",
        namespaces=NAMESPACE).append(first_row)
    # into the first colum of the first row, put all the signs
    newroot.find(NEW_ROW,
        namespaces=NAMESPACE).clear()
    newroot.find(NEW_ROW,
        namespaces=NAMESPACE).extend(current)
    return ET.ElementTree(element=newroot)

def iterate_table(obj):
    ROWS = "mathml:math/mathml:semantics/mathml:mtable/mathml:mtr"
    COLS = "mathml:mtd/mathml:mstyle/mathml:mrow"
    for row in obj.findall(ROWS, namespaces=NAMESPACE):
        for col in row.findall(COLS, namespaces=NAMESPACE):
            for elem in col:
                yield elem

def split(string=None, fail=True):
    obj = ET.fromstring(string)
    operators = "=≤≥<>"
    splits = []
    #multiline split
    MULTILINE_SPLIT = "mathml:math/mathml:semantics/mathml:mtable/mathml:mtr/mathml:mtd/mathml:mstyle/mathml:mrow/mathml:mo[.='{}']"
    # print(obj)
    # print(obj.findall("mathml:math/mathml:semantics", namespaces=NAMESPACE))
    if obj.find(NEW_ROW, namespaces=NAMESPACE) is None:
        if fail:
            return None
        return [string]
    for operator in operators:
        splits.extend(
            obj.findall(MULTILINE_SPLIT.format(operator),
                namespaces=NAMESPACE)
        )

    if not splits and fail:
        return
    elif not splits:
        return [string]
    # split_layer = obj.find("mathml:math/mathml:semantics/mathml:mrow",
    #     namespaces=NAMESPACE)
    # if split_layer is None:
    #     return None
    current = []
    count = 1
    results = []
    for elem in iterate_table(obj):
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
    if not fail or len(results) > 1:
        return results
    return None

@ray.remote
class EqualityHeuristic(arxiv_learning.data.heuristics.heuristic.Heuristic, torch.utils.data.IterableDataset):
    def __init__(self, test=False):
        super().__init__(test=test)

    def __iter__(self):
        while True:
            i = np.random.choice(len(self.data))
            paper = load_json(self.archive, self.data[i])
            if paper is None:
                # print("continue1")

                continue
            pair = sample_equation(paper, size=2)
            if pair is None:
                # del self.papers[i]
                # print("continue2")
                continue
            eq, other_eq = pair
            try:
                parts = split(string=eq)
                if parts is None:

                    # del eqs[j]
                    # print("continue3")

                    continue
                lengths = list([len(part) for part in parts])
                normalize = 1.0 * sum(lengths)
                ratios = list([l / normalize for l in lengths])
                deviations = list([min(len(ratios) * r, 1.0/(r * len(ratios))) for r in ratios])
                parts = [part for part, dev in zip(parts, deviations) if dev > 0.25]
                # print(deviations)
                # asdfa
                z = split(string=other_eq, fail=False)
                if z is None:
                    # del eqs[other_eq]
                    continue
                #filter parts that are too small
                part_a, part_b = np.random.choice(parts, size=2)
                part_c = np.random.choice(z)
                try:
                    x = load_mathml.load_pytorch(part_a, self.alphabet)
                    y = load_mathml.load_pytorch(part_b, self.alphabet)
                    z = load_mathml.load_pytorch(part_c, self.alphabet)
                    x.y = torch.LongTensor([[0]])
                    y.y = torch.LongTensor([[0]])
                    z.y = torch.LongTensor([[1]])
                    yield x
                    yield y
                    yield z
                    # self.item = (x,y,z)
                    # return self.item
                except Exception as identifier:
                    pass
                    # print(type(identifier), identifier)
                    # raise identifier
            except Exception as identifier:
                pass
                # print(type(identifier), identifier)
                # print(eq)
                # print(other_eq)
                # raise identifier
