"""Split Expressions into LHS and RHS of equalities and inequalities"""
from copy import deepcopy
import itertools
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
OPERATORS = "=≤≥<>"
MAIN_ROW = "mathml:math/mathml:semantics/mathml:mrow"
MROW_TEMPLATE = ET.fromstring("<?xml version=\"1.0\" ?><span><math xmlns=\"http://www.w3.org/1998/Math/MathML\">" + "<semantics><mrow></mrow></semantics></math></span>")
match_operator = lambda e: e.text in OPERATORS if e.text else False


def construct_tree(elements):
    root = deepcopy(MROW_TEMPLATE)
    main_row = root.find(MAIN_ROW, NAMESPACE)
    for i, elem in enumerate(elements):
        main_row.insert(i, elem)
    return root


def split_single(string=None, fail=True):
    tree = ET.fromstring(string)
    main_row = tree.find(MAIN_ROW, NAMESPACE)
    if main_row:
        # split the main_row on nodes that contain symbols like =,<,> etc.
        subtrees = itertools.groupby(main_row, match_operator)
        # groupby returns a tuple. Index 0 reports whether the object at index 1 was matched
        # by the lambda given to groupby.
        subtrees = [construct_tree(split[1]) for split in subtrees if not split[0]]
        # if no operator is found on that we can split, the variable subtrees contains only the full tree
        # but this is useless. Therefore we return an empty list in that case.
        return subtrees if len(subtrees) > 1 else []
    else:
        return None if fail else [string]

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

def split_multiline(string=None, fail=True):
    obj = ET.fromstring(string)
    splits = []
    #multiline split
    MULTILINE_SPLIT = "mathml:math/mathml:semantics/mathml:mtable/mathml:mtr/mathml:mtd/mathml:mstyle/mathml:mrow/mathml:mo[.='{}']"
    # print(obj)
    # print(obj.findall("mathml:math/mathml:semantics", namespaces=NAMESPACE))
    if obj.find(NEW_ROW, namespaces=NAMESPACE) is None:
        if fail:
            return None
        return [string]
    for operator in OPERATORS:
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


def split(string=None, fail=True):
    split_strategies = [split_multiline, split_single]
    for strategy in split_strategies:
        parts = strategy(string=string, fail=fail)
        if parts:
            return parts
    return None


@ray.remote
class EqualityHeuristic(arxiv_learning.data.heuristics.heuristic.Heuristic, torch.utils.data.IterableDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_seed = None
    def __iter__(self):
        while True:
            i = np.random.choice(len(self.data))
            paper = load_json(self.archive, self.data[i])
            if paper is None:
                # print("continue1")

                continue
            eq = sample_equation(paper)
            if eq is None:
                # del self.papers[i]
                # print("continue2")
                continue
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
                # z = split(string=other_eq, fail=False)
                # if z is None:
                #     # del eqs[other_eq]
                #     continue
                #filter parts that are too small
                part_a, part_b = np.random.choice(parts, size=2, replace=False)
                # part_c = np.random.choice(z)
                try:
                    x = load_mathml.load_pytorch(part_a, self.alphabet)
                    y = load_mathml.load_pytorch(part_b, self.alphabet)
                    # z = load_mathml.load_pytorch(part_c, self.alphabet)
                    x.y = torch.LongTensor([[0]])
                    y.y = torch.LongTensor([[0]])
                    # z.y = torch.LongTensor([[1]])
                    yield x
                    yield y
                    # yield z
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

    def seed(self, i):
        from random import seed
        if self.custom_seed == None:
            self.custom_seed = i
        seed(self.custom_seed * 12345)
        np.random.seed(self.custom_seed * 12345)
        self.custom_seed = randint(0, 100000)
        self.setup_iterator()