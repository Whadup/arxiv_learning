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
# multiline split
MULTILINE_SPLIT = "mathml:math/mathml:semantics/mathml:mtable/mathml:mtr/mathml:mtd/mathml:mstyle/mathml:mrow/mathml:mo[.='{}']"
NEW_ROW = "mathml:math/mathml:semantics/mathml:mtable/mathml:mtr[1]/mathml:mtd[1]/mathml:mstyle/mathml:mrow"
FIRST_ROW = "mathml:math/mathml:semantics/mathml:mtable/mathml:mtr[1]"
FIRST_COL = "mathml:math/mathml:semantics/mathml:mtable/mathml:mtr[1]/mathml:mtd[1]"
OPERATORS = "=≤≥<>"
MAIN_ROW = "mathml:math/mathml:semantics/mathml:mrow"
MROW_TEMPLATE = ET.fromstring("<?xml version=\"1.0\" ?><span><math xmlns=\"http://www.w3.org/1998/Math/MathML\">" + "<semantics><mrow></mrow></semantics></math></span>")
CONTENT = "mathml:math//*.='e'"
match_operator = lambda e: e.text in OPERATORS if e.text else False

MO_NODE = "mathml:math//mathml:mo"
MFRAC_NODE = "mathml:math//mathml:mfrac"
MSQRT_NODE = "mathml:math//mathml:msqrt"
MROOT_NODE = "mathml:math//mathml:mroot"
USEFUL_NODES = [MO_NODE, MFRAC_NODE, MSQRT_NODE, MROOT_NODE]
ALL_NODES = "mathml:math//*"

# Two eqs should not differ to much in their lenght eq1 should be at most 1.5 times longer than eq2 and vice_versa
SIZE_FACTOR = 0.3
FACTOR_MAX = 1 + SIZE_FACTOR
FACTOR_MIN = 1 - SIZE_FACTOR

count_nodes = lambda e: len(e.findall(ALL_NODES, NAMESPACE))


def is_useful_subeq(eq):
    for node in USEFUL_NODES:
        if eq.find(node, NAMESPACE):
            return True
    return False


def filter_useful(results):
    return list(filter(is_useful_subeq, results))


def filter_size(results):
    results = sorted(results, key=count_nodes)
    median_index = int(len(results)/2)
    if median_index < 1:
        return None
    median = count_nodes(results[median_index])
    upper_bound = median * FACTOR_MAX
    lower_bound = median * FACTOR_MIN
    results = list(filter(lambda e: lower_bound <= count_nodes(e) <= upper_bound, results))
    return results


def construct_tree(elements):
    root = deepcopy(MROW_TEMPLATE)
    main_row = root.find(MAIN_ROW, NAMESPACE)
    for i, elem in enumerate(elements):
        main_row.insert(i, elem)
    return root


def split_single(string=None, fail=True):
    tree = ET.fromstring(string)
    results = []
    main_row = tree.find(MAIN_ROW, NAMESPACE)
    if main_row:
        # split the main_row on nodes that contain symbols like =,<,> etc.
        results = itertools.groupby(main_row, match_operator)
        # groupby returns a tuple. Index 0 reports whether the object at index 1 was matched
        # by the lambda given to groupby.
        results = [construct_tree(split[1]) for split in results if not split[0]]
    return results

def subtree(obj, current):
    newroot = deepcopy(obj)
    first_row = newroot.find(FIRST_ROW, namespaces=NAMESPACE)
    first_col = newroot.find(FIRST_COL, namespaces=NAMESPACE)

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

    if obj.find(NEW_ROW, namespaces=NAMESPACE) is None:
        return None if fail else [string]

    splits = []
    for operator in OPERATORS:
        splits.extend(
            obj.findall(MULTILINE_SPLIT.format(operator),
                namespaces=NAMESPACE)
        )

    if not splits:
        return None if fail else [string]

    current = []
    count = 1
    results = []
    for elem in iterate_table(obj):
        if elem in splits:
            newroot = subtree(obj, current)
            results.append(newroot.getroot())
            count += 1
            current = []
        else:
            current.append(elem)
    newroot = subtree(obj, current)
    results.append(newroot.getroot())

    return results


def split(string=None, fail=True):
    split_strategies = [split_multiline, split_single]
    for strategy in split_strategies:
        results = strategy(string=string, fail=fail)
        if results and len(results) > 1:
            break

    results = filter_useful(results)
    results = filter_size(results)
    if results is None or len(results) < 1:
        return None if fail else [string]
    else:
        return [ET.tostring(result, encoding="unicode") for result in results]


#@ray.remote
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
                if parts is None or len(parts) < 2:
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
                part_a, part_b = np.random.choice(parts, size=2, replace=False)
                part_c = np.random.choice(z)
                try:
                    yield part_a, part_b, part_c
                    #x = load_mathml.load_pytorch(part_a, self.alphabet)
                    #y = load_mathml.load_pytorch(part_b, self.alphabet)
                    #z = load_mathml.load_pytorch(part_c, self.alphabet)
                    #yield x
                    #yield y
                    #yield z
                    # self.item = (x,y,z)
                    # return self.item
                except Exception as identifier:
                    print(type(identifier), identifier)
                    # raise identifier
            except Exception as identifier:
                print(type(identifier), identifier)
                # print(eq)
                # print(other_eq)
                # raise identifier
