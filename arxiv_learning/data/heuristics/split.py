import xml.etree.ElementTree as ET
import itertools
from copy import deepcopy

mrow_template = ET.fromstring("<?xml version=\"1.0\" ?><span><math xmlns=\"http://www.w3.org/1998/Math/MathML\">" +
                              "<semantics><mrow></mrow></semantics></math></span>")
NAMESPACE = {"mathml": "http://www.w3.org/1998/Math/MathML"}
ET.register_namespace("", NAMESPACE["mathml"])
xpath_main_row = "mathml:math/mathml:semantics/mathml:mrow"
operators = "=≤≥<>"
match_operator = lambda e: e.text in operators if e.text else False


def construct_tree(elements):
    root = deepcopy(mrow_template)
    main_row = root.find(xpath_main_row, NAMESPACE)
    for i, elem in enumerate(elements):
        main_row.insert(i, elem)
    return root


def split_tree(tree):
    main_row = tree.find(xpath_main_row, NAMESPACE)
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
        return []
