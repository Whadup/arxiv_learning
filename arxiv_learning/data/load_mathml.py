"""loads mathml xmls and returns a tree structure or a pytorch_geometric graph object"""
import xml.etree.ElementTree as ET
import pickle
import torch

EXAMPLE = "/data/d1/pfahler/arxiv_processed/subset_ml/train/mathml/\
1901.02705/Model-Predictive Policy Learning with Uncertainty Regularization_9.kmathml"

CONTENT_SYMBOLS = 192
ATTRIBUTE_SYMBOLS = 32
TAG_SYMBOLS = 32

DIM = CONTENT_SYMBOLS + ATTRIBUTE_SYMBOLS + TAG_SYMBOLS
MAX_POS = 256
def rn(x):
    """Remove Namespace"""
    return x.replace(r"{http://www.w3.org/1998/Math/MathML}", "")

def process(d):
    """Convert XML-Structure to Python dict-based representation"""
    children = [x for x in d]
    if children:
        children = []
        for x in d:
            tag = rn(x.tag)
            if tag == "annotation":
                #skip the latex annotation
                continue
            children.append(process(x))
        return dict(type=rn(d.tag),
                    children=children,
                    content="" if d.text is None else d.text)
    return dict(type=rn(d.tag),
                content="" if d.text is None else d.text,
                attributes=["=".join(y)for y in d.attrib.items()])

def load_dict(path, string=None):
    """Load XML into Python dictionaries"""
    # obj = xmltodict.parse(open(path, "rb"))
    if string is not None:
        obj = ET.fromstring(string)
    else:
        obj = ET.parse(path).getroot()
    #clean that shit up
    # print(obj)
    obj = obj[0][0][0]
    # print(obj.tag)
    # del obj['annotation']
    return process(obj)

def generate_skeleton(tree, edges=None, start=0):
    """
    Generates the edges of the given tree (represented in python dicts)
    Returns the number of nodes in the tree as well as the list of edges
    """
    if edges is None:
        e = []
        return generate_skeleton(tree, edges=e, start=0), e
    tree["_id"] = start
    newstart = start + 1
    if "children" in tree:
        edges.append((start, start, 0))
        for i, child in enumerate(tree["children"]):
            edges.append((start, newstart, 1))
            edges.append((newstart, start, min(i+2, MAX_POS - 1)))
            newstart = generate_skeleton(child, edges=edges, start=newstart)
    return newstart

def fill_skeleton(tree, X, alphabet):
    """Convert a tree and fills a torch tensor with features derived from the tree"""
    content, attributes, types = alphabet
    i = tree["_id"]
    if tree["type"] in types:
        X[i, types[tree["type"]]] = 1.0
    else:
        X[i, TAG_SYMBOLS-1] = 1.0
    # print(tree.keys())
    if "content" in tree:
        for char in tree["content"]:
            if char in content:
                X[i, TAG_SYMBOLS + content[char]] += 1
            else:
                X[i, TAG_SYMBOLS + CONTENT_SYMBOLS -1] += 1
    if "attributes" in tree:
        for attrib in tree["attributes"]:
            if attrib in attributes:
                X[i, TAG_SYMBOLS + CONTENT_SYMBOLS + attributes[attrib]] += 1
            else:
                X[i, DIM -1] += 1
    if "children" in tree:
        for child in tree["children"]:
            fill_skeleton(child, X, alphabet)

def load_pytorch(path, alphabet, string=None):
    """
    Loads a XML file and returns a pytorch representation consisting
    of node features 'X', edges 'E' and edge-features 'edge_features'.
    The alphabet dictionaries for content, node-types and attributes
    have to be supplied.
    """
    from torch_geometric.data import Data
    tree = load_dict(path, string)
    # print(json.dumps(tree, indent=4))
    num_nodes, e = generate_skeleton(tree)
    X = torch.zeros((num_nodes, DIM), dtype=torch.float32)
    fill_skeleton(tree, X, alphabet)
    edge_features = torch.zeros((len(e), 1), dtype=torch.int64)
    edges = torch.zeros((2, len(e)), dtype=torch.int64)
    for k, (i, j, y) in enumerate(e):
        edges[0, k] = i
        edges[1, k] = j
        edge_features[k, 0] = y
    # #self loops
    # for k in range(num_nodes):
    # 	edges[0, k + len(e)] = k
    # 	edges[1, k + len(e)] = k
    # 	edge_features[k + len(e), 0] = 0

    return Data(x=X, edge_index=edges, edge_attr=edge_features)
    #flatten into nodes, and edges

def update_alphabet(obj, content, attributes, types):
    """Grow the alphabet by the symbols in the tree 'obj'"""
    if "type" in obj:
        if obj["type"] not in types:
            types[obj["type"]] = 0
        types[obj["type"]] += 1
    if "content" in obj:
        for char in obj["content"]:
            if char not in content:
                content[char] = 0
            content[char] += 1
    if "attributes" in obj:
        for attrib in obj["attributes"]:
            if attrib.endswith("em"):
                #parse and round
                attrib = attrib.split("=")
                attrib[-1] = str(round(float(attrib[-1][:-2]), 1)) + "em"
                attrib = "=".join(attrib)
            if attrib not in attributes:
                attributes[attrib] = 0
            attributes[attrib] += 1
    if "children" in obj:
        for child in obj["children"]:
            update_alphabet(child, content, attributes, types)


def build_alphabet(path):
    """Build the alphabet based on all '*.kmathml' files in @path"""
    import os
    content = {}
    attributes = {}
    types = {}
    for p in os.listdir(path):
        if not os.path.isdir(os.path.join(path, p)):
            continue
        for f in os.listdir(os.path.join(path, p)):
            if f.endswith(".kmathml"):
                try:
                    obj = load_dict(os.path.join(path, p, f))
                    update_alphabet(obj, content, attributes, types)
                except Exception as e:
                    print(e)
    print(sorted(content.items(), key=lambda kv: kv[1]))
    print()
    print(sorted(attributes.items(), key=lambda kv: kv[1]))
    print()
    print(sorted(types.items(), key=lambda kv: kv[1]))
    return content, attributes, types

def load_alphabet(path=None, content=None, attributes=None, types=None):
    """Loads and prunes an alphabet either from pickle (@path) or from count dictionaries."""
    if path is not None:
        content, attributes, types = pickle.load(open(path, "rb"))
    content = {x:i for i, (x, y) in enumerate(sorted(content.items(), key=lambda kv: kv[1])[-(CONTENT_SYMBOLS-1):])}
    attributes = {x:i for i, (x, y) in enumerate(sorted(attributes.items(), key=lambda kv: kv[1])[-(ATTRIBUTE_SYMBOLS-1):])}
    types = {x:i for i, (x, y) in enumerate(sorted(types.items(), key=lambda kv: kv[1])[-(TAG_SYMBOLS-1):])}
    return content, attributes, types

if __name__ == "__main__":
    import sys, os
    PATH = sys.argv[1]
    pickle.dump(
        build_alphabet(os.path.join(PATH, "mathml/")),
        open(os.path.join(PATH, "alphabet.pickle"), "wb")
    )

