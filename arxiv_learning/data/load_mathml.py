
"""loads mathml xmls and returns a tree structure or a pytorch_geometric graph object"""
import xml.etree.ElementTree as ET
import pickle
import torch
import tqdm
EXAMPLE = "/data/s1/pfahler/arxiv_processed/subset_ml/train/mathml/\
1901.02705/Model-Predictive Policy Learning with Uncertainty Regularization_9.kmathml"

CONTENT_SYMBOLS = 192
ATTRIBUTE_SYMBOLS = 32
TAG_SYMBOLS = 32
VOCAB_SYMBOLS = 512

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
        # edges.append((start, start, 0))
        for i, child in enumerate(tree["children"]):
            edges.append((start, newstart, 1)) # topwdown
            edges.append((newstart, start, 0)) # bottomup
            newstart = generate_skeleton(child, edges=edges, start=newstart)
    return newstart

def fill_skeleton(tree, X, pos, alphabet):
    """Convert a tree and fills a torch tensor with features derived from the tree"""
    vocab = alphabet
    i = tree["_id"]
    
    representation_string, without_attr = build_representation_string(tree)
    if representation_string in vocab:
        X[i, 0] = vocab[representation_string]
    elif without_attr in vocab:
        X[i, 0] = vocab[without_attr]
    else:
        X[i, 0] = VOCAB_SYMBOLS - 1
    if "children" in tree:
        for i, child in enumerate(tree["children"]):
            pos[child["_id"]] = min(i, MAX_POS - 1)
            fill_skeleton(child, X, pos, alphabet)

def build_representation_string(obj):
    representation = ""
    if "type" in obj:
        representation+=obj["type"]+"_"
    if "content" in obj:
        for char in obj["content"]:
            representation += char
        representation += "_"
    without_attr = representation
    if "attributes" in obj:
        for attrib in sorted(obj["attributes"]):
            if attrib.endswith("em"):
                #parse and round
                attrib = attrib.split("=")
                attrib[-1] = str(round(float(attrib[-1][:-2]), 1)) + "em"
                attrib = "=".join(attrib)
            representation+=attrib+","
    return representation, without_attr

def load_pytorch(string, alphabet):
    """
    Loads a XML file and returns a pytorch representation consisting
    of node features 'X', edges 'E' and edge-features 'edge_features'.
    The alphabet dictionaries for content, node-types and attributes
    have to be supplied.
    """
    from torch_geometric.data import Data
    tree = load_dict(None, string)
    # print(json.dumps(tree, indent=4))
    num_nodes, e = generate_skeleton(tree)
    X = torch.zeros((num_nodes, 1), dtype=torch.int64)
    pos = torch.zeros((num_nodes, 1), dtype=torch.int64)
    fill_skeleton(tree, X, pos, alphabet)
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

    return Data(x=X, edge_index=edges, edge_attr=edge_features, pos=pos)
    # return Data(x=X, edge_index=edges, edge_attr=edge_features, pos=pos)
    #flatten into nodes, and edges

def update_alphabet(obj, vocab):
    """Grow the alphabet by the symbols in the tree 'obj'"""
    representation, _ = build_representation_string(obj)
    if representation not in vocab:
        vocab[representation] = 0
    vocab[representation] += 1
    if "children" in obj:
        for child in obj["children"]:
            update_alphabet(child, vocab)


def build_alphabet(path):
    """Build the alphabet based on all '*.kmathml' files in @path"""
    import os, zipfile
    import json
    vocab = {}
    archive = zipfile.ZipFile(path, "r")
    data = archive.namelist()
    for p in tqdm.tqdm(data):
        if not p.endswith(".json"):
            continue
        try:
            paper = json.load(archive.open(p, "r"))
            all_eqs = sum([
                [eq["mathml"] for eq in section["equations"] if "mathml" in eq] for section in paper["sections"]]
                , [])
        except json.decoder.JSONDecodeError as e:
            continue
        except:
            continue
        for eq in all_eqs:
            try:
                obj = load_dict(None, string=eq)
                update_alphabet(obj, vocab)
            except Exception as e:
                print(e)
    print(sorted(vocab.items(), key=lambda kv: kv[1]))
    print()
    return vocab

def load_alphabet(path=None, vocab=None):
    """Loads and prunes an alphabet either from pickle (@path) or from count dictionaries."""
    if path is not None:
        vocab = pickle.load(open(path, "rb"))
    vocab = {x:i for i, (x, y) in enumerate(sorted(vocab.items(), key=lambda kv: kv[1])[-(VOCAB_SYMBOLS-1):])}
    # attributes = {x:i for i, (x, y) in enumerate(sorted(attributes.items(), key=lambda kv: kv[1])[-(ATTRIBUTE_SYMBOLS-1):])}
    # types = {x:i for i, (x, y) in enumerate(sorted(types.items(), key=lambda kv: kv[1])[-(TAG_SYMBOLS-1):])}
    return vocab

if __name__ == "__main__":
    import sys
    import os
    PATH = sys.argv[1]
    pickle.dump(
        build_alphabet(PATH),
        open(os.path.join(os.path.dirname(PATH), "vocab.pickle"), "wb")
    )
