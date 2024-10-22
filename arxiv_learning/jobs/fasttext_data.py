import argparse
import os
import json
import zipfile
from tqdm import tqdm

import arxiv_learning.data.load_mathml

FLUSH_THRESHOLD = 100000


def all_eqs(paper_dict):
    result = []
    for section in paper_dict["sections"]:
        sec_eqs = section["equations"]
        for eq in sec_eqs:
            if eq.get('mathml'):
                result.append(eq['mathml'])
    return result


def flush_cache(cache):
    with open(args.out_path + ("_test.txt" if args.test else "_train.txt"), 'a') as f:
        for eq in cache:
            f.write(eq + "\n")

def mathml_to_root_path(mathml):
    preorder_str = ""
    tree_dict = arxiv_learning.data.load_mathml.load_dict(None, mathml)
    repr_string, without_attr = arxiv_learning.data.load_mathml.build_representation_string(tree_dict)
    preorder_str += "{:02X}".format(hash(without_attr) % (1<<32))
    if "children" in tree_dict:
        for child in tree_dict["children"]:
            preorder_str = _tree_to_root_path(child, preorder_str, without_attr)
    return preorder_str

def _tree_to_root_path(tree_dict, preorder_str, root_path):
    repr_string, without_attr = arxiv_learning.data.load_mathml.build_representation_string(tree_dict)
    root_path = root_path +"/"+ without_attr
    preorder_str += " " + "{:02X}".format(hash(root_path) % (1<<32))
    if "children" in tree_dict:
        for child in tree_dict["children"]:
            preorder_str = _tree_to_root_path(child, preorder_str, without_attr)
    return preorder_str

def mathml_to_string(mathml):
    preorder_str = ""
    tree_dict = arxiv_learning.data.load_mathml.load_dict(None, mathml)
    repr_string, without_attr = arxiv_learning.data.load_mathml.build_representation_string(tree_dict)
    preorder_str += " " + without_attr
    if "children" in tree_dict:
        for child in tree_dict["children"]:
            preorder_str = _tree_to_string(child, preorder_str)
    return preorder_str


def _tree_to_string(tree_dict, preorder_str):
    repr_string, without_attr = arxiv_learning.data.load_mathml.build_representation_string(tree_dict)
    preorder_str += " " + without_attr
    if "children" in tree_dict:
        for child in tree_dict["children"]:
            preorder_str = _tree_to_string(child, preorder_str)
    return preorder_str


def main():
    # TODO: make this zip-compliant
    #json_files = [os.path.join(args.json_path, filename) for filename in os.listdir(args.json_path)]
    json_files = zipfile.ZipFile(args.json_path, "r")
    cache = []
    names = json_files.namelist()
    if args.test:
        test_papers = set(json.load(open("test_papers_meta.json", "r")).keys())
        names = [x for x in names if os.path.basename(x).replace(".json", "") in test_papers]
    else:
        train_papers = set(json.load(open("train_papers_meta.json", "r")).keys())
        names = [x for x in names if os.path.basename(x).replace(".json", "") in train_papers]
    for json_file in tqdm(names):
        # with open(json_file) as f:
        try:
            paper_dict = json.load(json_files.open(json_file, "r"))
            eqs_xml = all_eqs(paper_dict)
            if args.root_path:
                eqs_string = [mathml_to_root_path(eq_xml) for eq_xml in eqs_xml]
            else:
                eqs_string = [mathml_to_string(eq_xml) for eq_xml in eqs_xml]
            eqs_string = [item for item in eqs_string if item]
            cache += eqs_string
        except json.decoder.JSONDecodeError:
            pass
        except Exception as e:
            print(e)
        if len(cache) > FLUSH_THRESHOLD:
            print("Flushing...")
            flush_cache(cache)
            cache = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This job creates a txt file with all equations from the arXiv-dataset."
                                                 "It will be a big file...")
    parser.add_argument("--json_path", help="Path to the directory with the json files that store the eqs.")
    parser.add_argument("--out_path", help="Path to file in that you want to store the output.")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--root_path", action="store_true")
    args = parser.parse_args()
    main()
