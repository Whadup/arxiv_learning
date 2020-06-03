import argparse
import os
import json
from tqdm import tqdm

import data.load_mathml

FLUSH_THRESHOLD = 1000


def all_eqs(paper_dict):
    result = []
    for section in paper_dict["sections"]:
        sec_eqs = section["equations"]
        for eq in sec_eqs:
            if eq.get('mathml'):
                result.append(eq['mathml'])
    return result


def flush_cache(cache):
    with open(args.out_path, 'a') as f:
        for eq in cache:
            f.write(eq + "\n")


def mathml_to_string(mathml):
    preorder_str = ""
    tree_dict = data.load_mathml.load_dict(None, mathml)
    repr_string, without_attr = data.load_mathml.build_representation_string(tree_dict)
    preorder_str += " " + without_attr
    if "children" in tree_dict:
        for child in tree_dict["children"]:
            preorder_str = _tree_to_string(child, preorder_str)
    return preorder_str


def _tree_to_string(tree_dict, preorder_str):
    repr_string, without_attr = data.load_mathml.build_representation_string(tree_dict)
    preorder_str += " " + without_attr
    if "children" in tree_dict:
        for child in tree_dict["children"]:
            preorder_str = _tree_to_string(child, preorder_str)
    return preorder_str


def main():
    json_files = [os.path.join(args.json_path, filename) for filename in os.listdir(args.json_path)]

    cache = []

    for json_file in tqdm(json_files[:500]):
        with open(json_file) as f:
            try:
                paper_dict = json.load(f)
                eqs_xml = all_eqs(paper_dict)
                eqs_string = [mathml_to_string(eq_xml) for eq_xml in eqs_xml]
                eqs_string = [item for item in eqs_string if item]
                cache += eqs_string
            except json.decoder.JSONDecodeError:
                pass
        if len(cache) > FLUSH_THRESHOLD:
            print("Flushing...")
            flush_cache(cache)
            cache = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This job creates a txt file with all equations from the arXiv-dataset."
                                                 "It will be a big file...")
    parser.add_argument("json_path", help="Path to the directory with the json files that store the eqs.")
    parser.add_argument("out_path", help="Path to file in that you want to store the output.")
    args = parser.parse_args()
    main()