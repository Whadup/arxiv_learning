import argparse
import os
import json
from tqdm import tqdm

FLUSH_THRESHOLD = 1000


def all_eqs(paper_dict):
    result = []
    for section in paper_dict["sections"]:
        sec_eqs = section["equations"]
        for eq in sec_eqs:
            if eq.get('mathml'):
                result.append(eq['mathml'])
    return result


def build_representation_string(obj):
    representation = ""
    if "type" in obj:
        representation+=obj["type"]+"_"
    if "content" in obj:
        for char in obj["content"]:
            representation += char
        representation += "_"
    without_attr = representation
    return without_attr


def flush_cache(cache):
    with open(args.out_path, 'a') as f:
        f.writelines(cache)


def main():
    json_files = [os.path.join(args.json_path, filename) for filename in os.listdir(args.json_path)]

    cache = []

    for json_file in tqdm(json_files):
        with open(json_file) as f:
            try:
                paper_dict = json.load(f)
                eqs_xml = all_eqs(paper_dict)
                eqs_string = [build_representation_string(eq_xml) for eq_xml in eqs_xml]
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