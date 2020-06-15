import numpy as np

import torch

import sys

from ml.graph_cnn import GraphCNN
from arxiv_lib.load_mathml import load_pytorch, load_alphabet
import json
import re
from annoy import AnnoyIndex
def eval(checkpoint):
    run="bow"
    net = None
    if checkpoint is not None:
        net = GraphCNN()
        net.load_state_dict_from_path(checkpoint)
        net = net.eval()
        run = re.search("models/([0-9]+)/", checkpoint).group(1)
        index = AnnoyIndex(64, "angular")
        index.load("/data/d1/pfahler/arxiv_processed/deep_{}.ann".format(run))
        keys = open("/data/d1/pfahler/arxiv_processed/all_keys_{}.csv".format(run), "r").read().split("\n")
    else:
        index = AnnoyIndex(256, "angular")
        index.load("/data/d1/pfahler/arxiv_processed/bow.ann")
        keys = open("/data/d1/pfahler/arxiv_processed/bow_keys.csv", "r").read().split("\n")

    alphabet = load_alphabet("alphabet_large.pickle")
    xml_data = json.load(open("example_queries.json", "r"))
    dataset = []
    for i, example in enumerate(xml_data):
        xml = example["query"]
        keywords = example["keywords"]
        query = load_pytorch(None, alphabet, string=xml)
        if net is None:
            x = query.x.sum(dim=0).cpu().numpy()
        else:
            x = net(query).cpu().detach().numpy()
        neighbors, distances = index.get_nns_by_vector(x, 1000, include_distances=True)
        results = [keys[i] for i in neighbors]
        dataset.append(dict(xml=xml, keywords=keywords, results=results, distances=["{:5f}".format(x) for x in distances]))
    return dataset, run

if __name__ == "__main__":
    data, run = eval(None if sys.argv[1] == "bow" else sys.argv[1])
    json.dump(data, open("search_results_{}.json".format(run), "w"), indent=4)
    print(json.dumps(data, indent=4))
