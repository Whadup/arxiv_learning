from annoy import AnnoyIndex
import torch
import gzip
import zipfile
import pickle
import json
import os
import re
import tqdm
import sys
from functools import partial
from  multiprocessing.pool import Pool
from torch_geometric.data import DataLoader
from arxiv_learning.nn.graph_cnn import GraphCNN
from arxiv_learning.data.load_mathml import load_pytorch, load_alphabet

def load_json(archive, file):
    try:
        return json.load(archive.open(file, "r"))
    except json.decoder.JSONDecodeError as e:
        return None

def main(checkpoint, data):
    run = re.search("checkpoints/([0-9]+)/", checkpoint).group(1)
    print(run)
    
    model = GraphCNN()
    model.load_state_dict_from_path(checkpoint)
    model = model.cuda().eval()
    
    alphabet = load_alphabet("vocab.pickle")

    data = zipfile.ZipFile(data, "r")

    # files = files[:1]

    index = AnnoyIndex(64, 'angular')
    index.on_disk_build("/data/s1/pfahler/arxiv_processed/deep_{}.ann".format(run))
    all_keys = []
    i = 0
    with tqdm.tqdm(total=len(data.namelist()), smoothing=0.1) as pbar:
        with Pool(20) as p:
            for f in data.namelist():
                pbar.update(1)
                if not f.endswith(".json"):
                    continue
                paper = load_json(data, f)
                if paper is None:
                    print("couldnt load", f)
                    continue
                mathmls = sum([
                    [eq["mathml"] for eq in section["equations"] if "mathml" in eq] for section in paper["sections"]],
                    []
                )
                nos = sum([
                    [eq["no"] for eq in section["equations"] if "mathml" in eq] for section in paper["sections"]],
                    []
                )

                X = p.map(partial(load_pytorch, alphabet=alphabet), mathmls)
                # Filter where loading failed
                keys = [(os.path.basename(f), no) for no, mml in zip(nos, X) if mml is not None]
                X = [mml for mml in X if mml is not None]

                all_keys += keys

                loader = DataLoader(X, batch_size=128, shuffle=False)
                with torch.no_grad():
                    for d in loader:
                        d = d.to("cuda")
                        batch = model.mean(d).detach().cpu().numpy()
                        for e in batch:
                            index.add_item(i, e)
                            i += 1
                pbar.set_description("#Embeddings: {}".format(i))

        print("Building Index")
        index.build(16)
        print("Saving Keys")
        with open("/data/s1/pfahler/arxiv_processed/ids_{}.pickle".format(run), "wb") as f:
            pickle.dump(all_keys, f)

if __name__ == "__main__":
    main(checkpoint=sys.argv[1], data=sys.argv[2])
    