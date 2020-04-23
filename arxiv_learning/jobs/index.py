from annoy import AnnoyIndex
import torch
import gzip
import pickle
import os
import re
import tqdm
import sys
from torch_geometric.data import DataLoader
import ml.graph_cnn
def main(checkpoint):
    run = re.search("models/([0-9]+)/", checkpoint).group(1)
    print(run)
    
    model = ml.graph_cnn.GraphCNN()
    model.load_state_dict_from_path(checkpoint)
    model = model.cuda().eval()
    files = [x for x in os.listdir("/data/d1/pfahler/arxiv_processed") if x.startswith("train_mathmls")]

    # files = files[:1]

    index = AnnoyIndex(64, 'angular')
    index.on_disk_build("/data/d1/pfahler/arxiv_processed/deep_{}.ann".format(run))
    all_keys = []
    i = 0
    for f in files:
        num = re.match("train_mathmls([0-9]+).pickle.gz", f).group(1)
        keys = open(os.path.join("/data/d1/pfahler/arxiv_processed",
                                 "train_keys_{}.csv".format(num)), "r").read().split("\n")
        all_keys += keys
        print("loading", f)
        X = pickle.load(gzip.open(os.path.join("/data/d1/pfahler/arxiv_processed", f), "rb"))
        loader = DataLoader(X, batch_size=128)
        emb = []
        with torch.no_grad():
            for d in tqdm.tqdm(loader):
                d.x = d.x.type(torch.FloatTensor)
                d.edge_index = d.edge_index.type(torch.LongTensor)
                d.edge_attr = d.edge_attr.type(torch.LongTensor)
                d = d.to("cuda")
                ee = model(d).detach().cpu().numpy()
                for e in ee:
                    index.add_item(i, e)
                    i+=1

    print("Building Index")
    index.build(16)
    print("Saving Keys")
    with open("/data/d1/pfahler/arxiv_processed/all_keys_{}.csv".format(run), "w") as f:
        f.write("\n".join(all_keys))

if __name__ == "__main__":
    main(checkpoint=sys.argv[1])
    