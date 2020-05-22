import numpy as np

import torch

import sys

from arxiv_learning.nn.graph_cnn import GraphCNN
from arxiv_learning.data.load_mathml import load_alphabet, load_pytorch
from arxiv_learning.data.load_mathml import VOCAB_SYMBOLS

import json

def compute_embeddings(dataloader, net, device):
    embs = []
    labs = []
    with torch.no_grad():
        for (x, y) in dataloader:
            x.to(device)
            # print(x,y)
            emb = net.mean(x).detach().cpu().numpy()
            # print(emb)
            embs.append(emb[0])
            labs.append(y)
            # for e in emb:
            #     embs.append(e)
            #     # print(e)
            # for l in y.cpu().detach().numpy():
            #     labs.append(l)
                # print(l)
    X = np.array(embs)
    y = np.expand_dims(np.array(labs),axis=1)
    return X,y

def compute_knn_accuracy(x, y):
    from sklearn.model_selection import LeaveOneOut, cross_validate
    from sklearn.neighbors import KNeighborsClassifier
    NN = KNeighborsClassifier(1, algorithm="brute")
    acc1 =  np.mean(cross_validate(estimator=NN, cv = LeaveOneOut(), X=x, y=y.ravel())["test_score"])
    NN = KNeighborsClassifier(2, algorithm="brute")
    acc2 =  np.mean(cross_validate(estimator=NN, cv = LeaveOneOut(), X=x, y=y.ravel())["test_score"])
    NN = KNeighborsClassifier(3, algorithm="brute")
    acc3 =  np.mean(cross_validate(estimator=NN, cv = LeaveOneOut(), X=x, y=y.ravel())["test_score"])
    return acc1, acc2, acc3

def compute_triples_loss(X, y, sample=False):
    dist = X.dot(X.transpose())
    counter = 0
    stat = 0
    hinge = 0
    for i in range(len(X)):
        for j in range(i+1, len(X)):
            if y[j] != y[i]:
                continue
            for k in range(len(X)) if not sample else np.random.choice(len(X), 20):
                if y[k] == y[i]:
                    continue
                dist_sim = dist[i, j]
                dist_dissim = max(dist[i, k], dist[j, k])
                hinge += max(0,dist_dissim-dist_sim+1)
                if dist_sim < dist_dissim:
                    stat += 1
                counter += 1
    return 1.0*stat/counter, 1.0*hinge/counter

def compute_pairs_loss(X, y, sample=False):
    dist = X.dot(X.transpose())
    counter_sim = 0
    counter_dissim = 0
    hinge_sim = 0.0
    hinge_dissim = 0.0
    for i in range(len(X)):
        for j in range(i+1, len(X)):
            if y[j] == y[i]:
                counter_sim += 1
                hinge_sim += max(0,1.0-dist[i, j])
        for j in range(i+1, len(X)) if not sample else np.random.choice(len(X), 20):
            if y[j] != y[i]:
                counter_dissim += 1
                hinge_dissim += max(0,1.0+dist[i, j])
    return hinge_sim/counter_sim, hinge_dissim/counter_dissim

def sorted_nicely(l):
    import re
    """ Sort the given iterable in the way that humans # expect.""" 
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

# @# ex.capture
def process_epochs(run):
    global VOCAB_SYMBOLS
    import os
    print(run)
    info = json.load(open(os.path.join(run,"info.json"), "r"))
    VOCAB_SYMBOLS = info.get("vocab_dim", VOCAB_SYMBOLS)
    vocab_file = info["vocab_file"]
    for filename in sorted_nicely(os.listdir(run))[::-1]:
        print(filename)
        if filename.endswith(".pt"):
            print("========", filename, "========")
            eval(pretrained_weights=os.path.join(run, filename), vocab_file=vocab_file)
            print("")
            break

def eval(pretrained_weights, vocab_file):
    net = GraphCNN()
    net.load_state_dict_from_path(pretrained_weights)
    net = net.eval()
    alphabet = load_alphabet(vocab_file)
    xml_data = json.load(open("eval.json", "r"))
    dataset = []
    for i, label in enumerate(xml_data):
        for xml in xml_data[label]:
            dataset.append((load_pytorch(xml, alphabet), i))
    

    device = "cpu"
    net = net.to(device)

    X, y = compute_embeddings(dataset, net, device)
    d = np.hstack((y, X))
    np.savetxt("eval.csv",d)

    acc1, acc2, acc3 = compute_knn_accuracy(X, y)
    print("accuracy_1nn", acc1,sep="\t")
    # ex.log_scalar("accuracy_1nn", acc1)
    print("accuracy_2nn", acc2,sep="\t")
    # ex.log_scalar("accuracy_2nn", acc2)
    print("accuracy_3nn", acc3,sep="\t")
    # ex.log_scalar("accuracy_3nn", acc3)

    discrete_triples_loss, hinge_triples_loss = compute_triples_loss(X,y)
    print("ranking", 1.0 - discrete_triples_loss,sep="\t")
    # ex.log_scalar("discrete_triples_loss", discrete_triples_loss)
    print("loss", hinge_triples_loss,sep="\t")
    # ex.log_scalar("hinge_triples_loss", hinge_triples_loss)

    pairs_pos_loss, pairs_neg_loss = compute_pairs_loss(X,y)
    print("pairs_pos_loss", pairs_pos_loss,sep="\t")
    # ex.log_scalar("pairs_pos_loss", pairs_pos_loss)
    print("pairs_neg_loss", pairs_neg_loss,sep="\t")
    # ex.log_scalar("pairs_neg_loss", pairs_neg_loss)
    pairs_loss = 0.5 * (pairs_pos_loss + pairs_neg_loss)
    print("pairs_loss",pairs_loss,sep="\t")
    # ex.log_scalar("pairs_loss",pairs_loss)

#@ex.automain
def main():
    import sys
    process_epochs(sys.argv[1])


if __name__ == "__main__":
    main()