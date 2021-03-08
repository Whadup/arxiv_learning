"""
We train another model, facebooks fasttext, on the pre-order trees.
Then we test with finetuning tasks of equality prediction.
"""
import torch
import fasttext
import numpy as np
import json
import zipfile
import pickle
import meticulous
from .fasttext_data import mathml_to_string
import annoy


def train(dim=64, epoch=1, ws=5, ngrams=2):
    model = fasttext.train_unsupervised(
        "/data/s1/pfahler/arxiv_v2/plaintext_train.txt",
        model="skipgram",
        dim=dim,
        ws=ws,
        epoch=epoch,
        wordNgrams=ngrams,
    )
    model.save_model("fasttext.bin")
    return model

def load_finetune_data(basefile, test=False):
    X1 = []
    X2 = []
    with open(basefile.replace("train", "test") if test else basefile, "r") as f:
        for example in f:
            example = json.loads(example)
            try:
                a = mathml_to_string(example["part_a"])
                b = mathml_to_string(example["part_b"])
                X1.append(a)
                X2.append(b)
            except:
                pass
    return X1, X2


def fine_tune(data, model, epoch=1, dim=64):
    X1 = []
    X2 = []
    for a,b in zip(*data):
        X1.append(model.get_sentence_vector(a))
        X2.append(model.get_sentence_vector(b))
    X1 = torch.tensor(np.array(X1))
    X2 = torch.tensor(np.array(X2))

    head = torch.nn.Sequential(
        torch.nn.Linear(model.get_dimension(), dim)
    )
    optimizer = torch.optim.Adam(head.parameters(), 0.01)
    dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X1, X2), batch_size=1024, drop_last=True)
    labels = torch.arange(1024)
    lossfunction = torch.nn.CrossentropyLoss()
    for i in range(epoch):
        for X, Y in dataloader:
            x = head(X)
            y = head(Y)
            x = x / (1e-6 + torch.norm(x, dim=1, keepdim=True))
            y = y / (1e-6 + torch.norm(x, dim=1, keepdim=True))
            sims = torch.matmul(x, y.transpose(0, 1)) / 0.01
            loss = lossfunction(sims, labels)
            print(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


def main():
    config = dict(dim=64, epoch=1, ws=5, ngrams=2)
    with meticulous.Experiment(config):
        model = train(**config)
        train_data = load_finetune_data("finetune_equalities_train,jsonl")
        fine_tune(train_data, model)
        test_data = load_finetune_data("finetune_equalities_train,jsonl")

if __name__ == "__main__":
    main()