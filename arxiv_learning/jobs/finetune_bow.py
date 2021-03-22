"""
We train another model, a linear BoW, on the pre-order trees for finetuning tasks of equality prediction.
"""
import os
import torch
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import json
import zipfile
import pickle
import meticulous
import tqdm
from .fasttext_data import mathml_to_string, mathml_to_root_path
import annoy


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


def fine_tune(data, epoch=1, dim=64, tau=0.05, lr=1e-3):
    X1 = []
    X2 = []
    vectorizer = CountVectorizer(token_pattern=r"[^ ]+")
    vectorizer.fit(data[0] + data[1])
    # print(list(vectorizer.vocabulary_.items())[:100])
    MAX_SEQ_LENGTH = 2048
    X = vectorizer.transform(data[0])
    for x in X:
        xt = torch.zeros(MAX_SEQ_LENGTH, dtype=torch.int64)
        tmp = []
        for i, c in zip(x.indices, x.data):
            for _ in range(c):
                tmp.append(i)
        xt[:len(tmp)] = torch.tensor(tmp, dtype=torch.int64)
        X1.append(xt)
    X = vectorizer.transform(data[1])
    for x in X:
        xt = torch.zeros(MAX_SEQ_LENGTH, dtype=torch.int64)
        tmp = []
        for i,c in zip(x.indices, x.data):
            for _ in range(c):
                tmp.append(i)
        xt[:len(tmp)] = torch.tensor(tmp, dtype=torch.int64)
        X2.append(xt)

    X1 = torch.stack(X1)
    X2 = torch.stack(X2)

    head = torch.nn.Sequential(
        torch.nn.Embedding(len(vectorizer.vocabulary_), dim, padding_idx=0)
    )
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X1, X2), batch_size=1024, drop_last=True, shuffle=True)
    labels = torch.arange(1024)
    lossfunction = torch.nn.CrossEntropyLoss()
    for i in range(epoch):
        for X, Y in dataloader:
            x = head(X).sum(dim=1) / (1e-5 + (X>0).sum(dim=1, keepdim=True))
            y = head(Y).sum(dim=1) / (1e-5 + (Y>0).sum(dim=1, keepdim=True))
            x = x / (1e-6 + torch.norm(x, dim=1, keepdim=True))
            y = y / (1e-6 + torch.norm(x, dim=1, keepdim=True))
            sims = torch.matmul(x, y.transpose(0, 1)) / tau
            loss = lossfunction(sims, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(loss.item())
    return (head, vectorizer)

def test(model, data):
    from annoy import AnnoyIndex
    model, vectorizer = model
    index = AnnoyIndex(model[-1].embedding_dim, "angular")
    MAX_SEQ_LENGTH = 2048
    X1 = []
    X2 = []
    X = vectorizer.transform(data[0])
    for x in X:
        xt = torch.zeros(MAX_SEQ_LENGTH, dtype=torch.int64)
        tmp = []
        for i, c in zip(x.indices, x.data):
            for _ in range(c):
                tmp.append(i)
        xt[:len(tmp)] = torch.tensor(tmp, dtype=torch.int64)
        X1.append(xt)
    X = vectorizer.transform(data[1])
    for x in X:
        xt = torch.zeros(MAX_SEQ_LENGTH, dtype=torch.int64)
        tmp = []
        for i,c in zip(x.indices, x.data):
            for _ in range(c):
                tmp.append(i)
        xt[:len(tmp)] = torch.tensor(tmp, dtype=torch.int64)
        X2.append(xt)

    X1 = torch.stack(X1)
    X2 = torch.stack(X2)

    dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X1, X2), batch_size=1024, drop_last=False)

    total = 0
    with torch.no_grad():
        for X, Y in tqdm.tqdm(dataloader):
            x = model.forward(X).sum(dim=1) / (1e-5 + (X > 0).sum(dim=1, keepdim=True))
            y = model.forward(Y).sum(dim=1) / (1e-5 + (Y > 0).sum(dim=1, keepdim=True))
            for i, (e1, e2) in enumerate(zip(x.cpu().numpy(), y.cpu().numpy())):
                index.add_item(total * 2, e1)
                index.add_item(total * 2 + 1, e2)
                total += 1
    index.build(16)
    index.save("tmp.ann")
    fail = 0
    ranks = []
    for i in tqdm.tqdm(range(total)):
        results = index.get_nns_by_item(2 * i, 1000)[1:]
        results = np.array(results) // 2
        rank = np.argwhere(results == i)
        if not len(rank):
            fail += 1
        else:
            ranks.append(rank[0])
    ranks = np.array(ranks)
    recall_at_1 = (ranks < 1).sum() / (1.0 * len(ranks) + fail)
    recall_at_10 = (ranks < 10).sum() / (1.0* len(ranks) + fail)
    recall_at_100 = (ranks < 100).sum() / (1.0 * len(ranks) + fail)
    print("RANKS: mean {}, fails {}, recall@1 {}, recall@10 {} recall@100 {}".format(
        ranks.mean(), fail, recall_at_1, recall_at_10, recall_at_100))
    return dict(mean_rank=ranks.mean(), fails=fail, fail_ratio=fail / (1.0 * len(ranks) + fail), recall_at_1=recall_at_1, recall_at_10=recall_at_10, recall_at_100=recall_at_100)

def main():
    for epochs in [1, 3, 10]:
        for dim in [64, 128, 256]:
            for lr in [1e-2, 1e-3, 1e-4]:
                train_config = dict(dim=dim, epoch=epochs, lr=lr, tau=0.05)
                print(train_config)
                with meticulous.Experiment(train_config, experiments_directory="bow_experiments") as exp:
                    for tuning_set in ["finetune_equalities_train.jsonl", "finetune_inequalities_train.jsonl", "finetune_relations_train.jsonl"]:
                        train_data = load_finetune_data(os.path.join("data",tuning_set))
                        model = fine_tune(train_data, **train_config)
                        test_data = load_finetune_data(os.path.join("data",tuning_set), test=True)
                        exp.summary({tuning_set:test(model, test_data)})

if __name__ == "__main__":
    main()