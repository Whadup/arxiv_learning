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
import tqdm
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
    # model = fasttext.load_model("fasttext.bin")
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


def fine_tune(data, model, epoch=1, dim=64, tau=0.05):
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
    optimizer = torch.optim.Adam(head.parameters(), tau)
    dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X1, X2), batch_size=1024, drop_last=True)
    labels = torch.arange(1024)
    lossfunction = torch.nn.CrossEntropyLoss()
    for i in range(epoch):
        for X, Y in dataloader:
            x = head(X)
            y = head(Y)
            x = x / (1e-6 + torch.norm(x, dim=1, keepdim=True))
            y = y / (1e-6 + torch.norm(x, dim=1, keepdim=True))
            sims = torch.matmul(x, y.transpose(0, 1)) / tau
            loss = lossfunction(sims, labels)
            print(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    return head

def test(model, finetuned_model, data):
    from annoy import AnnoyIndex
    index = AnnoyIndex(finetuned_model[-1].out_features, "angular")
    X1 = []
    X2 = []
    for a, b in zip(*data):
        X1.append(model.get_sentence_vector(a))
        X2.append(model.get_sentence_vector(b))
    X1 = torch.tensor(np.array(X1))
    X2 = torch.tensor(np.array(X2))
    dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X1, X2), batch_size=1024, drop_last=False)

    total = 0
    with torch.no_grad():
        for X, Y in tqdm.tqdm(dataloader):
            x = finetuned_model.forward(X)
            y = finetuned_model.forward(Y)
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
    #TODO: Log to Sacred
    print("RANKS: mean {}, fails {}, recall@1 {}, recall@10 {} recall@100 {}".format(
        ranks.mean(), fail, recall_at_1, recall_at_10, recall_at_100))
    return dict(mean_rank=ranks.mean(), fails=fail, fail_ratio=fail / (1.0 * len(ranks) + fail), recall_at_1=recall_at_1, recall_at_10=recall_at_10, recall_at_100=recall_at_100)

def main():
    for epochs in [1, 3, 5, 10]:
        for dim in [64, 128, 256]:
            train_config = dict(dim=dim, epoch=epochs, ws=5, ngrams=2)
            finetune_config = dict(dim=dim, epoch=10, tau=0.01)
            with meticulous.Experiment({"train": train_config, "fine_tune": finetune_config}) as exp:
                model = train(**train_config)
                for tuning_set in ["finetune_equalities_train.jsonl", "finetune_inequalities_train.jsonl", "finetune_relations_train.jsonl"]:
                    train_data = load_finetune_data(tuning_set)
                    finetuned_model = fine_tune(train_data, model, **finetune_config)
                    test_data = load_finetune_data(tuning_set, test=True)
                    exp.summary({tuning_set:test(model, finetuned_model, test_data)})

if __name__ == "__main__":
    main()