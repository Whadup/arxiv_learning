"""Finetune GraphCNNs for Equality Identification"""
import os
import json
import torch
import tqdm
import numpy as np
import meticulous
from torch_geometric.data import DataLoader
from torch_geometric.nn import GatedGraphConv, GraphConv, FeaStConv, GATConv
from torch_scatter import scatter_min, scatter_mean
from arxiv_learning.nn.graph_cnn import GraphCNN
import arxiv_learning.data.load_mathml as load_mathml

class FinetuneDataset(torch.utils.data.IterableDataset):
    def __init__(self, file, alphabet):
        super().__init__()
        self.lhs = []
        self.rhs = []
        self.alphabet = alphabet
        with open(file, "r") as f:
            for line in f:
                line = json.loads(line)
                try:
                    lhs = load_mathml.load_pytorch(line["part_a"], self.alphabet)
                    rhs = load_mathml.load_pytorch(line["part_b"], self.alphabet)
                except:
                    continue
                self.lhs.append(lhs)
                self.rhs.append(rhs)
    def __len__(self):
        return 2 * len(self.lhs)
    def __iter__(self):
        order = np.random.choice(len(self.lhs), len(self.lhs))
        for i in order:
            yield self.lhs[i]
            yield self.rhs[i]
    
def finetune(model, alphabet, train_file, epochs=10, tau=0.05, lr=1e-3):
    train_data = FinetuneDataset(train_file, alphabet)
    train_loader = DataLoader(train_data, batch_size=2 * 512, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr)
    loss_function = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        with tqdm.tqdm(total=len(train_loader)) as pbar:
            for data in train_loader:
                data = data.cuda()
                pbar.update(1)
                x = model(data)
                if hasattr(data, "batch"):
                    emb = scatter_mean(x, data.batch, dim=0)
                else:
                    emb = x.mean(dim=0, keepdim=True)
                norm = torch.norm(emb, dim=1, keepdim=True) + 1e-8
                emb = emb.div(norm.expand_as(emb))

                batch_size = data.num_graphs // 2

                emb = emb.view(batch_size, 2, -1)
                out1 = emb[:, 0, :]
                out2 = emb[:, 1, :]
                sims = torch.matmul(out1, out2.transpose(0, 1)) / tau
                gt = torch.arange(0, batch_size, dtype=torch.long, device=sims.get_device())
                loss = loss_function(sims, gt)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                pbar.set_description("{}".format(loss.item()))


def test(model, alphabet, test_file):
    from annoy import AnnoyIndex
    test_data = FinetuneDataset(test_file, alphabet)
    test_loader = DataLoader(test_data, batch_size=2 * 512)
    index = AnnoyIndex(model.width, "angular")
    model = model.eval()
    
    total = 0
    with tqdm.tqdm(total=len(test_data), ncols=120, dynamic_ncols=False, smoothing=0.1, position=0) as pbar:
        pbar.set_description("testing")
        with torch.no_grad():
            for data in test_loader:
                data = data.to("cuda")
                x = model(data)
                if hasattr(data, "batch"):
                    emb = scatter_mean(x, data.batch, dim=0)
                else:
                    emb = x.mean(dim=0, keepdim=True)
                norm = torch.norm(emb, dim=1, keepdim=True) + 1e-8
                emb = emb.div(norm.expand_as(emb))
                batch_size = data.num_graphs // 2

                emb = emb.view(batch_size, 2, -1)
                for i, e in enumerate(emb.cpu().numpy()):
                    index.add_item(total * 2, e[0, :])
                    index.add_item(total * 2 + 1, e[1, :])
                    total += 1
                pbar.update(emb.shape[0] * 2)
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
    for epochs in [1, 5, 10, 50]:
        for tau in [0.05, 0.01]:
            for lr in [1e-3, 5e-4, 1e-4]:
                finetune_config={
                    "checkpoint": "experiments/20/graph_cnn.pt",
                    "epochs": epochs,
                    "tau": tau,
                    "lr" : lr
                }
                with meticulous.Experiment(finetune_config) as exp:
                    checkpoint = finetune_config.pop("checkpoint")
                    for tuning_set in ["finetune_equalities_train.jsonl", "finetune_inequalities_train.jsonl", "finetune_relations_train.jsonl"]:
                        model = GraphCNN(width=256, layer=GatedGraphConv, args=(4,))
                        model.load_state_dict_from_path(checkpoint)
                        model = model.cuda().train()
                        
                        alphabet = load_mathml.load_alphabet("/data/pfahler/arxiv_v2/vocab.pickle")
                        finetune(model, alphabet, os.path.join("data", tuning_set), **finetune_config)
                        exp.summary({
                            tuning_set:test(model, alphabet, os.path.join("data", tuning_set.replace("train", "test")))
                        })

if __name__ == "__main__":
    main()