import json
import torch
import tqdm
import numpy as np
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
                self.rhs.append(lhs)
    def __len__(self):
        return 2 * len(self.lhs)
    def __iter__(self):
        order = np.random.choice(len(self.lhs), len(self.lhs))
        for i in order:
            yield self.lhs[i]
            yield self.rhs[i]
    
def finetune(model, alphabet, train_file, epochs=10, tau=0.05):
    train_data = FinetuneDataset(train_file, alphabet)
    train_loader = DataLoader(train_data, batch_size=2 * 512)

    optimizer = torch.optim.Adam(model.parameters(), tau)
    loss_function = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        with tqdm.tqdm(total=len(train_loader)) as pbar:
            for data in train_loader:
                data = data.to("cuda")
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
                pbar.set_description("{}".format(loss.mean().item()))


def test(model, alphabet, test_file):
    test_data = FinetuneDataset(test_file, alphabet)
    test_loader = DataLoader(test_data, batch_size=2 * 512)


def main():
    checkpoint = "pretrained_graph_cnn.pt"
    model = GraphCNN(width=256, layer=GatedGraphConv, args=(4,))
    model.load_state_dict_from_path(checkpoint)
    model = model.cuda().train()

    alphabet = load_mathml.load_alphabet("/data/pfahler/arxiv_v2/vocab.pickle") 
    finetune(model, alphabet, "data/finetune_inequalities_train.jsonl")


if __name__ == "__main__":
    main()
