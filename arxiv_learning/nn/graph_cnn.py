import torch
import numpy as np
from torch.nn import BatchNorm1d
from torch_geometric.nn import MessagePassing, GatedGraphConv, GraphConv
from torch_scatter import scatter_min, scatter_mean
from arxiv_learning.data.load_mathml import MAX_POS, VOCAB_SYMBOLS
from arxiv_learning.nn.positional_embeddings import init_positional_embeddings
from arxiv_learning.nn.softnormalization import SoftNormalization

class GraphCNN(torch.nn.Module):
    def __init__(self, layers=4, width=512, layer=GraphConv, args=(512,), kwargs={"aggr": "mean"}):
        super(GraphCNN, self).__init__()
        self.width = width
        self.layers = layers
        self.output_dim = 64
        # self.first_layer = FirstLayer(self.width)

        self.positional_emb = torch.nn.Embedding(MAX_POS, self.width)
        self.vocab_emb = torch.nn.Embedding(VOCAB_SYMBOLS, self.width)
        init_positional_embeddings(self.positional_emb, normalize=True)
        torch.nn.init.xavier_normal_(self.vocab_emb.weight, np.sqrt(2))
        
        self.deep_stuff = torch.nn.ModuleList(sum([
            [BatchNorm1d(self.width),
            layer(self.width, *args, **kwargs),
            torch.nn.ReLU()] for i in range(layers)
            ], []))
        
        # Move these parts to individual heuristic Heads
        self.output = torch.nn.Linear(self.width, self.output_dim)
        self.normalizer = SoftNormalization(self.output_dim)
        self.masked_head = torch.nn.Linear(self.width, VOCAB_SYMBOLS)
        self.loss_vocab = torch.nn.CrossEntropyLoss(reduction="none")
        
        self.checkpoint_string = "equation_encoder_graph_cnn_checkpoint{}.pt"
        self.save_path = "graph_cnn.pt"

    def forward(self, *inp):
        data = inp[0]
        inp, edge_index, edge_attr, pos = data.x, data.edge_index, data.edge_attr, data.pos
        # print(x.shape)
        if  self.training:
            mask = torch.rand((inp.shape[0], 1)) < 0.15
            unk = torch.ones(1, dtype=torch.int64) * (VOCAB_SYMBOLS - 1)
            if inp.is_cuda:
                mask = mask.to(data.batch.get_device())
                unk = unk.to(data.batch.get_device())
            x = torch.where(mask, unk, inp)
            mask = mask.type(torch.float32)
        else:
            x = inp
        x = (self.vocab_emb(x) + self.positional_emb(pos)).view(-1, self.width)
        for m in self.deep_stuff:
            if isinstance(m, MessagePassing):
                x = m(x, edge_index=edge_index)
            else:
                x = m(x)

        #Move to individual heads
        if self.training:
            outputs = self.masked_head(x)
            loss = (1-mask).view(-1) * self.loss_vocab(outputs, inp.view(-1))
            loss = loss.sum() / torch.clamp((1-mask).sum(), min=1)
            return x, loss
        return x

    def mean(self, data):
        x = self(data)
        loss = None
        if isinstance(x, tuple):
            x, loss = x
        if hasattr(data, "batch"):
            emb = scatter_mean(x, data.batch, dim=0)
        else:
            emb = x.mean(dim=0, keepdim=True)
        emb = self.output(emb)
        emb = self.normalizer(emb)
        if loss:
            return emb, loss
        return emb

    def root(self, data):
        x = self(data)
        loss = None
        if isinstance(x, tuple):
            x, loss = x
        if hasattr(data, "batch"):
            indices = torch.arange(data.batch.shape[0], dtype=torch.int64).to(data.batch.get_device())
            _, roots = scatter_min(indices, data.batch, dim=0)
            emb = x[roots, :]
        else:
            emb = x[0].view(1, -1)
        emb = self.output(emb)
        emb = self.normalizer(emb)
        if loss:
            return emb, loss
        return emb

    def forward3(self, data):
        batch_size = data.num_graphs // 3
        if self.training:
            emb, loss = self.mean(data)
        else:
            emb = self.mean(data)
        emb = emb.view(batch_size, 3, -1)
        out1 = emb[:, 0, :]
        out2 = emb[:, 1, :]
        out3 = emb[:, 2, :]
        dist_sim = torch.bmm(out1.view(-1, 1, self.output_dim),
                                out2.view(-1, self.output_dim, 1)).view(-1)
        dist_dissim = torch.bmm(out1.view(-1, 1, self.output_dim),
                                out3.view(-1, self.output_dim, 1)).view(-1)
        dist_dissim2 = torch.bmm(out2.view(-1, 1, self.output_dim),
                                out3.view(-1, self.output_dim, 1)).view(-1)
        if self.training:
            d = torch.matmul(out1, out3.transpose(0,1))#.mean(dim=1)
            dist_dissim = d
            return dist_sim, dist_dissim, dist_dissim2, loss
        return dist_sim, dist_dissim, dist_dissim2

    def save(self):
        """store the final model"""
        torch.save(self.state_dict(), self.save_path)

    def load(self):
        """load a trained model"""
        self.load_state_dict(torch.load(self.save_path, map_location='cpu'))

    def save_checkpoint(self, epoch):
        """save a checkpoint"""
        torch.save(self.state_dict(), self.checkpoint_string.format(epoch))

    def load_checkpoint(self, epoch):
        """restore a checkpoint"""
        self.load_state_dict(torch.load(self.checkpoint_string.format(epoch), map_location='cpu'))

    def load_state_dict_from_path(self, path):
        """load a model from a wrong path"""
        self.load_state_dict(torch.load(path, map_location='cpu'))



if __name__ == "__main__":
    import arxiv_learning.data.load_mathml
    a = GraphCNN(layer=GatedGraphConv, args=(1,))
    vocab = arxiv_learning.data.load_mathml.load_alphabet("/data/s1/pfahler/arxiv_processed/subset_ml/train/vocab.pickle")
    x = arxiv_learning.data.load_mathml.load_pytorch(arxiv_learning.data.load_mathml.EXAMPLE, vocab)
    print(a(x))
