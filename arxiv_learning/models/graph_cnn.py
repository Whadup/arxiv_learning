import torch
from torch.nn import BatchNorm1d
from torch_geometric.nn import MessagePassing, GatedGraphConv, GraphConv
from torch_scatter import scatter_min, scatter_mean
from arxiv_learning.data.load_mathml import DIM, MAX_POS, TAG_SYMBOLS, CONTENT_SYMBOLS, ATTRIBUTE_SYMBOLS
from arxiv_learning.models.positional_embeddings import init_positional_embeddings
from arxiv_learning.models.softnormalization import SoftNormalization


class FirstLayer(MessagePassing):
    def __init__(self, input_dim, hidden_dim):
        super(FirstLayer, self).__init__(aggr='mean')  # "Add" aggregation.
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            # torch.nn.ReLU(hidden_dim),
            # torch.nn.Linear(hidden_dim, hidden_dim)
        )
        self.positional_emb = torch.nn.Embedding(MAX_POS, hidden_dim)
        self.scale = torch.nn.Parameter(torch.ones(1) * 0.1)

        self.gate = torch.nn.Linear(hidden_dim, hidden_dim)

        init_positional_embeddings(self.positional_emb)
        torch.nn.init.normal_(self.nn[0].weight)
        self.relu = torch.nn.ReLU()
        self.hidden_dim = hidden_dim

    def forward(self, x, edge_index, edge_attr):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.nn(x)
        # Step 3-5: Start propagating messages.
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_index, edge_attr):
        # x_j has shape [E, out_channels]

        # Step 3: Normalize node features.
        # print(x_j.shape, edge_attr.shape)
        tmp = self.relu(x_j + self.scale * self.positional_emb(edge_attr).view(-1, self.hidden_dim))
        # print(tmp.shape, self.positional_emb(edge_attr).shape)
        return tmp

    def update(self, aggr_out, x):
        # aggr_out has shape [N, out_channels]

        # Step 5: Return new node embeddings.
        h = torch.sigmoid(self.gate(x))
        return h * aggr_out + (1.0 - h) * x
        # return aggr_out

class GraphCNN(torch.nn.Module):
    def __init__(self):
        super(GraphCNN, self).__init__()
        self.verbose = 0
        self.mean_aggregation = True
        self.intermediate = 512
        self.later = 512
        self.output_dim = 64
        self.drop = torch.nn.Dropout(0.15)
        self.first_layer = FirstLayer(DIM, self.intermediate)
        # self.second_layer = FirstLayer(self.intermediate, self.intermediate)


        self.batch_norm = BatchNorm1d(self.intermediate)
        self.deep_stuff = torch.nn.ModuleList([
            GraphConv(self.intermediate, self.later),
            torch.nn.ReLU(),
            GraphConv(self.later, self.later),
            torch.nn.ReLU(),
            BatchNorm1d(self.later),
            GraphConv(self.later, self.later),
            torch.nn.ReLU(),
            # GatedGraphConv(self.intermediate, num_layers=10, aggr="mean"),
            #BatchNorm1d(self.intermediate),
            #torch.nn.Linear(self.intermediate, self.later),
    ])

        self.batch_norm2 = BatchNorm1d(self.later)
        self.output = torch.nn.Linear(self.later, self.output_dim)
        self.normalizer = SoftNormalization(self.output_dim)

        self.masked_head = torch.nn.Linear(self.later, DIM+2)
        self.loss_type = torch.nn.CrossEntropyLoss(reduction="none")
        weights = torch.ones(ATTRIBUTE_SYMBOLS+1)
        weights[:-1] += 0.5/ATTRIBUTE_SYMBOLS
        weights[-1] -= 0.5
        self.loss_attr = torch.nn.CrossEntropyLoss(weight=weights, reduction="none")
        weights = torch.ones(CONTENT_SYMBOLS+1)
        weights[:-1] += 0.5/CONTENT_SYMBOLS
        weights[-1] -= 0.5
        self.loss_char = torch.nn.CrossEntropyLoss(weight=weights, reduction="none")

        self.checkpoint_string = "equation_encoder_graph_cnn_checkpoint{}.pt"
        self.save_path = "graph_cnn.pt"

    def forward(self, data, aggregation=True):
        inp, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # print(x.shape)
        if inp.is_cuda:
            mask = torch.ones((inp.shape[0], 1)).to(data.batch.get_device())
        else:
            mask = torch.ones((inp.shape[0], 1))
        if self.training:
            mask = self.drop(mask) * (1-0.15)
            x = self.first_layer(mask * inp, edge_index, edge_attr)
        else:
            x = self.first_layer(inp, edge_index, edge_attr)
        x = self.batch_norm(x)

        # x = self.second_layer(x, edge_index, edge_attr)

        # print(x.shape)
        for m in self.deep_stuff:
            if isinstance(m, MessagePassing):
                x = m(x, edge_index=edge_index)
            else:
                x = m(x)

        if self.training:
            outputs = self.masked_head(x)
            outputs_type = outputs[:, :TAG_SYMBOLS]
            outputs_char = outputs[:, TAG_SYMBOLS:TAG_SYMBOLS+CONTENT_SYMBOLS+1]
            outputs_attr = outputs[:, TAG_SYMBOLS+CONTENT_SYMBOLS+1:]
            label_type = torch.argmax(inp[:, :TAG_SYMBOLS], dim=1)
            m_char, label_char = torch.max(inp[:, TAG_SYMBOLS:TAG_SYMBOLS+CONTENT_SYMBOLS], dim=1)
            const_char = torch.tensor(CONTENT_SYMBOLS)
            const_attr = torch.tensor(ATTRIBUTE_SYMBOLS)
            if inp.is_cuda:
                const_char = const_char.to(data.batch.get_device())
                const_attr = const_attr.to(data.batch.get_device())
            label_char = torch.where(m_char != 0, label_char, const_char)
            m_attr, label_attr = torch.max(inp[:, TAG_SYMBOLS+CONTENT_SYMBOLS+1:], dim=1)
            label_attr = torch.where(m_attr != 0, label_attr, const_attr)

            # if self.verbose%100 == 99:
            # 	sm = torch.nn.functional.softmax(outputs_char[13])
            # 	print(sm, sm.shape, CONTENT_SYMBOLS)
            # 	print(sm[label_char[13]].item(), sm.max().item(), sm.mean().item(), )
            # 	print(self.loss_char(outputs_char, label_char)[13], label_char[13].item())
            # 	self.verbose = 0
            # self.verbose+=1

            loss = 1.0/3 * (1-mask).view(-1) * (self.loss_type(outputs_type, label_type) +
                    self.loss_attr(outputs_attr, label_attr) +
                    self.loss_char(outputs_char, label_char))
            loss = loss.sum() / torch.clamp((1-mask).sum(), min=1)
        # print(x.shape)
        # Get all root nodes from data.batch

        if not aggregation:
            emb = self.output(x)
            emb = self.normalizer(emb)
            return emb

        if hasattr(data, "batch"):
            # print("batch")
            if self.mean_aggregation:
                emb = scatter_mean(x, data.batch, dim=0)
            else:
                indices = torch.arange(data.batch.shape[0], dtype=torch.int64).to(data.batch.get_device())
                _, roots = scatter_min(indices, data.batch, dim=0)
                # # emb = torch.zeros((data.num_graphs, 64))
                # # emb = torch.gather(x, 0, roots)
                emb = x[roots, :]
        else:
            # print("single item")
            if self.mean_aggregation:
                emb = x.mean(dim=0)
            else:
                emb = x[0]
        #emb = self.batch_norm2(emb)
        emb = self.output(emb)
        # print(emb.shape)
        emb = self.normalizer(emb)
        # normalize
        if self.training:
            return emb, loss
        return emb

    def forward3(self, data):
        batch_size = data.num_graphs // 3
        if self.training:
            emb, loss = self.forward(data)
        else:
            emb = self.forward(data)
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
            #semihard negative sampling (smallest violating example)
            # d = torch.matmul(out1, out3.transpose(0,1))
            # d = torch.where(d>dist_sim.view(-1,1), d, torch.tensor(100.0).to(dist_sim.get_device())).min(dim=1)[0]

            #averagehard negative sampling
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
