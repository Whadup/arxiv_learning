import torch
import torch.optim as optim
from arxiv_learning.nn.softnormalization import SoftNormalization
from arxiv_learning.nn.scheduler import WarmupLinearSchedule
import arxiv_learning.nn.loss as losses
from .head import Head

class HistogramLossHead(Head):
    def __init__(self, model, width=512, output_dim=64):

        super().__init__(model)
        self.width = width
        self.output_dim = output_dim
        self.loss = losses.HistogramLoss(weighted=False)#.to(device)
        self.output = torch.nn.Linear(self.width, self.output_dim)
        self.normalizer = SoftNormalization(self.output_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.scheduler = WarmupLinearSchedule(self.optimizer, 500, 20 * 10000)
    def reset_metrics(self):
        self.example_cnt = 0
        self.running_accuracy = 0
        self.running_loss = 0

    def metrics(self):
        return {
            "Number-Examples": self.example_cnt,
            "Histogram-Loss": round(self.running_loss / (1 if self.example_cnt < 1 else self.example_cnt), 4),
            "Accuracy": round(self.running_accuracy / (1 if self.example_cnt < 1 else self.example_cnt), 4)
        }

    def forward(self, data):
        x = self.model(data)
        if hasattr(data, "batch"):
            emb = scatter_mean(x, data.batch, dim=0)
        else:
            emb = x.mean(dim=0, keepdim=True)
        emb = self.output(emb)
        emb = self.normalizer(emb)

        batch_size = data.num_graphs // 3

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

        # dist_sim, dist_dissim1, dist_dissim2 = self.model.forward3(data)
        loss = self.loss(dist_sim, dist_dissim.view(-1), dist_dissim2, torch.ones(1), torch.zeros(1))

        acc = dist_sim > dist_dissim
        loss = torch.mean(loss)
        actual_batch_size = dist_sim.shape[0]
        self.running_accuracy += acc.sum().item()
        self.running_loss += loss.item()*dist_sim.shape[0]
        self.example_cnt += actual_batch_size

        self.target = loss
