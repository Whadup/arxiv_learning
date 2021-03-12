import torch
import torch.optim as optim
from torch_scatter import scatter_min, scatter_mean
from arxiv_learning.nn.softnormalization import SoftNormalization
from arxiv_learning.nn.scheduler import WarmupLinearSchedule
from .head import Head

class InfoNCEHead(Head):
    def __init__(self, model, width=512, output_dim=64, tau=0.05):

        super().__init__(model)
        self.width = width
        self.output_dim = output_dim
        self.tau = tau
        self.output = torch.nn.Linear(self.width, self.output_dim)
        self.loss_function = torch.nn.CrossEntropyLoss(reduction="none")
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.scheduler = WarmupLinearSchedule(self.optimizer, 50, 50 * 500)
    def reset_metrics(self):
        self.example_cnt = 0
        self.running_accuracy = 0
        self.running_loss = 0

    def metrics(self):
        return {
            "Number-Examples": self.example_cnt,
            "InfoNCE-Loss": round(self.running_loss / (1 if self.example_cnt < 1 else self.example_cnt), 4),
            "Accuracy": round(self.running_accuracy / (1 if self.example_cnt < 1 else self.example_cnt), 4)
        }

    def forward(self, data):
        x = self.model(data)
        if hasattr(data, "batch"):
            emb = scatter_mean(x, data.batch, dim=0)
        else:
            emb = x.mean(dim=0, keepdim=True)
        emb = self.output(emb)
        norm = torch.norm(emb, dim=1, keepdim=True) + 1e-8
        emb = emb.div(norm.expand_as(emb))

        batch_size = data.num_graphs // 2

        emb = emb.view(batch_size, 2, -1)
        out1 = emb[:, 0, :]
        out2 = emb[:, 1, :]
        sims = torch.matmul(out1, out2.transpose(0, 1)) / self.tau
        gt = torch.arange(0, batch_size, dtype=torch.long, device=sims.get_device())
        # print(sims.shape, gt.shape)
        # print(gt)
        loss = self.loss_function(sims, gt)
        
        acc = (gt == torch.max(sims, dim=1)[1])
        loss = torch.mean(loss)
        actual_batch_size = sims.shape[0]
        self.running_accuracy += acc.sum().item()
        self.running_loss += loss.item()*sims.shape[0]
        self.example_cnt += actual_batch_size

        self.target = loss
        return emb