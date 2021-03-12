import torch
import torch.optim as optim
from arxiv_learning.nn.softnormalization import SoftNormalization
from arxiv_learning.nn.scheduler import WarmupLinearSchedule
from .head import Head
class RelationTypeHead(Head):
    def __init__(self, model, width=512, hidden_dim=64, num_classes=2):
        super().__init__(model)
        self.width = width
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.hidden = torch.nn.Linear(self.width, self.hidden_dim)
        self.output = torch.nn.Linear(3 * self.hidden_dim, self.num_classes)
        self.bn = torch.nn.BatchNorm1d(3 * self.hidden_dim)
        self.loss = torch.nn.CrossEntropyLoss(reduction="none")
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.scheduler = WarmupLinearSchedule(self.optimizer, 500, 20 * 10000)

    def reset_metrics(self):
        self.example_cnt = 0
        self.running_accuracy = 0
        self.running_loss = 0

    def metrics(self):
        return {
            "Number-Examples": self.example_cnt,
            "Cross-Entropy": round(self.running_loss / (1 if self.example_cnt < 1 else self.example_cnt), 4),
            "Accuracy": round(self.running_accuracy / (1 if self.example_cnt < 1 else self.example_cnt), 4)
        }

    def forward(self, data):
        x = self.model(data)
        if hasattr(data, "batch"):
            emb = scatter_mean(x, data.batch, dim=0)
        else:
            emb = x.mean(dim=0, keepdim=True)
        emb = self.hidden(emb)

        batch_size = data.num_graphs // 3
        emb = emb.view(batch_size, 3, -1)
        out1 = emb[:, 0, :]
        out2 = emb[:, 1, :]
        out3 = emb[:, 2, :]

        features1 = torch.cat([out1, out1 * out2, out2], dim=1)
        features2 = torch.cat([out1, out1 * out3, out3], dim=1)
        features = torch.cat([features1, features2], dim=0)
        # print(features1.shape, out1.shape)
        features = torch.nn.functional.relu(self.bn(features))
        preds = self.output(features)
        # preds2 = self.output(features2)

        targets = data.y.view(batch_size, 3)
        targets = torch.cat([targets[:, 0], targets[:, 2]], dim=0)


        # loss = self.loss(preds1, targets[:, 0]) + self.loss(preds2, targets[:, 2])
        # acc = (preds1.argmax(dim=1) == targets[:, 0]).sum() + (preds2.argmax(dim=1) == targets[:, 2]).sum()

        loss = self.loss(preds, targets)
        acc = (preds.argmax(dim=1) == targets).sum()


        self.running_accuracy += acc.item() / 2.0
        self.running_loss += loss.sum().item() / 2.0
        self.example_cnt += batch_size

        self.target = loss.mean()