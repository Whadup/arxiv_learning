import torch
import torch.optim as optim
from arxiv_learning.data.load_mathml import VOCAB_SYMBOLS
from arxiv_learning.nn.softnormalization import SoftNormalization
from arxiv_learning.nn.scheduler import WarmupLinearSchedule
from .head import Head

class MaskedHead(Head):
    def __init__(self, model, width=512, **kwargs):
        super().__init__(model, **kwargs)
        self.width = width
        self.output = torch.nn.Linear(self.width, VOCAB_SYMBOLS)
        self.loss = torch.nn.CrossEntropyLoss(reduction="mean")
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.scheduler = WarmupLinearSchedule(self.optimizer, 500, 20 * 10000)

    def reset_metrics(self):
        self.token_cnt = 0
        self.running_loss = 0

    def metrics(self):
        return {
            "Tokens": self.token_cnt,
            "Cross-Entropy": round(self.running_loss / (1 if self.token_cnt < 1 else self.token_cnt), 4)
        }

    def forward(self, data):
        x = self.model(data)
        preds = self.output(x)
        y = data.y
        # print(preds.shape, y.shape)
        l = self.loss(preds, y.view(-1))
        self.target = l
        self.running_loss += l.item() * (y >= 0).sum().item()
        self.token_cnt += (y >= 0).sum().item()