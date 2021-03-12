import torch
class Head(torch.nn.Module):
    def __init__(self, model, lr=0.001, scheduler=None, scheduler_kwargs=None):
        super().__init__()
        self.model = model
        self.target = None
        self.optimizer = None
        
        self.lr = lr
        self.reset_metrics()
    def forward(self, data):
        pass
    def backward(self):
        self.target.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        if self.scheduler:
            self.scheduler.step()
    def metrics(self):
        pass
    def reset_metrics(self):
        pass