import torch
import torch.optim as optim
from arxiv_learning.nn.softnormalization import SoftNormalization
from arxiv_learning.nn.scheduler import WarmupLinearSchedule
from .head import Head

class BYOLHead(Head):
    def __init__(self, model, width=512, output_dim=64, tau=0.05):
        from copy import deepcopy
        super().__init__(model)
        self.width = width
        self.output_dim = output_dim
        self.tau = tau
        self.eta = 512
        self.output = torch.nn.Linear(self.width, self.output_dim, bias=False)
        self.mlp1 = torch.nn.Linear(self.output_dim, 1024, bias=False)
        self.mlp2 = torch.nn.Linear(1024, self.output_dim, bias=False)
        self.mlpbn = torch.nn.BatchNorm1d(1024)
        self.normalizer = torch.nn.BatchNorm1d(self.width)
        print(list(self.named_parameters()))
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)
        # self.optimizer = arxiv_learning.nn.lars.LARS(self.parameters(), lr=3.2, eta=1e-3, )
        
        self.scheduler = None #WarmupLinearSchedule(self.optimizer, 500, 20 * 10000)
        self.target_model = deepcopy(model)
        self.target_output = torch.nn.Linear(self.width, self.output_dim, bias=False)
        self.target_normalizer = torch.nn.BatchNorm1d(self.width, track_running_stats=False)#.eval()
        for layer in self.target_model.modules():
            if hasattr(layer, 'reset_parameters'):
                print("reset")
                layer.reset_parameters()

        for param in self.target_model.parameters():
            param.requires_grad = False
        for param in self.target_output.parameters():
            param.requires_grad = False
        for param in self.target_normalizer.parameters():
            param.requires_grad = False
    def reset_metrics(self):
        self.example_cnt = 0
        # self.running_accuracy = 0
        self.running_loss = 0
        

    def metrics(self):
        return {
            "Number-Examples": self.example_cnt,
            "MSE-Loss": round(self.running_loss / (1 if self.example_cnt < 1 else self.example_cnt), 4),
            "Mean-Variance": self.normalizer.running_var.mean(),
            # "Accuracy": round(self.running_accuracy / (1 if self.example_cnt < 1 else self.example_cnt), 4)
        }

    def train(self, par=True):
        print("set BYOL Head to Train=", par)
        print(self.target_model.training, self.model.training, self.output.training, self.target_output.training)
        ret = super().train(par)
        # self.target_model = self.target_model.train(False)
        # self.target_normalizer = self.target_normalizer.train(True)
        # self.target_output = self.target_output.train(par)
        print(self.target_model.training, self.model.training, self.output.training, self.target_output.training)
        return ret

    def uniformize(self, x2, eta=512):
        # eta = 512
        tau = 1
        before = None
        for i in range(4):
            x2 = torch.autograd.Variable(x2, requires_grad=True)
            # print(x2.requires_grad)
            K = 2.0 - 2 * torch.matmul(x2, x2.transpose(0, 1))
            # print(K.requires_grad)
            # K = 2.0 - x2.dot(x2.transpose(0, 1))
            K = torch.exp(-tau * K)
            loss = 1.0 / (x2.shape[0]**2) * K.sum()
            # loss = -torch.log()
            # print(loss.requires_grad)
            if before is None:
                before = loss
            if not K.requires_grad:
                return x2
            loss.backward(retain_graph=True)
            # print("gradnorm:", torch.norm(x2.grad))
            x2 = x2 - eta * x2.grad
            norm = torch.norm(x2, dim=1, keepdim=True) + 1e-8
            x2 = x2.div(norm.expand_as(x2))
            x2 = x2.detach()

        K = 2.0 - 2 * torch.matmul(x2, x2.transpose(0, 1))
        # print(K.requires_grad)
        # K = 2.0 - x2.dot(x2.transpose(0, 1))
        K = torch.exp(-tau * K)
        loss = 1.0 / (x2.shape[0]**2) * K.sum()
        print("\nLoss{} -> {}".format(before, loss))

        return x2.detach()
    def backward(self):
        with torch.autograd.detect_anomaly():
            self.target.backward()
            clip_grad_norm_(self.model.parameters(), 16.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.scheduler:
                self.scheduler.step()

    def forward(self, data):
        # MOVING AVERAGE
        rho = 0.995
        if self.model.training:
            for (i,u), (j,v) in zip(self.target_model.named_parameters(), self.model.named_parameters()):
                if i!=j:
                    print(i, j)
                u.data = rho * u.data + (1.0 - rho) * v.data.detach()
            for u, v in zip(self.target_output.parameters(), self.output.parameters()):
                u.data = rho * u.data + (1.0 - rho) * v.data.detach()
            for u, v in zip(self.target_normalizer.parameters(), self.normalizer.parameters()):
                u.data = rho * u.data + (1.0 - rho) * v.data.detach()

        # INFERENCE
        x = self.model(data)
        x2 = self.target_model(data)
        # print(x.shape[0]/4096.0)
        # x = self.normalizer(x)
        # x2 = self.target_normalizer(x2)

        if hasattr(data, "batch"):
            emb = scatter_mean(x, data.batch, dim=0)
            emb2 = scatter_mean(x2, data.batch, dim=0)
        else:
            emb = x.mean(dim=0, keepdim=True)
            emb2 = x2.mean(dim=0, keepdim=True)


        # PROJECTION 
        emb = self.output(emb)
        emb2 = self.target_output(emb2)

        svs = np.linalg.svd(emb.detach().cpu().numpy(), compute_uv=False)
        # print("\n",np.sum(svs / svs[0]))

        # PREDICTION
        emb = self.mlp1(torch.nn.functional.selu(emb))
        emb = self.mlp2(torch.nn.functional.selu(self.mlpbn(emb)))


        # emb = self.normalizer(emb)
        # emb2 = self.target_normalizer(emb2)

        norm = torch.norm(emb, dim=1, keepdim=True) + 1e-8
        norm2 = torch.norm(emb2, dim=1, keepdim=True) + 1e-8
        
        emb = emb.div(norm.expand_as(emb))
        emb2 = emb2.div(norm2.expand_as(emb2))

        emb2 = emb2.detach()
        emb2 = self.uniformize(emb2, eta=self.eta)
        if self.training:
            self.eta *= 1.0 # 0.9995
        batch_size = data.num_graphs // 2

        emb = emb.view(batch_size, 2, -1)
        emb2 = emb2.view(batch_size, 2, -1)

        out11 = emb[:, 0, :]
        out21 = emb2[:, 1, :]
        out12 = emb[:, 1, :]
        out22 = emb2[:, 0, :]

        sim1 = torch.bmm(out11.view(-1, 1, self.output_dim),
                                out21.view(-1, self.output_dim, 1)).view(-1)
        sim2 = torch.bmm(out12.view(-1, 1, self.output_dim),
                                out22.view(-1, self.output_dim, 1)).view(-1)
        loss = (-2 * sim1  + -2 * sim2).mean()
        # loss = torch.mean((diff1 * diff1).sum(dim=1)) + torch.mean((diff2 * diff2).sum(dim=1))
        actual_batch_size = out11.shape[0] * 2
        self.running_loss += loss.item() * actual_batch_size
        self.example_cnt += actual_batch_size
        self.target = loss

        return emb