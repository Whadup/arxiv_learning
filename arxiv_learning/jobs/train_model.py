import datetime
import torch
import os
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch_geometric.nn import GatedGraphConv, GraphConv, FeaStConv, GATConv
from torch_scatter import scatter_min
import tqdm
from sacred import Experiment
from sacred.observers import FileStorageObserver
from arxiv_learning.data.dataloader import RayManager
import arxiv_learning.data.heuristics.json_dataset
import arxiv_learning.data.heuristics.equations
import arxiv_learning.data.heuristics.shuffle
import arxiv_learning.data.heuristics.context
from arxiv_learning.nn.softnormalization import SoftNormalization
from arxiv_learning.data.load_mathml import VOCAB_SYMBOLS
from torch_scatter import scatter_min, scatter_mean
from arxiv_learning.nn.graph_cnn import GraphCNN
import arxiv_learning.nn.loss as losses
import arxiv_learning.nn.lars
from arxiv_learning.nn.scheduler import WarmupLinearSchedule
from arxiv_learning.jobs.gitstatus import get_repository_status
from numpy import round
import numpy as np
import contextlib
# from quantiler.quantiler import Quantiler

@contextlib.contextmanager
def null():
    yield

def replace_tabs(s):
    return ' '.join('%-4s' % item for item in s.split('\t'))

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

class MaskedHead(Head):
    def __init__(self, model, width=512):
        super().__init__(model)
        self.width = width
        self.output = torch.nn.Linear(self.hidden_dim, VOCAB_SYMBOLS)
        self.loss = torch.nn.CrossEntropyLoss(reduction="none")
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.scheduler = WarmupLinearSchedule(self.optimizer, 500, 20 * 10000)

    def reset_metrics(self):
        self.example_cnt = 0
        self.running_loss = 0

    def metrics(self):
        return {
            "Number-Examples": self.example_cnt,
            "Cross-Entropy": round(self.running_loss / (1 if self.example_cnt < 1 else self.example_cnt), 4)
        }

    def forward(self, data):
        x = self.model(data)
        y = data.y

class BinaryClassificationHead(Head):
    def __init__(self, model, width=512, hidden_dim=64, num_classes=2):
        super().__init__(model)
        self.width = width
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.hidden = torch.nn.Linear(self.width, self.hidden_dim)
        self.output = torch.nn.Linear( self.hidden_dim, self.num_classes)
        self.bn = torch.nn.BatchNorm1d(self.hidden_dim)
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
        targets = data.y
        x = self.model(data)
        if hasattr(data, "batch"):
            emb = scatter_mean(x, data.batch, dim=0)
        else:
            emb = x.mean(dim=0, keepdim=True)
        emb = torch.nn.functional.relu(self.hidden(emb))
        preds = self.output(emb)
        batch_size = data.num_graphs

        # loss = self.loss(preds1, targets[:, 0]) + self.loss(preds2, targets[:, 2])
        # acc = (preds1.argmax(dim=1) == targets[:, 0]).sum() + (preds2.argmax(dim=1) == targets[:, 2]).sum()

        loss = self.loss(preds, targets)
        acc = (preds.argmax(dim=1) == targets).sum()


        self.running_accuracy += acc.item()
        self.running_loss += loss.sum().item()
        self.example_cnt += batch_size

        self.target = loss.mean()

SACRED_EXPERIMENT = Experiment(name="train_model_{}".format(datetime.date.today().isoformat()))
MODEL_PATH = "/data/s1/pfahler/arxiv_v2/checkpoints/"
from sacred.observers import MongoObserver

# SACRED_EXPERIMENT.observers.append(MongoObserver.create(url='mongodb://lukas:lukas@s876pn04:27017'))
SACRED_EXPERIMENT.observers.append(FileStorageObserver.create(MODEL_PATH))


def test_model(model, test_loader, head, batch_size, device="cuda"):
    from annoy import AnnoyIndex
    index = AnnoyIndex(head.output_dim, "dot")
    # model = model.eval()
    head = head.eval()
    total = 0
    with tqdm.tqdm(total=test_loader.total * batch_size, ncols=120, dynamic_ncols=False, smoothing=0.1, position=0) as pbar:
        pbar.set_description("testing")
        with torch.no_grad():
            for data in test_loader:
                dataset, data = data
                data = data.to(device)
                embeddings = head.forward(data)
                for i, e in enumerate(embeddings.cpu().numpy()):
                    index.add_item(total * 2, e[0, :])
                    index.add_item(total * 2 + 1, e[1, :])
                    total += 1
                pbar.update(embeddings.shape[0] * 2)
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


    # model = model.train()
    head = head.train()
    return recall_at_1, recall_at_10, recall_at_100

@SACRED_EXPERIMENT.capture
def train_model(batch_size, learning_rate, epochs, masked_language_training, data_augmentation, sacred_experiment):
    """
    train a model
    """
    global MASKED_LANGUAGE_TRAINING
    MASKED_LANGUAGE_TRAINING = masked_language_training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    width = 256

    # net = GraphCNN(width=width, layer=GraphConv, args=(width, ))
    net = GraphCNN(width=width, layer=GatedGraphConv, args=(5,))
    # net = GraphCNN(width=width, layer=FeaStConv, args=(width, 6),)
    # net = GraphCNN(width=width, layer=GATConv, args=(width, 6),)

    # torch.set_default_dtype(torch.float16)
    net = net.to(device)

    heuristics = {
        "equalities": {
            # "data_set": arxiv_learning.data.heuristics.context.SamePaper,
            "data_set": arxiv_learning.data.heuristics.equations.EqualityHeuristic,
            "head": InfoNCEHead,
            "head_kwargs": {"width": width, "output_dim": 256}

        },
    }
    batch_size = 1024
    trainloader = RayManager(total=1024, blowout=30, custom_heuristics=heuristics, batch_size=batch_size, data_augmentation=data_augmentation)
    testloader = RayManager(test=True, total=256, blowout=20, custom_heuristics=heuristics, batch_size=1024, data_augmentation=data_augmentation)

    basefile = arxiv_learning.data.heuristics.heuristic.Heuristic().basefile
    vocab_file = os.path.abspath(os.path.join(os.path.split(basefile)[0], "vocab.pickle"))


    sacred_experiment.info["data_file"] = basefile
    sacred_experiment.info["vocab_file"] = vocab_file
    sacred_experiment.info["vocab_dim"] = VOCAB_SYMBOLS
    sacred_experiment.add_artifact(vocab_file)

    loss_log_interval = 250

    heads = {
        dataset : head["head"](net, **head.get("head_kwargs", {})).to(device).train() for dataset, head in heuristics.items()
    }
    bars = {}
    for i, head in enumerate(heads):
        bars[head] = tqdm.tqdm(bar_format=" ⮑  " + "{:<14}".format(head+":") + " {desc}", position=1 + i)
        bars[head].refresh()
    for epoch in range(epochs):  # loop over the dataset multiple times
        for loader in [trainloader]:
            with tqdm.tqdm(total=loader.total * batch_size, ncols=120, dynamic_ncols=False, smoothing=0.1, position=0) as pbar:

                pbar.set_description("epoch {}/{}".format(epoch+1, epochs))
                if loader == testloader:
                    net = net.eval()
                else:
                    net = net.train()

                for head in heads.values():
                    head.reset_metrics()
                with (torch.enable_grad() if loader == trainloader else torch.no_grad()):
                    for i, data in enumerate(loader):
                        dataset, data = data
                        data = data.to(device)
                        heads[dataset].forward(data)
                    
                        if loader is trainloader:
                            heads[dataset].backward()

                        pbar.update(batch_size)
                        # pbar.set_description("epoch [{}/{}] - loss {:.4f} ({:.4f}) [{:.3f}, {:.3f}%]"
                        #     .format(epoch+1, epochs, running_loss/example_cnt, smooth_running_loss,
                        #         multitask_loss/example_cnt, running_accuracy/example_cnt))
                        bars[dataset].desc = replace_tabs("\t".join(["{:18}{:.4f}".format("{}:".format(key), value) for key, value in heads[dataset].metrics().items()]))
                        bars[dataset].refresh()
                        if i % loss_log_interval == loss_log_interval - 1:
                            for dataset, head in heads.items():
                                for metric, value in head.metrics().items():
                                    if loader is testloader:
                                        sacred_experiment.log_scalar("test.{}.{}".format(dataset, metric), value)
                                    else:
                                        sacred_experiment.log_scalar("train.{}.{}".format(dataset, metric), value)
            print()
            for dataset, head in heads.items():
                for metric, value in head.metrics().items():
                    if loader is testloader:
                        sacred_experiment.log_scalar("test.{}.{}".format(dataset, metric), value)
                    else:
                        sacred_experiment.log_scalar("train.{}.{}".format(dataset, metric), value)
            # return
        net.save_checkpoint(epoch)
        sacred_experiment.add_artifact(net.checkpoint_string.format(epoch))
        r1, r10, r100 = test_model(net, testloader, heads["equalities"], 256, device)
        sacred_experiment.log_scalar("test.recall@1", r1)
        sacred_experiment.log_scalar("test.recall@10", r10)
        sacred_experiment.log_scalar("test.recall@100", r100)

        # scheduler.step()
    net.to("cpu")
    net.save()
    for bar in bars.values():
        bar.close()
        print()




@SACRED_EXPERIMENT.capture
def train(batch_size, learning_rate, epochs, masked_language_training, data_augmentation, _run):
    SACRED_EXPERIMENT.info["gitstatus"] = get_repository_status()
    train_model(batch_size, learning_rate, epochs, masked_language_training, data_augmentation, SACRED_EXPERIMENT)
    print("FINISHED RUN", _run._id)

@SACRED_EXPERIMENT.config
def hyperparamters():
    batch_size = 256
    learning_rate = 0.0001
    # learning_rate = 0.001
    epochs = 50
    masked_language_training = True
    data_augmentation = True


@SACRED_EXPERIMENT.automain
def main():
    # pylint: disable=no-value-for-parameter
    train()
