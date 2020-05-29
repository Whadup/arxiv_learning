import datetime
import torch
import torch.optim as optim
from torch_geometric.nn import GatedGraphConv
import tqdm
from sacred import Experiment
from sacred.observers import FileStorageObserver
from arxiv_learning.data.dataloader import RayManager
import arxiv_learning.data.heuristics.json_dataset
import arxiv_learning.data.heuristics.equations
import arxiv_learning.data.heuristics.context
from arxiv_learning.nn.softnormalization import SoftNormalization
from torch_scatter import scatter_min, scatter_mean
from arxiv_learning.nn.graph_cnn import GraphCNN
import arxiv_learning.nn.loss as losses
from arxiv_learning.nn.scheduler import WarmupLinearSchedule
from numpy import round

def replace_tabs(s):
    return ' '.join('%-4s' % item for item in s.split('\t'))

class Head(torch.nn.Module):
    def __init__(self, model, lr=0.0001, scheduler=None, scheduler_kwargs=None):
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
    

def train_model(batch_size, learning_rate, epochs, model, sacred_experiment):
    """
    train a model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    width = 256

    net = GraphCNN(width=width, layer=GatedGraphConv, args=(2,))
    # criterion = losses.HistogramLoss(weighted=False).to(device)
    # torch.set_default_dtype(torch.float16)
    net = net.to(device)

    heuristics = {
        "equalities": {
            "data_set": arxiv_learning.data.heuristics.equations.EqualityHeuristic,
            "head": RelationTypeHead,
            "head_kwargs": {"width": width, "hidden_dim": 128}

        },
        "same_paper": {
            "data_set": arxiv_learning.data.heuristics.context.SamePaper,
            "head": HistogramLossHead,
            "head_kwargs": {"width": width, "output_dim": 64}
        },
        "same_section": {
            "data_set": arxiv_learning.data.heuristics.context.SamePaper,
            "head": HistogramLossHead,
            "head_kwargs": {"width": width, "output_dim": 64}
        }
    }

    trainloader = RayManager(total=10000, blowout=8, custom_heuristics=heuristics)
    testloader = RayManager(test=True, total=100, blowout=8, custom_heuristics=heuristics)

    loss_log_interval = 250

    heads = {
        dataset : head["head"](net, **head.get("head_kwargs", {})).to(device) for dataset, head in heuristics.items()
    }
    bars = {}
    for i, head in enumerate(heads):
        bars[head] = tqdm.tqdm(bar_format=" ⮑  " + "{:<14}".format(head+":") + " {desc}", position=1 + i)
        bars[head].refresh()
    for epoch in range(epochs):  # loop over the dataset multiple times
        for loader in [trainloader, testloader]:
            with tqdm.tqdm(total=loader.total * 128, ncols=120, dynamic_ncols=False, smoothing=0.1, position=0) as pbar:
                pbar.set_description("epoch {}/{}".format(epoch+1, epochs))
                if loader == testloader:
                    net = net.eval()
                else:
                    net = net.train()

                for head in heads.values():
                    head.reset_metrics()

                for i, data in enumerate(loader):
                    dataset, data = data
                    data = data.to(device)
                    heads[dataset].forward(data)
                   
                    if loader is trainloader:
                        heads[dataset].backward()

                    pbar.update(128)
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
            if i % loss_log_interval != loss_log_interval - 1:
                for dataset, head in heads.items():
                    for metric, value in head.metrics().items():
                        if loader is testloader:
                            sacred_experiment.log_scalar("test.{}.{}".format(dataset, metric), value)
                        else:
                            sacred_experiment.log_scalar("train.{}.{}".format(dataset, metric), value)
        net.save_checkpoint(epoch)
        sacred_experiment.add_artifact(net.checkpoint_string.format(epoch))
        # scheduler.step()
    net.to("cpu")
    net.save()
    for bar in bars.values():
        bar.close()
        print()

SACRED_EXPERIMENT = Experiment(name="train_model_{}".format(datetime.date.today().isoformat()))
MODEL_PATH = "checkpoints/"
SACRED_EXPERIMENT.observers.append(FileStorageObserver.create(MODEL_PATH))

@SACRED_EXPERIMENT.capture
def train(batch_size, learning_rate, epochs, model, _run):
    train_model(batch_size, learning_rate, epochs, model, SACRED_EXPERIMENT)
    print("FINISHED RUN", _run._id)

@SACRED_EXPERIMENT.config
def hyperparamters():
    batch_size = 256
    learning_rate = 0.0001
    epochs = 20
    model = "graph"


@SACRED_EXPERIMENT.automain
def main():
    # pylint: disable=no-value-for-parameter
    train()
