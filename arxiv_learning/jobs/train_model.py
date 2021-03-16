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
import arxiv_learning.data.heuristics.masking
from arxiv_learning.nn.softnormalization import SoftNormalization
from arxiv_learning.data.load_mathml import VOCAB_SYMBOLS
from torch_scatter import scatter_min, scatter_mean
from arxiv_learning.nn.graph_cnn import GraphCNN
import arxiv_learning.nn.loss as losses
import arxiv_learning.nn.lars
from arxiv_learning.nn.heads import *
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

    #net = GraphCNN(width=width, layer=GraphConv, args=(width, ))
    net = GraphCNN(width=width, layer=GatedGraphConv, args=(4,))
    # net = GraphCNN(width=width, layer=FeaStConv, args=(width, 6),)
    #net = GraphCNN(width=width, layer=GATConv, args=(width, 6),)

    # torch.set_default_dtype(torch.float16)
    net = net.to(device)

    heuristics = {
        "same_paper": {
            "data_set": arxiv_learning.data.heuristics.context.SamePaper,
            # "data_set": arxiv_learning.data.heuristics.equations.EqualityHeuristic,
            "head": InfoNCEHead,
            "head_kwargs": {"width": width, "output_dim": 256, "tau":0.01}

        },
        "mask": {
           "data_set": arxiv_learning.data.heuristics.masking.MaskingHeuristic,
            "head": MaskedHead,
            "head_kwargs": {"width": width}
        }
    }
    batch_size = 1024
    trainloader = RayManager(total=1024, blowout=64, custom_heuristics=heuristics, batch_size=batch_size, data_augmentation=data_augmentation)
    testloader = RayManager(test=True, total=256, blowout=16, custom_heuristics=heuristics, batch_size=512, data_augmentation=data_augmentation)

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
        for loader in [trainloader, testloader]:
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
        # r1, r10, r100 = test_model(net, testloader, heads["equalities"], 256, device)
        # sacred_experiment.log_scalar("test.recall@1", r1)
        # sacred_experiment.log_scalar("test.recall@10", r10)
        # sacred_experiment.log_scalar("test.recall@100", r100)

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
