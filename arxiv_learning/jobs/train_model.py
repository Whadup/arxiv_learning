import argparse
import datetime
import torch
import os
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch_geometric.nn import GatedGraphConv, GraphConv, FeaStConv, GATConv
from torch_scatter import scatter_min
import tqdm
import meticulous
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

def replace_tabs(s):
    return ' '.join('%-4s' % item for item in s.split('\t'))


def construct_model(width, layer, *layer_args):
    return GraphCNN(width=width, layer=layer, args=layer_args)

def train_model(net, width=256, batch_size=512, lr=1e-3, epochs=50, train_steps=1024, test_steps=256, data_augmentation=True, exp=None, **kwargs):
    """
    train a model
    """
    print("IGNORING kwargs", kwargs)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)

    heuristics = {
        "same_paper": {
            "data_set": arxiv_learning.data.heuristics.context.SamePaper,
            # "data_set": arxiv_learning.data.heuristics.equations.EqualityHeuristic,
            "head": InfoNCEHead,
            "head_kwargs": {"width": width, "output_dim": 256, "lr": lr}

        },
        "mask": {
            "data_set": arxiv_learning.data.heuristics.masking.MaskingHeuristic,
            # "data_set": arxiv_learning.data.heuristics.equations.EqualityHeuristic,
            "head": MaskedHead,
            "head_kwargs": {"width": width, "lr": lr}
        }
    }

    trainloader = RayManager(total=train_steps, blowout=32, custom_heuristics=heuristics, batch_size=batch_size, data_augmentation=data_augmentation)
    testloader = RayManager(test=True, total=test_steps, blowout=8, custom_heuristics=heuristics, batch_size=512, data_augmentation=data_augmentation)

    basefile = arxiv_learning.data.heuristics.heuristic.Heuristic().basefile
    vocab_file = os.path.abspath(os.path.join(os.path.split(basefile)[0], "vocab.pickle"))

    if exp is not None:
        with exp.open("vocab.pickle", "wb") as f:
            f.write(open(vocab_file, "rb").read())

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
                    if test_steps <= 0:
                        continue
                else:
                    net = net.train()
                    if train_steps <= 0:
                        continue

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
                                    pass
                                    # if loader is testloader:
                                    #     sacred_experiment.log_scalar("test.{}.{}".format(dataset, metric), value)
                                    # else:
                                    #     sacred_experiment.log_scalar("train.{}.{}".format(dataset, metric), value)
            print()
            for dataset, head in heads.items():
                for metric, value in head.metrics().items():
                    pass
                    # if loader is testloader:
                    #     sacred_experiment.log_scalar("test.{}.{}".format(dataset, metric), value)
                    # else:
                    #     sacred_experiment.log_scalar("train.{}.{}".format(dataset, metric), value)
            # return
        # net.save_checkpoint(epoch)
        if exp is not None:
            with exp.open(net.checkpoint_string.format(epoch), "wb") as f:
                net.save_file(f)
        # sacred_experiment.add_artifact(net.checkpoint_string.format(epoch))
        # r1, r10, r100 = test_model(net, testloader, heads["equalities"], 256, device)
        # sacred_experiment.log_scalar("test.recall@1", r1)
        # sacred_experiment.log_scalar("test.recall@10", r10)
        # sacred_experiment.log_scalar("test.recall@100", r100)

        # scheduler.step()
    net.to("cpu")
    if exp is not None:
        with exp.open(net.save_path, "wb") as f:
            net.save_file(f)
    else:
        net.save()
    for bar in bars.values():
        bar.close()
        print()
    return net







def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=1024,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--width', type=int, default=256,
                        help='hidden layer size (default: 256)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='initial  learning rate for training (default: 1e-3)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs for training (default: 50)')
    parser.add_argument('--train_steps', type=int, default=1024, 
                        help='number of training batches to generate per epoch')
    parser.add_argument('--test_steps', type=int, default=256, 
                        help='number of test batches to generate per epoch')
    meticulous.Experiment.add_argument_group(parser)
    args = parser.parse_args()
    with meticulous.Experiment.from_parser(parser) as exp:
        net = construct_model(args.width, GraphConv, args.width)
        train_model(net, exp=exp, **vars(args))


if __name__ == "__main__":
    main()