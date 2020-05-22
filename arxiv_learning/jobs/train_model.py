import datetime
import os
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
import arxiv_learning.data.heuristics.heuristic
from arxiv_learning.data.load_mathml import VOCAB_SYMBOLS
from arxiv_learning.nn.graph_cnn import GraphCNN
import arxiv_learning.nn.loss as losses
from arxiv_learning.nn.scheduler import WarmupLinearSchedule
from arxiv_learning.flags import MASKED_LANGUAGE_TRAINING
#7b359659bee4917525cca7909d301cbd20fb1e8e

def train_model(batch_size, learning_rate, epochs, masked_language_training, sacred_experiment):
    """
    train a model
    """
    global MASKED_LANGUAGE_TRAINING
    MASKED_LANGUAGE_TRAINING = masked_language_training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = GraphCNN()
    criterion = losses.HistogramLoss(weighted=False).to(device)
    # criterion = losses.triple_loss
    # torch.set_default_dtype(torch.float16)
    net = net.to(device)
    heuristics = {
            "same_paper": {
                "data_set": arxiv_learning.data.heuristics.context.SamePaper,
                "head": None
            },
            "same_section": {
                "data_set": arxiv_learning.data.heuristics.context.SameSection,
                "head": None
            },
        }
    trainloader = RayManager(total=1000, blowout=20, custom_heuristics=heuristics)
    basefile = arxiv_learning.data.heuristics.heuristic.Heuristic().basefile
    vocab_file = os.path.abspath(os.path.join(os.path.split(basefile)[0], "vocab.pickle"))
    sacred_experiment.info["data_file"] = basefile
    sacred_experiment.info["vocab_file"] = vocab_file
    sacred_experiment.info["vocab_dim"] = VOCAB_SYMBOLS
    sacred_experiment.add_artifact(vocab_file)

    testloader = RayManager(test=True, total=100, blowout=10, custom_heuristics=heuristics)

    loss_log_interval = 1000

    # criterion = ml.loss.triple_loss
    # criterion = ml.loss.CorrectedHistogramLoss(weighted=False).to(device)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = WarmupLinearSchedule(optimizer, 1000, \
        epochs * trainloader.total)
    for epoch in range(epochs):  # loop over the dataset multiple times
        # if epoch == 10:
        # 	trainloader = ml.dataloader.data_loader(batch_size * 4, curriculum=curriculum, graph=model=="graph")
        for loader in [trainloader, testloader]:
            if loader == testloader:
                net = net.eval()
            else:
                net = net.train()
            running_loss = 0.0
            running_accuracy = 0.0
            running_percentage = 0.0
            smooth_running_loss = 0.0
            multitask_loss = 0.0
            example_cnt = 0
            print("MASKED", MASKED_LANGUAGE_TRAINING)
            with tqdm.tqdm(total=loader.total*128, ncols=140, smoothing=0.1) as pbar:
                for i, data in enumerate(loader):
                    dataset, data = data
                    data = data.to(device)
                    # print(data)
                    if loader == trainloader:
                        dist_sim, dist_dissim1, dist_dissim2, mask_loss = net.forward3(data)
                    else:
                        dist_sim, dist_dissim1, dist_dissim2 = net.forward3(data)

                    loss = criterion(dist_sim, dist_dissim1.view(-1), dist_dissim2, torch.ones(1).cuda(), torch.zeros(1).cuda())
                    if len(dist_dissim1.shape) > 1:
                        dist_dissim2 = dist_dissim1
                        dist_dissim1 = dist_dissim1.max(dim=1)[0]
                    percentage = (torch.clamp(loss, min=0.0)).sum().item()
                    acc = dist_sim >  dist_dissim1
                    running_accuracy += acc.sum().item()
                    running_percentage += percentage
                    # loss = loss * loss # squared hinge instead of hinge
                    loss = torch.mean(loss)

                    actual_batch_size = dist_sim.shape[0]
                    running_loss += loss.item()*dist_sim.shape[0]
                    smooth_running_loss = loss if i == 0 else \
                        smooth_running_loss * 0.99 + loss.item() * 0.01
                    example_cnt += actual_batch_size
                    if loader is trainloader:
                        if MASKED_LANGUAGE_TRAINING:
                            
                            loss = loss + mask_loss
                        multitask_loss += loss.item()*dist_sim.shape[0]
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()
                        # for name, param in net.named_parameters():
                        # 	print("=================")
                        # 	print(name)
                        # 	print(param)

                    pbar.update(actual_batch_size)
                    pbar.set_description("epoch [{}/{}] - loss {:.4f} ({:.4f}) [{:.3f}, {:.3f}%]"
                        .format(epoch+1, epochs, running_loss/example_cnt, smooth_running_loss,
                            multitask_loss/example_cnt, running_accuracy/example_cnt))
                    if i % loss_log_interval == loss_log_interval - 1:
                        if loader is testloader:
                            sacred_experiment.log_scalar("test.loss", running_loss / example_cnt)
                            sacred_experiment.log_scalar("test.accuracy", running_accuracy/example_cnt)
                            sacred_experiment.log_scalar("test.multitask", multitask_loss/example_cnt)
                        else:
                            sacred_experiment.log_scalar("training.loss", running_loss / example_cnt)
                            sacred_experiment.log_scalar("training.accuracy", running_accuracy/example_cnt)
                            sacred_experiment.log_scalar("training.multitask", multitask_loss/example_cnt)
                        # print()
                        # ml.histogram.print_histogram(dist_sim, compute_min=-1, compute_max=1)
                        # ml.histogram.print_histogram(dist_dissim2.view(-1), compute_min=-1, compute_max=1)
            if loader is testloader:
                sacred_experiment.log_scalar("test.loss", running_loss / example_cnt)
                sacred_experiment.log_scalar("test.accuracy", running_accuracy/example_cnt)
                sacred_experiment.log_scalar("test.multitask", multitask_loss/example_cnt)
            else:
                sacred_experiment.log_scalar("training.loss", running_loss / example_cnt)
                sacred_experiment.log_scalar("training.accuracy", running_accuracy/example_cnt)
                sacred_experiment.log_scalar("training.multitask", multitask_loss/example_cnt)
        net.save_checkpoint(epoch)
        sacred_experiment.add_artifact(net.checkpoint_string.format(epoch))
        # scheduler.step()
    net.to("cpu")
    net.save()

SACRED_EXPERIMENT = Experiment(name="train_model_{}".format(datetime.date.today().isoformat()))
MODEL_PATH = "checkpoints/"
SACRED_EXPERIMENT.observers.append(FileStorageObserver.create(MODEL_PATH))

@SACRED_EXPERIMENT.capture
def train(batch_size, learning_rate, epochs, masked_language_training, _run):
    train_model(batch_size, learning_rate, epochs, masked_language_training, SACRED_EXPERIMENT)
    print("FINISHED RUN", _run._id)

@SACRED_EXPERIMENT.config
def hyperparamters():
    batch_size = 256
    learning_rate = 0.0001
    # learning_rate = 0.001
    epochs = 20
    masked_language_training = True


@SACRED_EXPERIMENT.automain
def main():
    # pylint: disable=no-value-for-parameter
    train()
