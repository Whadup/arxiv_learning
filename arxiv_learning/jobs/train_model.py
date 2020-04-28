import datetime
import torch
import torch.optim as optim
import tqdm
from sacred import Experiment
from sacred.observers import FileStorageObserver
from arxiv_learning.data.dataloader import RayManager
import arxiv_learning.data.heuristics.json_dataset
from arxiv_learning.nn.graph_cnn import GraphCNN
import arxiv_learning.nn.loss as losses
from arxiv_learning.nn.scheduler import WarmupLinearSchedule



def train_model(batch_size, learning_rate, epochs, model, sacred_experiment):
    """
    train a model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = GraphCNN() #layer=GatedGraphConv, args=(3,))
    criterion = losses.HistogramLoss(weighted=False).to(device)
    # torch.set_default_dtype(torch.float16)
    net = net.to(device)
    trainloader = RayManager(total=500, blowout=16)
    # testloader = ml.dataloader.RayManager(total=500, blowout=16, test=True)
    testloader = RayManager(
        # custom_heuristics={
        #  "json_heuristic": {
        #      "data_set" : arxiv_learning.data.heuristics.json_dataset.JsonDataset,
        #      "head": None
        #  }}, 
         total=1000, blowout=10, test=True)
    # trainloader = ml.dataloader.data_loader(batch_size, curriculum=curriculum, graph=model == "graph")
    # testloader = ml.dataloader.data_loader(batch_size, curriculum=False, test=True, graph=model == "graph")
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
            with tqdm.tqdm(total=loader.total*128, ncols=140, smoothing=0.1) as pbar:
                for i, data in enumerate(loader):
                    if model == "graph":
                        dataset, data = data
                        data = data.to(device)
                        # print(data)
                        if loader == trainloader:
                            dist_sim, dist_dissim1, dist_dissim2, mask_loss = net.forward3(data)
                        else:
                            dist_sim, dist_dissim1, dist_dissim2 = net.forward3(data)

                    loss = criterion(dist_sim, dist_dissim1.view(-1), dist_dissim2, torch.ones(1), torch.zeros(1))
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
                        print()
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
