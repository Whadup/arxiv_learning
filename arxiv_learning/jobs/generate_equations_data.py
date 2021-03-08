import os
import json
import tqdm
from arxiv_learning.data.dataloader import RayManager
import arxiv_learning.data.heuristics.equations
from .train_model import InfoNCEHead
def generate_data():
    heuristics = {
        "equalities": {
            # "data_set": arxiv_learning.data.heuristics.context.SamePaper,
            "data_set": arxiv_learning.data.heuristics.equations.EqualityHeuristic,
            "head": InfoNCEHead,
            "head_kwargs": {"width": 128, "output_dim": 256}

        },
    }
    trainloader = RayManager(total=128, blowout=8, custom_heuristics=heuristics, batch_size=1024, data_augmentation=False)
    testloader = RayManager(test=True, total=128, blowout=8, custom_heuristics=heuristics, batch_size=1024, data_augmentation=False)


    for i in tqdm.tqdm(trainloader, total=128):
        pass
    for i in tqdm.tqdm(testloader, total=128):
        pass
    equations_train = []
    equations_test = []
    for file in os.listdir("/home/pfahler"):
        if file.startswith("tmp") and file.endswith("_train.eq"):
            with open(os.path.join("/home/pfahler", file)) as f:
                for line in f:
                    equations_train.append(line)
        if file.startswith("tmp") and file.endswith("_test.eq"):
            with open(os.path.join("/home/pfahler", file)) as f:
                for line in f:
                    equations_test.append(line)
    equations_train = list(dict.fromkeys(equations_train))
    equations_test = list(dict.fromkeys(equations_test))
    with open("finetune_inequalities_train.jsonl", "w") as f:
        f.write("".join(equations_train))
    with open("finetune_inequalities_test.jsonl", "w") as f:
        f.write("".join(equations_test))

if __name__ == "__main__":
    generate_data()

