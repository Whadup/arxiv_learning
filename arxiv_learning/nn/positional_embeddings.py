import math
import torch

def init_positional_embeddings(layer, normalize=False):
    layer.weight.requires_grad = False
    for i in range(layer.weight.data.shape[0]):
        for j in range(layer.weight.data.shape[1]//2):
            layer.weight.data[i, 2*j] = 0.1*math.sin(i/math.pow(1000.0, 2.0*j/layer.weight.data.shape[1]))
            layer.weight.data[i, 2*j+1] = 0.1*math.cos(i/math.pow(1000.0, 2.0*j/layer.weight.data.shape[1]))
    if normalize:
        norm = torch.max(layer.weight.data.norm(dim=1))
        layer.weight.data /= norm
