import torch
import numpy as np
from arxiv_learning.data.load_mathml import VOCAB_SYMBOLS

def prepare_permutations(vocabulary):
    math_identifiers = []
    for x in vocabulary:
        if x.startswith("mi_"):
            math_identifiers.append(vocabulary[x])
    perms = []
    for i in range(10000):
        perm = torch.arange(VOCAB_SYMBOLS)
        for _ in range(np.random.poisson(32)):
            a, b = np.random.choice(math_identifiers, size=2)
            tmp = perm[a]
            perm[a] = perm[b]
            perm[b] = tmp
        perms.append(perm)
        # print(perm)
    return perms



def permute(X, perms):
    from torch_geometric.data import Data
    perm = perms[np.random.choice(len(perms))]
    if isinstance(X, Data):
        X.x = perm[X.x]
        return X
    else:
        return perm[X]
# X[where, TAG_SYMBOLS:TAG_SYMBOLS+CONTENT_SYMBOLS-1] = X[where, :][:, perm]
# 	return X

def break_order(X, proba=0.15):
    mask = torch.rand((X.pos.shape[0], 1)) < proba
    rand_pos = torch.randint_like(X.pos, 10)
    X.pos = torch.where(mask, X.pos, rand_pos)
    return X