"""Transformations to apply on Formulas"""
import torch
# PERMS = []
# NUM_PERMS = 100
# if NUM_PERMS < 10000:
#     print("WARNING: NUMBER OF PERMUATIONS IS TOO SMALL: ", NUM_PERMS)
# for i in range(NUM_PERMS):
#     perm = torch.arange(TAG_SYMBOLS, TAG_SYMBOLS+CONTENT_SYMBOLS-1)
#     for _ in range(np.random.poisson(32)):
#         a, b = np.random.randint(CONTENT_SYMBOLS - 1, size=2)
#         tmp = perm[a]
#         perm[a] = perm[b]
#         perm[b] = tmp
#         PERMS.append(perm)

# def permute(X, mi_index, flips=10):
#     where = X[:, mi_index] == 1
#     perm = PERMS[np.random.choice(len(PERMS))]
#     X[where, TAG_SYMBOLS:TAG_SYMBOLS+CONTENT_SYMBOLS-1] = X[where, :][:, perm]
#     return X

# def subsample(data):
#     from torch_geometric.utils import subgraph
#     from torch_geometric.data import Data
#     interval = torch.round(torch.rand(2) * data.num_nodes)
#     start = int(interval.min().item())
#     end = int(interval.max().item())
#     interval = list(range(start, end))
#     edge_index, edge_attr = subgraph(interval, data.edge_index, data.edge_attr, relabel_nodes=True)
#     new_data = Data(
#         X=data.X[interval],
#         y=torch.FloatTensor([[1, 1]]),
#         edge_index=edge_index,
#         edge_attr=edge_attr)
#     return new_data

