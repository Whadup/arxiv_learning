"""Implements the instance-dependent margin triples loss"""
import torch


def triple_loss(sim, dissim1, dissim2, margin, anchor_swap):
    """Compute a Margin Ranking Loss where the margins are part of the data"""
    # l = anchor_swap * margin - (sim - torch.max(dissim1, dissim2)) + \
    # 	(1.0 - anchor_swap) * (margin - (sim - dissim1))
    l = anchor_swap * torch.clamp(margin - (sim - torch.max(dissim1, dissim2)), min=0.0) + \
        (1.0 - anchor_swap) * torch.clamp(margin - (sim - dissim1), min=0)
    return l # / margin


class HistogramLoss(torch.nn.Module):
    """
    Source: https://papers.nips.cc/paper/6464-learning-deep-embeddings-with-histogram-loss.pdf
    """
    def __init__(self, R=64, weighted=False):
        super(HistogramLoss, self).__init__()
        self.R = R
        if weighted:
            weight = torch.arange(R, 0, -1) / (1.0 * R)
        else:
            weight = torch.ones(R)
        self.register_buffer('weight', weight)
        lam = 2.0/(self.R - 1)
        self.lam = lam
        t1 = torch.arange(R) * lam - 1 - lam
        t2 = t1 + 2.0/(R-1)
        t3 = t2 + 2.0/(R-1)
        self.register_buffer('t1', t1)
        self.register_buffer('t2', t2)
        self.register_buffer('t3', t3)

    def forward(self, sim, dissim1, dissim2, margin, anchor_swap):
        #anchor swap
        dissim = dissim1 #torch.where(anchor_swap == 1, torch.max(dissim1, dissim2), dissim1).clamp(-1, 1)

        sim = sim.clamp(-1, 1)
        dissim = dissim.clamp(-1, 1)
        #broadcastable shapes:
        sim = sim.view(-1, 1)
        dissim = dissim.view(-1, 1)
        hist_minus = (((dissim > self.t1) & (dissim < self.t2)) * (dissim - self.t1)/self.lam).sum(dim=0) + \
            (((dissim > self.t2) & (dissim < self.t3)) * (self.t3 - dissim)/self.lam).sum(dim=0)
        hist_plus = (((sim > self.t1) & (sim < self.t2)) * (sim - self.t1)/self.lam).sum(dim=0) + \
            (((sim > self.t2) & (sim < self.t3)) * (self.t3 - sim)/self.lam).sum(dim=0)
        hist_minus /= dissim.shape[0]
        hist_plus /= sim.shape[0]
        hist_plus_cum = torch.cumsum(hist_plus, dim=0)
        hist_minus = hist_minus * self.weight
        # print(hist_plus.dot(hist_plus_cum))
        return hist_plus_cum.dot(hist_minus)


class CorrectedHistogramLoss(torch.nn.Module):
    """
    Source: https://papers.nips.cc/paper/6464-learning-deep-embeddings-with-histogram-loss.pdf
    """
    def __init__(self, p=0.1, R=64, weighted=False):
        super(CorrectedHistogramLoss, self).__init__()
        self.R = R
        self.p = p
        if weighted:
            weight = torch.arange(R, 0, -1) / (1.0 * R)
        else:
            weight = torch.ones(R)
        self.register_buffer('weight', weight)
        lam = 2.0/(self.R - 1)
        self.lam = lam
        t1 = torch.arange(R) * lam - 1 - lam
        t2 = t1 + 2.0/(R-1)
        t3 = t2 + 2.0/(R-1)
        self.register_buffer('t1', t1)
        self.register_buffer('t2', t2)
        self.register_buffer('t3', t3)

    def forward(self, sim, dissim1, dissim2, margin, anchor_swap):
        #anchor swap
        # dissim = torch.where(anchor_swap == 1, torch.max(dissim1, dissim2), dissim1).clamp(-1, 1)

        sim = sim.clamp(-1, 1)
        dissim = dissim1.clamp(-1, 1)
        #broadcastable shapes:
        sim = sim.view(-1, 1)
        dissim = dissim.view(-1, 1)
        hist_minus = (((dissim > self.t1) & (dissim < self.t2)) * (dissim - self.t1)/self.lam).sum(dim=0) + \
            (((dissim > self.t2) & (dissim < self.t3)) * (self.t3 - dissim)/self.lam).sum(dim=0)
        hist_plus = (((sim > self.t1) & (sim < self.t2)) * (sim - self.t1)/self.lam).sum(dim=0) + \
            (((sim > self.t2) & (sim < self.t3)) * (self.t3 - sim)/self.lam).sum(dim=0)
        hist_minus /= dissim.shape[0]
        hist_plus /= sim.shape[0]
        hist_plus_cum = torch.cumsum(hist_plus, dim=0)
        hist_minus = hist_minus
        hist_minus_cum = torch.cumsum(hist_minus, dim=0)
        return ((1-self.p) * (1-self.p) * hist_plus_cum.dot(hist_minus) -
                (1-self.p) * self.p * hist_plus_cum.dot(hist_plus) -
                (1-self.p) * self.p * hist_minus_cum.dot(hist_minus) +
                self.p * self.p * hist_minus_cum.dot(hist_plus)) / (1 - 4 * self.p + 4 * self.p * self.p)

class ScaledCrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(ScaledCrossEntropyLoss, self).__init__()
        scales = torch.exp(torch.linspace(-10, 0)).view(1, -1)
        self.register_buffer('scales', scales)
        self.loss = torch.nn.CrossEntropyLoss(reduction="none")
    def forward(self, x, y):
        xy = x.view((*x.shape, 1))
        scaled_x = xy * self.scales
        stacked = scaled_x.transpose(1, 2).reshape(-1, self.scales.shape[0])
        losses = self.loss(stacked, torch.repeat_interleave(y, 10))
        losses_a = losses.reshape(x.shape[0], -1)
        return losses_a.mean(dim=0).min()
