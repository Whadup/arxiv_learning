import torch
import torch.nn as nn
class SoftNormalization(nn.Module):
    """
    Normalizes Embeddings by μ + σ where μ,σ are batchwise-computed mean and standard deviation
    of the norms of the embeddings
    """
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a soft normalizer"""
        super(SoftNormalization, self).__init__()
        self.register_buffer('running_mean', torch.zeros(1))
        self.register_buffer('running_var', torch.ones(1))

        self.variance_epsilon = eps

    def forward(self, x):
        if self.training:
            norms = torch.norm(x, dim=1)
            u = norms.mean() + self.variance_epsilon
            s = norms.var(unbiased=False)
            self.running_mean = self.running_mean * 0.99 + 0.01 * u.detach()
            self.running_var = self.running_var * 0.99 + 0.01 * s.detach()
            #if s>0.1:
            #       x = x / (u + s)
            #else:
            #       x = x / (u + 0.1)
            x = x / (u + torch.sqrt(torch.clamp(s, min=self.variance_epsilon)))
        else:
                x = x / (self.running_mean + torch.sqrt(torch.clamp(self.running_var, min=self.variance_epsilon)))
        return x
