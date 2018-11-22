"""
Train a Dirichlet Mixture Model
"""
import logging

import torch
from torch.distributions import Categorical, Dirichlet
import torch.nn.functional as F
from tqdm import tqdm

from dmm.dmm import DMM

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logging.basicConfig(level=logging.INFO)


# pylint: disable=not-callable
mixture = Categorical(torch.tensor([1/2, 1/4, 1/4]))
dirichlets = [
    Dirichlet(torch.tensor([10.0, 1.0, 1.0])),
    Dirichlet(torch.tensor([1.0, 10.0, 1.0])),
    Dirichlet(torch.tensor([1.0, 1.0, 10.0]))
]


def sample(batch_size):
    out = torch.empty((batch_size, 3))
    for i in range(batch_size):
        latent = mixture.sample().item()
        out[i] = dirichlets[latent].sample()
    return out


# Train a model for 1000 iterations
batch_size = 36
dmm = DMM(3, 3)
optimizer = torch.optim.SGD(dmm.parameters(), lr=1e-3)

for i in range(1000000):
    optimizer.zero_grad()
    data = sample(batch_size)
    nll, _ = dmm(data)
    nll = nll.mean()
    nll.backward()
    optimizer.step()

    if not i % 1000:
        logger.info('NLL: %0.4f', nll.item())
        logger.info('Mixture weights: %s', F.softmax(dmm.mixture_logits, dim=-1))
        for log_alpha in dmm.log_alphas:
            logger.info('Alpha: %s', log_alpha.exp())