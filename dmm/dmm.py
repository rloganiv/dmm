"""Dirichlet Mixture Model implementation"""
from typing import List

from overrides import overrides
import torch
from torch.nn import Module, Parameter, ParameterList
import torch.nn.functional as F


def log_p(observed_data: torch.FloatTensor,
          log_alpha: torch.FloatTensor) -> torch.FloatTensor:
    """Computes the log probability of observed data under a Dirichlet distribution with
    parameters alpha."""
    alpha = log_alpha.exp()
    return ((torch.log(observed_data) * (alpha - 1.0)).sum(-1) +
            torch.lgamma(alpha.sum(-1)) -
            torch.lgamma(alpha).sum(-1))


class DMM(Module):
    """Dirichlet Mixture Model

    Parameters
    ==========
    dim : ``int``
        Dimension of the observed data.
    n_components : ``int``
        Number of mixture components.
    """
    def __init__(self,
                 dim: int,
                 n_components: int) -> None:
        super(DMM, self).__init__()
        self._dim = dim
        self._n_components = n_components

        mixture_logits = torch.zeros((n_components,), dtype=torch.float)
        self.mixture_logits = Parameter(mixture_logits)

        self.log_alphas = ParameterList()
        for _ in range(n_components):
            log_alpha = Parameter(torch.randn(dim, dtype=torch.float)/3)
            self.log_alphas.append(log_alpha)

    @overrides
    def forward(self, # pylint: disable=arguments-differ
                observed_data: torch.FloatTensor):
        """Computes the expected value of the log-likelihood function (e.g. the E-step)

        Parameters
        ==========
        observed_data : ``torch.FloatTensor(size=(batch_size, dim))``
            The observed data.

        Returns
        =======
        nll : ``torch.FloatTensor(size=(batch_size,))``
            The negative likelihood of the observed data.
        membership_probs : ``torch.FloatTensor(size=(batch_size, n_components))``
            The membership probabilities.
        """
        batch_size = observed_data.size()[0]

        # Convert mixture logits to log probabilities
        prior_log_probs = F.log_softmax(self.mixture_logits, dim=-1)

        # Compute membership probabilities.
        # NOTE: Need to use no_grad() here to prevent torch from trying to differentiate through
        # this step.
        membership_log_probs = torch.empty(size=(batch_size, self._n_components), requires_grad=False)
        for i in range(self._n_components):
            membership_log_probs[:, i] = prior_log_probs[i] + log_p(observed_data, self.log_alphas[i])
        denom = torch.logsumexp(membership_log_probs, dim=1)
        denom = denom.unsqueeze(1)
        membership_log_probs -= denom
        # Need to detach since gradient does not propagate through membership probabilities in EM.
        membership_probs = membership_log_probs.exp().detach()

        # Compute expected negative log-likelihood w.r.t membership probabilities
        nll = torch.empty(size=(batch_size,), requires_grad=False)
        for i in range(self._n_components):
            log_likelihood = log_p(observed_data, self.log_alphas[i]) + prior_log_probs[i]
            nll[:,] -= membership_probs[:, i] * log_likelihood

        return nll, membership_probs
