from unittest import TestCase

import torch
from torch.distributions import Dirichlet

from dmm.dmm import DMM


class DmmTest(TestCase):
    def setUp(self):
        self.dmm = DMM(dim=10, n_components=4)

    def test_init(self):
        parameter_list = list(self.dmm.parameters())
        # Validate model parameters. There should be one parameter for the mixture weights and four
        # parameters for tracking each Dirichlet's parameters.
        self.assertEqual(len(parameter_list), 5)
        # Check that mixture weights sum to one.
        self.assertAlmostEqual(self.dmm.log_mixture_weights.exp().sum().item(), 1)

    def test_forward(self):
        source_dirichlet = Dirichlet(torch.ones(10))
        batch_size = 12
        observed_data = source_dirichlet.sample((batch_size,))
        # Check that forward function returns
        _, membership_probs = self.dmm(observed_data)
        # Ensure membership probabilities sum to one
        for prob in membership_probs.sum(dim=1):
            self.assertAlmostEqual(prob.item(), 1, places=5)

    def test_backward(self):
        source_dirichlet = Dirichlet(torch.ones(10))
        batch_size = 12
        observed_data = source_dirichlet.sample((batch_size,))
        # Obtain loss
        nll, _ = self.dmm(observed_data)
        loss = nll.sum()
        # Check that gradient is non-zero for params
        loss.backward()
        for param in self.dmm.parameters():
            self.assertIsNotNone(param.grad)
