import unittest
import torch
import torch.nn as nn
from ..utils.losses import (
    common_loss,
    EUC_loss,
    loglikelihood_loss,
    OnlineTripletLoss,
    kl_divergence
)

class TestLosses(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def test_common_loss(self):
        emb1 = torch.randn(10, 5)
        emb2 = torch.randn(10, 5)
        loss = common_loss(emb1, emb2)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # Should be a scalar

    def test_loglikelihood_loss(self):
        y = torch.tensor([[1., 0.], [0., 1.]])
        alpha = torch.tensor([[2., 1.], [1., 2.]])
        loss = loglikelihood_loss(y, alpha, self.device)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape[1], 1)

    def test_kl_divergence(self):
        alpha = torch.tensor([[2., 1.], [1., 2.]])
        num_classes = 2
        kl_div = kl_divergence(alpha, num_classes, self.device)
        self.assertIsInstance(kl_div, torch.Tensor)
        self.assertEqual(kl_div.shape[1], 1)

    def test_EUC_loss(self):
        alpha = torch.tensor([[2., 1.], [1., 2.]]).cuda() if torch.cuda.is_available() else torch.tensor([[2., 1.], [1., 2.]])
        u = torch.tensor([[0.1], [0.2]]).cuda() if torch.cuda.is_available() else torch.tensor([[0.1], [0.2]])
        true_labels = torch.tensor([0, 1]).cuda() if torch.cuda.is_available() else torch.tensor([0, 1])
        loss = EUC_loss(alpha, u, true_labels, epoch_num=1)
        self.assertIsInstance(loss, torch.Tensor)

if __name__ == '__main__':
    unittest.main()
