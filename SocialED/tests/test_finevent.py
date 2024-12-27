import unittest
import torch
import numpy as np
from ..detector.finevent import (
    FinEvent, 
    args_define,
    Preprocessor,
    MarGNN,
    OnlineTripletLoss,
    RandomNegativeTripletSelector
)
from ..dataset.dataloader import DatasetLoader

class TestFinEvent(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.args = args_define()
        self.dataset = DatasetLoader("maven").load_data()
        self.finevent = FinEvent(self.args, self.dataset)

    def test_init(self):
        """Test initialization of FinEvent"""
        self.assertIsInstance(self.finevent, FinEvent)
        self.assertEqual(self.finevent.dataset, self.dataset)

    def test_evaluate(self):
        """Test evaluate method"""
        # Create sample predictions and ground truths
        predictions = np.array([0, 1, 0, 1, 2])
        ground_truths = np.array([0, 1, 0, 1, 1])
        
        # Test evaluate method
        ars, ami, nmi = self.finevent.evaluate(predictions, ground_truths)
        
        # Check if metrics are between 0 and 1
        self.assertTrue(0 <= ars <= 1)
        self.assertTrue(0 <= ami <= 1)
        self.assertTrue(0 <= nmi <= 1)


class TestPreprocessor(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.preprocessor = Preprocessor()

    def test_extract_time_feature(self):
        """Test time feature extraction"""
        # Test with a sample datetime
        time_str = "2023-01-01T12:00:00"
        features = self.preprocessor.extract_time_feature(time_str)
        
        self.assertEqual(len(features), 2)
        self.assertIsInstance(features[0], float)
        self.assertIsInstance(features[1], float)


class TestMarGNN(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.in_dim = 300
        self.hidden_dim = 128
        self.out_dim = 64
        self.heads = 4
        self.num_relations = 3
        self.gnn_args = (self.in_dim, self.hidden_dim, self.out_dim, self.heads)
        
        self.model = MarGNN(
            self.gnn_args,
            num_relations=self.num_relations,
            inter_opt='cat_w_avg',
            is_shared=False
        )

    def test_init(self):
        """Test initialization of MarGNN"""
        self.assertEqual(len(self.model.intra_aggs), self.num_relations)
        self.assertEqual(self.model.inter_opt, 'cat_w_avg')
        self.assertFalse(self.model.is_shared)

    def test_forward_shape(self):
        """Test forward pass output shape"""
        batch_size = 32
        x = torch.randn(batch_size, self.in_dim)
        
        # Create dummy adjacency matrices
        adjs = [(torch.tensor([[0, 1], [1, 0]]), None, (batch_size, batch_size)) for _ in range(self.num_relations)]
        
        # Create dummy node IDs
        n_ids = [torch.arange(batch_size) for _ in range(self.num_relations)]
        
        # Create dummy thresholds
        RL_thresholds = torch.ones(self.num_relations, 1)
        
        # Move tensors to CPU for testing
        device = torch.device('cpu')
        
        # Forward pass
        output = self.model(x, adjs, n_ids, device, RL_thresholds)
        
        # Check output shape
        expected_shape = (batch_size, self.out_dim * self.num_relations)
        self.assertEqual(output.shape, expected_shape)


class TestTripletLoss(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.margin = 1.0
        self.triplet_selector = RandomNegativeTripletSelector(margin=self.margin)
        self.loss_fn = OnlineTripletLoss(margin=self.margin, triplet_selector=self.triplet_selector)

    def test_triplet_loss(self):
        """Test triplet loss computation"""
        # Create sample embeddings and labels
        embeddings = torch.randn(10, 64)  # 10 samples, 64 dimensions
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])  # 5 classes, 2 samples each
        
        # Compute loss
        loss, num_triplets = self.loss_fn(embeddings, labels)
        
        # Check if loss is a scalar
        self.assertEqual(loss.dim(), 0)
        # Check if loss is non-negative
        self.assertGreaterEqual(loss.item(), 0)
        # Check if number of triplets is non-negative
        self.assertGreaterEqual(num_triplets, 0)


class TestArgsDefine(unittest.TestCase):
    def test_default_args(self):
        """Test default arguments"""
        args = args_define()
        
        # Test some default values
        self.assertEqual(args.n_epochs, 1)
        self.assertEqual(args.window_size, 3)
        self.assertEqual(args.patience, 5)
        self.assertEqual(args.margin, 3.0)
        self.assertEqual(args.lr, 1e-3)
        self.assertEqual(args.batch_size, 50)
        self.assertEqual(args.hidden_dim, 128)
        self.assertEqual(args.out_dim, 64)
        self.assertEqual(args.heads, 4)

    def test_custom_args(self):
        """Test custom arguments"""
        custom_args = {
            'n_epochs': 5,
            'window_size': 5,
            'batch_size': 32,
            'hidden_dim': 256
        }
        
        args = args_define(**custom_args)
        
        # Test custom values
        self.assertEqual(args.n_epochs, 5)
        self.assertEqual(args.window_size, 5)
        self.assertEqual(args.batch_size, 32)
        self.assertEqual(args.hidden_dim, 256)
        
        # Test that other defaults remain unchanged
        self.assertEqual(args.margin, 3.0)
        self.assertEqual(args.lr, 1e-3)


if __name__ == '__main__':
    unittest.main()
