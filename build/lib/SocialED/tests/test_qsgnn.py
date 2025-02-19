import unittest
import torch
import numpy as np
from ..detector.qsgnn import (
    QSGNN,
    args_define,
    Preprocessor,
    GAT,
    OnlineTripletLoss,
    RandomNegativeTripletSelector,
    Arabic_preprocessor
)
from ..dataset.dataloader import DatasetLoader

class TestQSGNN(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.args = args_define()
        self.dataset = DatasetLoader("maven").load_data()
        self.qsgnn = QSGNN(self.args, self.dataset)

    def test_init(self):
        """Test initialization of QSGNN"""
        self.assertIsInstance(self.qsgnn, QSGNN)
        self.assertEqual(self.qsgnn.dataset, self.dataset)
        self.assertEqual(self.qsgnn.use_cuda, self.args.use_cuda)

    def test_evaluate(self):
        """Test evaluate method"""
        # Create sample predictions and ground truths
        predictions = np.array([0, 1, 0, 1, 2])
        ground_truths = np.array([0, 1, 0, 1, 1])
        
        # Test evaluate method
        ars, ami, nmi = self.qsgnn.evaluate(predictions, ground_truths)
        
        # Check if metrics are between 0 and 1
        self.assertTrue(0 <= ars <= 1)
        self.assertTrue(0 <= ami <= 1)
        self.assertTrue(0 <= nmi <= 1)


class TestPreprocessor(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.args = args_define()
        self.preprocessor = Preprocessor(self.args)

    def test_extract_time_feature(self):
        """Test time feature extraction"""
        # Test with a sample datetime
        time_str = "2023-01-01T12:00:00"
        features = self.preprocessor.extract_time_feature(time_str)
        
        self.assertEqual(len(features), 2)
        self.assertIsInstance(features[0], float)
        self.assertIsInstance(features[1], float)


class TestArabicPreprocessor(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.arabic_preprocessor = Arabic_preprocessor(tokenizer=None)

    def test_clean_text(self):
        """Test Arabic text cleaning"""
        # Test with sample Arabic text containing common patterns
        test_text = "أهلاً وسهلاً..."
        cleaned_text = self.arabic_preprocessor.clean_text(test_text)
        
        # Check if text is cleaned properly
        self.assertIsInstance(cleaned_text, str)
        self.assertNotEqual(cleaned_text, test_text)
        self.assertNotIn('...', cleaned_text)


class TestGAT(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.in_dim = 300
        self.hidden_dim = 16
        self.out_dim = 64
        self.num_heads = 4
        self.use_residual = True
        
        self.model = GAT(
            self.in_dim,
            self.hidden_dim,
            self.out_dim,
            self.num_heads,
            self.use_residual
        )

    def test_init(self):
        """Test initialization of GAT"""
        self.assertIsInstance(self.model.layer1, torch.nn.Module)
        self.assertIsInstance(self.model.layer2, torch.nn.Module)

    def test_forward_shape(self):
        """Test forward pass output shape"""
        batch_size = 32
        
        # Create dummy blocks
        blocks = [
            {
                'srcdata': {'features': torch.randn(batch_size, self.in_dim)},
                'number_of_dst_nodes': lambda: batch_size
            },
            {
                'srcdata': {'features': torch.randn(batch_size, self.hidden_dim * self.num_heads)},
                'number_of_dst_nodes': lambda: batch_size
            }
        ]
        
        # Forward pass
        output = self.model(blocks)
        
        # Check output shape
        expected_shape = (batch_size, self.out_dim)
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
        self.assertEqual(args.finetune_epochs, 1)
        self.assertEqual(args.n_epochs, 5)
        self.assertEqual(args.window_size, 3)
        self.assertEqual(args.patience, 5)
        self.assertEqual(args.margin, 3.0)
        self.assertEqual(args.lr, 1e-3)
        self.assertEqual(args.batch_size, 1000)
        self.assertEqual(args.hidden_dim, 16)
        self.assertEqual(args.out_dim, 64)
        self.assertEqual(args.num_heads, 4)

    def test_custom_args(self):
        """Test custom arguments"""
        custom_args = {
            'finetune_epochs': 3,
            'n_epochs': 10,
            'batch_size': 500,
            'hidden_dim': 32
        }
        
        args = args_define(**custom_args)
        
        # Test custom values
        self.assertEqual(args.finetune_epochs, 3)
        self.assertEqual(args.n_epochs, 10)
        self.assertEqual(args.batch_size, 500)
        self.assertEqual(args.hidden_dim, 32)
        
        # Test that other defaults remain unchanged
        self.assertEqual(args.margin, 3.0)
        self.assertEqual(args.lr, 1e-3)


if __name__ == '__main__':
    unittest.main()
