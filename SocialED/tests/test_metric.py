import unittest
import numpy as np
from ..metrics.metric import (
    eval_nmi,
    eval_ami,
    eval_ari,
    eval_f1,
    eval_acc
)

class TestMetrics(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method"""
        # Create sample ground truth and prediction arrays
        self.ground_truths = np.array([0, 0, 1, 1, 2, 2, 2])
        self.predictions = np.array([0, 0, 1, 1, 1, 2, 2])
        
        # Perfect predictions for testing upper bounds
        self.perfect_predictions = np.array([0, 0, 1, 1, 2, 2, 2])
        
        # Completely wrong predictions for testing lower bounds
        self.wrong_predictions = np.array([2, 2, 0, 0, 1, 1, 1])

    def test_eval_nmi(self):
        """Test Normalized Mutual Information score"""
        # Test with sample predictions
        nmi = eval_nmi(self.ground_truths, self.predictions)
        self.assertTrue(0 <= nmi <= 1)
        
        # Test perfect predictions
        perfect_nmi = eval_nmi(self.ground_truths, self.perfect_predictions)
        self.assertAlmostEqual(perfect_nmi, 1.0)
        
        # Test symmetry
        nmi_1 = eval_nmi(self.ground_truths, self.predictions)
        nmi_2 = eval_nmi(self.predictions, self.ground_truths)
        self.assertAlmostEqual(nmi_1, nmi_2)

    def test_eval_ami(self):
        """Test Adjusted Mutual Information score"""
        # Test with sample predictions
        ami = eval_ami(self.ground_truths, self.predictions)
        self.assertTrue(-1 <= ami <= 1)
        
        # Test perfect predictions
        perfect_ami = eval_ami(self.ground_truths, self.perfect_predictions)
        self.assertAlmostEqual(perfect_ami, 1.0)
        
        # Test symmetry
        ami_1 = eval_ami(self.ground_truths, self.predictions)
        ami_2 = eval_ami(self.predictions, self.ground_truths)
        self.assertAlmostEqual(ami_1, ami_2)

    def test_eval_ari(self):
        """Test Adjusted Rand Index score"""
        # Test with sample predictions
        ari = eval_ari(self.ground_truths, self.predictions)
        self.assertTrue(-1 <= ari <= 1)
        
        # Test perfect predictions
        perfect_ari = eval_ari(self.ground_truths, self.perfect_predictions)
        self.assertAlmostEqual(perfect_ari, 1.0)
        
        # Test symmetry
        ari_1 = eval_ari(self.ground_truths, self.predictions)
        ari_2 = eval_ari(self.predictions, self.ground_truths)
        self.assertAlmostEqual(ari_1, ari_2)

    def test_eval_f1(self):
        """Test F1 score"""
        # Test with sample predictions
        f1 = eval_f1(self.ground_truths, self.predictions)
        self.assertTrue(0 <= f1 <= 1)
        
        # Test perfect predictions
        perfect_f1 = eval_f1(self.ground_truths, self.perfect_predictions)
        self.assertAlmostEqual(perfect_f1, 1.0)
        
        # Test completely wrong predictions
        wrong_f1 = eval_f1(self.ground_truths, self.wrong_predictions)
        self.assertLess(wrong_f1, perfect_f1)

    def test_eval_acc(self):
        """Test Accuracy score"""
        # Test with sample predictions
        acc = eval_acc(self.ground_truths, self.predictions)
        self.assertTrue(0 <= acc <= 1)
        
        # Test perfect predictions
        perfect_acc = eval_acc(self.ground_truths, self.perfect_predictions)
        self.assertAlmostEqual(perfect_acc, 1.0)
        
        # Test completely wrong predictions
        wrong_acc = eval_acc(self.ground_truths, self.wrong_predictions)
        self.assertLess(wrong_acc, perfect_acc)

    def test_input_validation(self):
        """Test input validation and error handling"""
        # Test with arrays of different lengths
        with self.assertRaises(ValueError):
            eval_nmi(self.ground_truths, self.predictions[:-1])
        
        # Test with empty arrays
        with self.assertRaises(ValueError):
            eval_ami([], [])
        
        # Test with non-numeric data
        with self.assertRaises(ValueError):
            eval_ari(['a', 'b'], ['c', 'd'])

    def test_edge_cases(self):
        """Test edge cases"""
        # Single class
        single_class = np.zeros(5)
        score = eval_nmi(single_class, single_class)
        self.assertTrue(0 <= score <= 1)
        
        # Binary classification
        binary_gt = np.array([0, 0, 1, 1])
        binary_pred = np.array([0, 0, 1, 1])
        score = eval_f1(binary_gt, binary_pred)
        self.assertAlmostEqual(score, 1.0)
        
        # All wrong predictions
        all_wrong = np.ones(len(self.ground_truths))
        score = eval_acc(self.ground_truths, all_wrong)
        self.assertLess(score, 0.5)


if __name__ == '__main__':
    unittest.main()
