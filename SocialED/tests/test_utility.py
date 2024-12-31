import unittest
import torch
import numpy as np
from ..utils.utility import (
    tokenize_text,
    validate_device,
    check_parameter,
    preprocess_sentence,
    evaluate_metrics,
    DS_Combin
)

class TestUtility(unittest.TestCase):
    def test_tokenize_text(self):
        text = "This is a test sentence."
        tokens = tokenize_text(text)
        self.assertIsInstance(tokens, list)
        self.assertEqual(len(tokens), 5)
        
        # Test max_length parameter
        long_text = " ".join(["word"] * 1000)
        tokens = tokenize_text(long_text, max_length=100)
        self.assertEqual(len(tokens), 100)

    def test_validate_device(self):
        # Test CPU
        device = validate_device(-1)
        self.assertEqual(device, 'cpu')
        
        # Test invalid GPU ID
        with self.assertRaises(ValueError):
            validate_device(100)

    def test_check_parameter(self):
        # Test valid parameter
        self.assertTrue(check_parameter(5, 0, 10, "test_param"))
        
        # Test invalid parameter
        with self.assertRaises(ValueError):
            check_parameter(-1, 0, 10, "test_param")

    def test_preprocess_sentence(self):
        text = "@user Hello! This is a test :) http://test.com"
        processed = preprocess_sentence(text)
        self.assertNotIn("@user", processed)
        self.assertNotIn("http://", processed)
        self.assertNotIn(":)", processed)

    def test_evaluate_metrics(self):
        labels_true = [0, 0, 1, 1]
        labels_pred = [0, 0, 1, 1]
        nmi, ami, ari = evaluate_metrics(labels_true, labels_pred)
        self.assertEqual(nmi, 1.0)
        self.assertEqual(ami, 1.0)
        self.assertEqual(ari, 1.0)

    def test_DS_Combin(self):
        alpha = [
            torch.tensor([[2., 1.], [1., 2.]]),
            torch.tensor([[1.5, 1.5], [2., 1.]])
        ]
        classes = 2
        alpha_combined, u_combined = DS_Combin(alpha, classes)
        self.assertEqual(alpha_combined.shape, (2, 2))
        self.assertEqual(u_combined.shape, (2, 1))

if __name__ == '__main__':
    unittest.main()
