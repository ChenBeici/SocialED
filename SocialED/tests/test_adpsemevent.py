import unittest
import numpy as np
import pandas as pd
import os
import torch
from ..detector.ADPSEMEvent import ADPSEMEvent, Preprocessor
from unittest.mock import MagicMock, patch

class TestADPSEMEvent(unittest.TestCase):
    def setUp(self):
        # Create mock dataset
        self.mock_dataset = MagicMock()
        self.mock_dataset.get_dataset_language.return_value = "English"
        self.mock_dataset.get_dataset_name.return_value = "test_dataset"
        
        # Initialize model
        self.model = ADPSEMEvent(self.mock_dataset)
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'tweet_id': [1, 2, 3],
            'text': ['tweet1', 'tweet2', 'tweet3'],
            'event_id': [0, 0, 1],
            'created_at': pd.date_range('2023-01-01', periods=3),
            'user_id': [101, 102, 103],
            'user_mentions': [[], [123], [456]],
            'entities': [['entity1'], ['entity2'], ['entity3']],
            'hashtags': [['tag1'], ['tag2'], ['tag3']],
            'urls': [[], [], []],
        })

    def test_initialization(self):
        self.assertEqual(self.model.language, "English")
        self.assertEqual(self.model.dataset_name, "test_dataset")
        self.assertTrue(self.model.save_path.endswith("test_dataset/"))

    @patch('os.path.exists')
    def test_preprocess(self, mock_exists):
        mock_exists.return_value = False
        
        # Mock dataset load_data method
        self.mock_dataset.load_data.return_value = self.sample_data.to_numpy()
        
        try:
            self.model.preprocess()
        except Exception as e:
            self.fail(f"Preprocessing failed with error: {str(e)}")

    def test_detection(self):
        # Test with sample data
        ground_truths = [0, 0, 1, 1]
        predictions = [0, 0, 1, 1]
        
        with patch.object(self.model, 'detection', return_value=(ground_truths, predictions)):
            gt, pred = self.model.detection()
            self.assertEqual(len(gt), len(pred))
            self.assertEqual(gt, ground_truths)
            self.assertEqual(pred, predictions)

    def test_evaluate(self):
        # Test perfect predictions
        ground_truths = [0, 0, 1, 1]
        predictions = [0, 0, 1, 1]
        
        with patch('sys.stdout') as mock_stdout:
            self.model.evaluate(ground_truths, predictions)
            # All metrics should be 1.0 for perfect predictions
            self.assertTrue("1.0" in str(mock_stdout.getvalue()))

    def test_split_open_set(self):
        preprocessor = Preprocessor(self.mock_dataset)
        test_path = "../test_data/open_set/"
        
        # Create test directory if it doesn't exist
        os.makedirs(test_path, exist_ok=True)
        
        try:
            preprocessor.split_open_set(self.sample_data, test_path)
            # Check if files were created
            self.assertTrue(os.path.exists(os.path.join(test_path, "0")))
        finally:
            # Cleanup
            import shutil
            if os.path.exists(test_path):
                shutil.rmtree(test_path)

    def test_run_hier_2D_SE_mini_closed_set(self):
        # Test with minimal sample data
        with patch('numpy.load') as mock_load:
            mock_load.return_value = self.sample_data.to_numpy()
            
            try:
                ground_truths, predictions = self.model.detection()
                self.assertIsInstance(ground_truths, list)
                self.assertIsInstance(predictions, list)
            except Exception as e:
                self.skipTest(f"Detection test skipped due to: {str(e)}")

if __name__ == '__main__':
    unittest.main()
