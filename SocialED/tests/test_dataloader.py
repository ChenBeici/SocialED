import unittest
import os
import shutil
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from ..dataset.dataloader import (
    DatasetLoader,
    Event2012,
    Event2018,
    Event2012_100,
    Event2018_100,
    Mix_Data,
    CrisisLexT6
)

class TestDatasetLoader(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = os.path.join(os.path.dirname(__file__), "test_data")
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create sample data
        self.sample_data = np.array([
            [1, "text1", 0, ["word1"], ["filtered1"], ["entity1"], 101, "2023-01-01", [], ["tag1"], []],
            [2, "text2", 1, ["word2"], ["filtered2"], ["entity2"], 102, "2023-01-02", [], ["tag2"], []]
        ])

    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_dataset_loader_initialization(self):
        """Test DatasetLoader initialization"""
        loader = DatasetLoader(dataset="test_dataset", dir_path=self.test_dir)
        self.assertEqual(loader.dataset, "test_dataset")
        self.assertEqual(loader.dir_path, self.test_dir)
        self.assertEqual(len(loader.required_columns), 11)  # Check number of required columns

    @patch('subprocess.run')
    def test_download_and_cleanup(self, mock_run):
        """Test download_and_cleanup method"""
        loader = DatasetLoader(dataset="test_dataset")
        
        # Mock successful download
        mock_run.return_value = MagicMock(returncode=0)
        
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True
            with patch('shutil.copy2') as mock_copy:
                success = loader.download_and_cleanup(
                    "mock_url",
                    "test_dataset",
                    self.test_dir
                )
                self.assertTrue(success)

    def test_event2012_initialization(self):
        """Test Event2012 dataset initialization"""
        dataset = Event2012(dir_path=self.test_dir)
        self.assertEqual(dataset.dataset, "Event2012")

    def test_event2018_initialization(self):
        """Test Event2018 dataset initialization"""
        dataset = Event2018(dir_path=self.test_dir)
        self.assertEqual(dataset.dataset, "Event2018")

    @patch('numpy.load')
    def test_load_data(self, mock_load):
        """Test data loading functionality"""
        mock_load.return_value = self.sample_data
        
        # Test with Event2012
        dataset = Event2012(dir_path=self.test_dir)
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True
            df = dataset.load_data()
            self.assertIsInstance(df, pd.DataFrame)
            self.assertEqual(len(df), 2)
            self.assertEqual(len(df.columns), 11)

    def test_event2012_100_initialization(self):
        """Test Event2012_100 dataset initialization"""
        dataset = Event2012_100(dir_path=self.test_dir)
        self.assertEqual(dataset.dataset, "Event2012_100")

    def test_event2018_100_initialization(self):
        """Test Event2018_100 dataset initialization"""
        dataset = Event2018_100(dir_path=self.test_dir)
        self.assertEqual(dataset.dataset, "Event2018_100")

    def test_mix_data_initialization(self):
        """Test Mix_Data dataset initialization"""
        dataset = Mix_Data(dir_path=self.test_dir)
        self.assertEqual(dataset.dataset, "Mix_Data")

    def test_crisis_lext6_initialization(self):
        """Test CrisisLexT6 dataset initialization"""
        dataset = CrisisLexT6(dir_path=self.test_dir)
        self.assertEqual(dataset.dataset, "CrisisLexT6")

    @patch('numpy.load')
    def test_data_loading_error_handling(self, mock_load):
        """Test error handling during data loading"""
        mock_load.side_effect = FileNotFoundError
        
        dataset = Event2012(dir_path=self.test_dir)
        with self.assertRaises(FileNotFoundError):
            dataset.load_data()

    def test_required_columns_consistency(self):
        """Test that all dataset classes use the same required columns"""
        base_loader = DatasetLoader()
        event2012 = Event2012()
        event2018 = Event2018()
        event2012_100 = Event2012_100()
        event2018_100 = Event2018_100()
        mix_data = Mix_Data()
        crisis_lext6 = CrisisLexT6()
        
        self.assertEqual(base_loader.required_columns, event2012.required_columns)
        self.assertEqual(base_loader.required_columns, event2018.required_columns)
        self.assertEqual(base_loader.required_columns, event2012_100.required_columns)
        self.assertEqual(base_loader.required_columns, event2018_100.required_columns)
        self.assertEqual(base_loader.required_columns, mix_data.required_columns)
        self.assertEqual(base_loader.required_columns, crisis_lext6.required_columns)

    @patch('os.makedirs')
    def test_directory_creation(self, mock_makedirs):
        """Test directory creation functionality"""
        DatasetLoader(dataset="test_dataset", dir_path=self.test_dir)
        mock_makedirs.assert_called()

    @patch('numpy.load')
    def test_data_format(self, mock_load):
        """Test data format validation"""
        # Test with invalid data format
        mock_load.return_value = np.array([[1, 2, 3]])  # Invalid data format
        
        dataset = Event2012(dir_path=self.test_dir)
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True
            with self.assertRaises(Exception):
                df = dataset.load_data()

if __name__ == '__main__':
    unittest.main()
