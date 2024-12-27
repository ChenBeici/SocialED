import unittest
import numpy as np
import torch
import os
import pandas as pd
import networkx as nx
from ..utils.utility import (
    construct_graph_from_df,
    tokenize_text,
    load_data,
    logger,
    pprint,
    validate_device,
    check_parameter
)

class TestGraphConstruction(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method"""
        # Create sample DataFrame
        self.df = pd.DataFrame({
            'tweet_id': [1, 2],
            'user_mentions': [['user1', 'user2'], ['user3']],
            'user_id': ['main_user1', 'main_user2'],
            'entities': [['entity1'], ['entity2', 'entity3']],
            'sampled_words': [['word1', 'word2'], ['word3']]
        })

    def test_construct_graph_from_df(self):
        """Test graph construction from DataFrame"""
        G = construct_graph_from_df(self.df)
        
        # Check if graph is created
        self.assertIsInstance(G, nx.Graph)
        
        # Check if nodes are created correctly
        self.assertTrue('t_1' in G.nodes())
        self.assertTrue('u_main_user1' in G.nodes())
        self.assertTrue('w_word1' in G.nodes())
        
        # Check node attributes
        self.assertTrue(G.nodes['t_1']['tweet_id'])
        self.assertTrue(G.nodes['u_main_user1']['user_id'])
        
        # Check edges
        self.assertTrue(G.has_edge('t_1', 'u_main_user1'))


class TestTextProcessing(unittest.TestCase):
    def test_tokenize_text(self):
        """Test text tokenization"""
        # Test normal case
        text = "Hello World! This is a test."
        tokens = tokenize_text(text)
        self.assertEqual(tokens, ['hello', 'world!', 'this', 'is', 'a', 'test.'])
        
        # Test max length
        long_text = " ".join(["word"] * 1000)
        tokens = tokenize_text(long_text, max_length=5)
        self.assertEqual(len(tokens), 5)
        
        # Test whitespace handling
        text_with_spaces = "  Multiple    Spaces   "
        tokens = tokenize_text(text_with_spaces)
        self.assertEqual(tokens, ['multiple', 'spaces'])


class TestDataLoading(unittest.TestCase):
    def test_load_data(self):
        """Test data loading functionality"""
        # Test with invalid dataset name
        with self.assertRaises(RuntimeError):
            load_data("nonexistent_dataset")
        
        # Test cache directory creation
        test_cache_dir = "test_cache"
        if os.path.exists(test_cache_dir):
            os.rmdir(test_cache_dir)
            
        try:
            load_data("maven", cache_dir=test_cache_dir)
            self.assertTrue(os.path.exists(test_cache_dir))
        finally:
            if os.path.exists(test_cache_dir):
                os.rmdir(test_cache_dir)


class TestLogging(unittest.TestCase):
    def test_logger(self):
        """Test logger functionality"""
        # Test basic logging
        logger(epoch=1, loss=0.5, verbose=1)
        
        # Test with all parameters
        logger(
            epoch=1,
            loss=(0.3, 0.2),
            score=torch.tensor([1, 0, 1]),
            target=torch.tensor([1, 0, 1]),
            time=1.5,
            verbose=3,
            train=True,
            deep=True
        )
        
        # Test non-training mode
        logger(epoch=1, loss=0.5, verbose=1, train=False)


class TestParameterValidation(unittest.TestCase):
    def test_validate_device(self):
        """Test device validation"""
        # Test CPU
        self.assertEqual(validate_device(-1), 'cpu')
        
        # Test invalid GPU ID
        if torch.cuda.is_available():
            with self.assertRaises(ValueError):
                validate_device(torch.cuda.device_count() + 1)

    def test_check_parameter(self):
        """Test parameter checking"""
        # Test normal cases
        self.assertTrue(check_parameter(5, 0, 10))
        
        # Test boundary cases
        with self.assertRaises(ValueError):
            check_parameter(5, 10, 0)  # invalid range
            
        with self.assertRaises(ValueError):
            check_parameter(5, 0, 5, include_right=False)
            
        # Test type checking
        with self.assertRaises(TypeError):
            check_parameter("5", 0, 10)
            
        # Test parameter name
        with self.assertRaises(ValueError):
            check_parameter(-1, 0, 10, param_name="test_param")


class TestPrinting(unittest.TestCase):
    def test_pprint(self):
        """Test pretty printing"""
        # Test with different types of values
        params = {
            'int_param': 42,
            'float_param': 3.14,
            'str_param': 'test',
            'bool_param': True
        }
        
        result = pprint(params)
        
        # Check if all parameters are in the output
        self.assertIn('int_param=42', result)
        self.assertIn('float_param=3.14', result)
        self.assertIn('str_param=', result)
        self.assertIn('bool_param=True', result)
        
        # Test with long string
        long_params = {
            'long_string': 'x' * 1000
        }
        result = pprint(long_params)
        self.assertLess(len(result), 1000)  # Should be truncated


if __name__ == '__main__':
    unittest.main()
