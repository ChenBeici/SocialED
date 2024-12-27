import unittest
import torch
import numpy as np
from datetime import datetime
from ..detector.hcrc import (
    HCRC,
    args_define,
    SinglePass,
    extract_time_feature,
    df_to_t_features,
    evaluate_fun,
    random_cluster
)
from ..dataset.dataloader import DatasetLoader


class TestHCRC(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.args = args_define()
        self.dataset = DatasetLoader("maven").load_data()
        self.hcrc = HCRC(self.args, self.dataset)

    def test_init(self):
        """Test initialization of HCRC"""
        self.assertIsInstance(self.hcrc, HCRC)
        self.assertEqual(self.hcrc.dataset, self.dataset)
        self.assertEqual(self.hcrc.args, self.args)

    def test_evaluate(self):
        """Test evaluate method"""
        # Create sample predictions and ground truths
        predictions = np.array([0, 1, 0, 1, 2])
        ground_truths = np.array([0, 1, 0, 1, 1])
        
        # Test evaluate method
        ars, ami, nmi = self.hcrc.evaluate(predictions, ground_truths)
        
        # Check if metrics are between 0 and 1
        self.assertTrue(0 <= ars <= 1)
        self.assertTrue(0 <= ami <= 1)
        self.assertTrue(0 <= nmi <= 1)


class TestSinglePass(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.threshold = 0.7
        self.embeddings = np.random.rand(10, 64)  # 10 samples, 64 dimensions
        self.flag = 0
        self.pred_label = None
        self.size = 5
        self.para = 0.5
        self.sim_init = 0.8
        self.sim = True
        
        self.single_pass = SinglePass(
            self.threshold,
            self.embeddings,
            self.flag,
            self.pred_label,
            self.size,
            self.para,
            self.sim_init,
            self.sim
        )

    def test_get_center(self):
        """Test get_center method"""
        # Create sample data
        data = np.random.rand(10, 5)  # 10 samples, 5 features
        labels = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 2])  # 3 clusters
        
        # Get centers and indices
        centers, indices_per_cluster = self.single_pass.get_center(labels, data)
        
        # Check centers shape
        self.assertEqual(len(centers), len(np.unique(labels)))
        self.assertEqual(len(centers[0]), data.shape[1])
        
        # Check indices
        self.assertTrue(all(isinstance(idx_list, list) for idx_list in indices_per_cluster))


class TestTimeFeatures(unittest.TestCase):
    def test_extract_time_feature(self):
        """Test time feature extraction"""
        # Test with a sample datetime
        time_str = "2023-01-01T12:00:00"
        features = extract_time_feature(time_str)
        
        # Check output
        self.assertEqual(len(features), 2)
        self.assertIsInstance(features[0], float)
        self.assertIsInstance(features[1], float)
        self.assertTrue(0 <= features[1] <= 1)  # seconds should be normalized

    def test_df_to_t_features(self):
        """Test dataframe to time features conversion"""
        # Create sample dataframe
        dates = ["2023-01-01T12:00:00", "2023-01-02T13:30:00"]
        df = {'created_at': dates}
        df = pd.DataFrame(df)
        
        # Convert to features
        features = df_to_t_features(df)
        
        # Check output
        self.assertEqual(features.shape, (2, 2))
        self.assertTrue(np.all(features >= 0))


class TestClustering(unittest.TestCase):
    def test_random_cluster(self):
        """Test random clustering"""
        # Create sample embeddings
        embeddings = np.random.rand(20, 64)
        block_num = 0
        pred_label = None
        
        # Perform clustering
        cluster_result, threshold = random_cluster(embeddings, block_num, pred_label)
        
        # Check outputs
        self.assertTrue(0.6 <= threshold <= 0.8)
        self.assertEqual(len(cluster_result), len(embeddings))
        self.assertTrue(all(isinstance(x, (int, np.integer)) for x in cluster_result))

    def test_evaluate_fun(self):
        """Test evaluate function"""
        # Create sample data
        embeddings = np.random.rand(20, 64)
        labels = np.array([0, 0, 1, 1, 1] * 4)
        block_num = 0
        pred_label = None
        result_path = "test_results.txt"
        task = "random"
        
        # Run evaluation
        y_pred = evaluate_fun(embeddings, labels, block_num, pred_label, result_path, task)
        
        # Check output
        self.assertEqual(len(y_pred), len(labels))
        self.assertTrue(all(isinstance(x, (int, np.integer)) for x in y_pred))


if __name__ == '__main__':
    unittest.main()
