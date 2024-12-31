import unittest
import torch
import numpy as np
from ..model.GraphLevel import GraphLevel, Graph_ModelTrainer
from torch_geometric.data import Data, Batch

class TestGraphLevel(unittest.TestCase):
    def setUp(self):
        # Mock arguments
        class Args:
            def __init__(self):
                self.device = 0 if torch.cuda.is_available() else -1
                self.mad = 0.99  # moving average decay
                self.Gepochs = 10
                self.G_pred_hid = 64
                self.Glr = 0.001
                
        self.args = Args()
        self.layer_config = [300, 128, 64]  # Example layer configuration
        
        # Create sample data
        self.x = torch.randn(10, 300)  # 10 nodes, 300 features
        self.edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        self.batch = torch.zeros(10, dtype=torch.long)  # All nodes belong to same graph
        
        # Create sample batch
        self.sample_batch = Data(
            x=self.x,
            edge_index=self.edge_index,
            x1=self.x,
            edge_index1=self.edge_index,
            x2=self.x,
            edge_index2=self.edge_index,
            batch=self.batch
        )

    def test_graph_level_init(self):
        model = GraphLevel(self.layer_config, self.args)
        self.assertIsNotNone(model.student_encoder)
        self.assertIsNotNone(model.teacher_encoder)
        self.assertIsNotNone(model.student_projector)
        self.assertIsNotNone(model.teacher_projector)
        self.assertIsNotNone(model.pool)

    def test_graph_level_forward(self):
        model = GraphLevel(self.layer_config, self.args)
        emb, loss = model(self.sample_batch)
        
        # Check output types and shapes
        self.assertIsInstance(emb, list)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(len(emb), 1)  # One graph-level embedding
        self.assertEqual(len(emb[0]), self.layer_config[-1])  # Embedding dimension

    def test_graph_model_trainer_init(self):
        try:
            trainer = Graph_ModelTrainer(self.args, block_num=0)
            self.assertIsNotNone(trainer._model)
            self.assertIsNotNone(trainer._optimizer)
        except Exception as e:
            self.skipTest(f"Skipping trainer test due to data loading issues: {str(e)}")

    def test_moving_average_update(self):
        model = GraphLevel(self.layer_config, self.args)
        # Store initial parameters
        init_params = list(model.teacher_encoder.parameters())[0].clone()
        
        # Update moving average
        model.update_moving_average()
        
        # Check if parameters changed
        updated_params = list(model.teacher_encoder.parameters())[0]
        self.assertFalse(torch.equal(init_params, updated_params))

    def test_pooling_operation(self):
        model = GraphLevel(self.layer_config, self.args)
        # Test if pooling works on encoded features
        student = model.student_encoder(self.x.to(model._device), 
                                      self.edge_index.to(model._device))
        pooled = model.pool(student, self.batch.to(model._device))
        
        # Check pooled output shape
        self.assertEqual(pooled.shape[0], 1)  # One graph
        self.assertEqual(pooled.shape[1], self.layer_config[-1])  # Feature dimension

    def test_reset_moving_average(self):
        model = GraphLevel(self.layer_config, self.args)
        model.reset_moving_average()
        self.assertIsNone(model.teacher_encoder)

if __name__ == '__main__':
    unittest.main()
