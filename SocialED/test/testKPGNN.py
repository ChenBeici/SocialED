import pytest
import torch
import numpy as np
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detector.KPGNN import KPGNN, args_define, SocialDataset

tmpdir="../model/model_saved/kpgnn/kpgnn_incremental_test/"

# 模拟数据集
class MockDataset:
    def __init__(self):
        self.features = np.random.rand(100, 300)
        self.labels = np.random.randint(0, 10, 100)
        self.matrix = np.random.rand(100, 100)

# 测试 KPGNN 类的初始化
def test_kpgnn_init():
    args = args_define().args
    dataset = MockDataset()
    kpgnn = KPGNN(args, dataset)
    assert kpgnn.args == args
    assert kpgnn.dataset == dataset
    assert kpgnn.model is None
    assert kpgnn.loss_fn is None
    assert kpgnn.loss_fn_dgi is None
    assert kpgnn.metrics is None
    assert kpgnn.train_indices is None
    assert kpgnn.indices_to_remove is None
    assert kpgnn.embedding_save_path is None
    assert kpgnn.data_split is None

# 测试 KPGNN 类的 preprocess 方法
def test_kpgnn_preprocess(tmpdir):
    args = args_define().args
    args.data_path = str(tmpdir)
    dataset = MockDataset()
    kpgnn = KPGNN(args, dataset)
    kpgnn.preprocess()
    # 这里可以添加更多的断言来检查预处理后的数据

# 测试 KPGNN 类的 fit 方法
def test_kpgnn_fit(tmpdir):
    args = args_define().args
    args.data_path = str(tmpdir)
    dataset = MockDataset()
    kpgnn = KPGNN(args, dataset)
    kpgnn.fit()
    # 这里可以添加更多的断言来检查模型是否正确训练

# 测试 KPGNN 类的 detection 方法
def test_kpgnn_detection(tmpdir):
    args = args_define().args
    args.data_path = str(tmpdir)
    dataset = MockDataset()
    kpgnn = KPGNN(args, dataset)
    kpgnn.fit()
    predictions, ground_truths = kpgnn.detection()
    assert isinstance(predictions, np.ndarray)
    assert isinstance(ground_truths, np.ndarray)
    assert len(predictions) == len(ground_truths)

# 测试 KPGNN 类的 evaluate 方法
def test_kpgnn_evaluate(tmpdir):
    args = args_define().args
    args.data_path = str(tmpdir)
    dataset = MockDataset()
    kpgnn = KPGNN(args, dataset)
    kpgnn.fit()
    predictions, ground_truths = kpgnn.detection()
    ars, ami, nmi = kpgnn.evaluate(predictions, ground_truths)
    assert isinstance(ars, float)
    assert isinstance(ami, float)
    assert isinstance(nmi, float)
    assert 0 <= ars <= 1
    assert 0 <= ami <= 1
    assert 0 <= nmi <= 1

# 测试 SocialDataset 类的初始化
def test_social_dataset_init(tmpdir):
    data_path = str(tmpdir.mkdir("0"))
    np.save(os.path.join(data_path, "features.npy"), np.random.rand(100, 300))
    np.save(os.path.join(data_path, "labels.npy"), np.random.randint(0, 10, 100))
    sparse.save_npz(os.path.join(data_path, "s_bool_A_tid_tid.npz"), sparse.csr_matrix(np.random.rand(100, 100)))
    dataset = SocialDataset(str(tmpdir), 0)
    assert isinstance(dataset.features, np.ndarray)
    assert isinstance(dataset.labels, np.ndarray)
    assert isinstance(dataset.matrix, sparse.csr_matrix)

# 测试 SocialDataset 类的 __len__ 方法
def test_social_dataset_len(tmpdir):
    data_path = str(tmpdir.mkdir("0"))
    np.save(os.path.join(data_path, "features.npy"), np.random.rand(100, 300))
    np.save(os.path.join(data_path, "labels.npy"), np.random.randint(0, 10, 100))
    sparse.save_npz(os.path.join(data_path, "s_bool_A_tid_tid.npz"), sparse.csr_matrix(np.random.rand(100, 100)))
    dataset = SocialDataset(str(tmpdir), 0)
    assert len(dataset) == dataset.features.shape[0]

# 测试 SocialDataset 类的 __getitem__ 方法
def test_social_dataset_getitem(tmpdir):
    data_path = str(tmpdir.mkdir("0"))
    np.save(os.path.join(data_path, "features.npy"), np.random.rand(100, 300))
    np.save(os.path.join(data_path, "labels.npy"), np.random.randint(0, 10, 100))
    sparse.save_npz(os.path.join(data_path, "s_bool_A_tid_tid.npz"), sparse.csr_matrix(np.random.rand(100, 100)))
    dataset = SocialDataset(str(tmpdir), 0)
    feature, label = dataset[0]
    assert isinstance(feature, np.ndarray)
    assert isinstance(label, np.ndarray)

# 测试 SocialDataset 类的 load_adj_matrix 方法
def test_social_dataset_load_adj_matrix(tmpdir):
    data_path = str(tmpdir.mkdir("0"))
    np.save(os.path.join(data_path, "features.npy"), np.random.rand(100, 300))
    np.save(os.path.join(data_path, "labels.npy"), np.random.randint(0, 10, 100))
    sparse.save_npz(os.path.join(data_path, "s_bool_A_tid_tid.npz"), sparse.csr_matrix(np.random.rand(100, 100)))
    dataset = SocialDataset(str(tmpdir), 0)
    assert isinstance(dataset.matrix, sparse.csr_matrix)

# 测试 SocialDataset 类的 remove_obsolete_nodes 方法
def test_social_dataset_remove_obsolete_nodes(tmpdir):
    data_path = str(tmpdir.mkdir("0"))
    np.save(os.path.join(data_path, "features.npy"), np.random.rand(100, 300))
    np.save(os.path.join(data_path, "labels.npy"), np.random.randint(0, 10, 100))
    sparse.save_npz(os.path.join(data_path, "s_bool_A_tid_tid.npz"), sparse.csr_matrix(np.random.rand(100, 100)))
    dataset = SocialDataset(str(tmpdir), 0)
    original_length = len(dataset)
    dataset.remove_obsolete_nodes([0, 1, 2])
    assert len(dataset) == original_length - 3

# 运行测试
if __name__ == "__main__":
    pytest.main()