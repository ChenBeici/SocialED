import pytest
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_mutual_info_score, adjusted_rand_score
from sentence_transformers import SentenceTransformer
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.dataloader import DatasetLoader
from detector.hisevent import *
# Mock data for testing
@pytest.fixture
def mock_dataset():
    data = {
        'filtered_words': [['word1', 'word2'], ['word3', 'word4'], ['word5', 'word6']],
        'event_id': [1, 2, 1]
    }
    return pd.DataFrame(data)

@pytest.fixture
def hisevent_instance():
    return HISEvent()

@pytest.fixture
def preprocessor_instance():
    return Preprocessor()

def test_preprocess(hisevent_instance):
    hisevent_instance.preprocess()
    # Add assertions to check if preprocessing was successful
    assert True

def test_detection(hisevent_instance):
    hisevent_instance.detection()
    # Add assertions to check if detection was successful
    assert True
'''
def test_get_stable_point(tmpdir):
    path = tmpdir.mkdir("test_dir")
    embeddings = np.random.rand(10, 100)
    embeddings_path = os.path.join(path, 'SBERT_embeddings.pkl')
    with open(embeddings_path, 'wb') as f:
        pickle.dump(embeddings, f)
    
    # 确保文件存在
    assert os.path.exists(embeddings_path)
    
    stable_points = get_stable_point(str(path))
    assert isinstance(stable_points, dict)
    assert 'first' in stable_points
    assert 'global' in stable_points
'''
def test_run_hier_2D_SE_mini_Event2012_open_set(tmpdir):
    save_path = str(tmpdir.mkdir("test_dir"))
    embeddings = np.random.rand(10, 100)
    df_np = np.array([[i, i % 3, f'tweet_{i}', f'user_{i}', '2023-01-01'] for i in range(10)])
    df = pd.DataFrame(data=df_np, columns=["original_index", "event_id", "text", "user_id", "created_at"])
    block_path = os.path.join(save_path, '1')
    os.makedirs(block_path, exist_ok=True)
    np.save(os.path.join(block_path, '1.npy'), df_np)
    embeddings_path = os.path.join(block_path, 'SBERT_embeddings.pkl')
    with open(embeddings_path, 'wb') as f:
        pickle.dump(embeddings, f)
    
    # 确保文件存在
    assert os.path.exists(embeddings_path)
    
    run_hier_2D_SE_mini_Event2012_open_set(n=10, e_a=True, e_s=True, test_with_one_block=True)
    # Add assertions to check if the function ran successfully
    assert True

def test_evaluate():
    labels_true = [1, 2, 1]
    labels_pred = [0, 1, 0]
    nmi, ami, ari = evaluate(labels_true, labels_pred)
    assert isinstance(nmi, float)
    assert isinstance(ami, float)
    assert isinstance(ari, float)

def test_decode():
    division = [[1, 2], [3, 4]]
    prediction = decode(division)
    assert isinstance(prediction, list)
    assert len(prediction) == 4

# Additional tests can be added as needed