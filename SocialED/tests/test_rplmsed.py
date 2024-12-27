import pytest
import torch
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detector.rplmsed import *
# 创建一个示例数据集
@pytest.fixture
def sample_dataset():
    data = [
        DataItem(event_id=1, tweet_id=1, text="Hello world", user_id=1, created_at="2023-10-01", user_loc="USA", place_type="", place_full_name="", place_country_code="", hashtags=[], user_mentions=[], urls=[], entities=[], words=["Hello", "world"], filtered_words=[], sampled_words=[]),
        DataItem(event_id=1, tweet_id=2, text="Goodbye world", user_id=2, created_at="2023-10-02", user_loc="USA", place_type="", place_full_name="", place_country_code="", hashtags=[], user_mentions=[], urls=[], entities=[], words=["Goodbye", "world"], filtered_words=[], sampled_words=[]),
        DataItem(event_id=2, tweet_id=3, text="Hello universe", user_id=3, created_at="2023-10-03", user_loc="USA", place_type="", place_full_name="", place_country_code="", hashtags=[], user_mentions=[], urls=[], entities=[], words=["Hello", "universe"], filtered_words=[], sampled_words=[])
    ]
    return data

# 测试 Preprocessor 类的功能
def test_preprocessor_split_into_blocks(sample_dataset):
    preprocessor = Preprocessor()
    blocks = preprocessor.split_into_blocks(sample_dataset)
    assert len(blocks) > 0
    assert isinstance(blocks[0], list)

def test_preprocessor_process_block(sample_dataset):
    preprocessor = Preprocessor()
    block = {"train": sample_dataset, "test": sample_dataset, "valid": sample_dataset}
    processed_block = preprocessor.process_block(block)
    assert "train" in processed_block
    assert "test" in processed_block
    assert "valid" in processed_block

# 测试 RPLM_SED 类的功能
def test_rplmsed_preprocess(sample_dataset):
    args = args_define().args
    rplmsed = RPLM_SED(args, sample_dataset)
    rplmsed.preprocess()
    # 这里可以添加更多的断言来验证预处理的结果

def test_rplmsed_fit(sample_dataset):
    args = args_define().args
    rplmsed = RPLM_SED(args, sample_dataset)
    rplmsed.fit()
    # 这里可以添加更多的断言来验证模型的训练结果

def test_rplmsed_detection(sample_dataset):
    args = args_define().args
    rplmsed = RPLM_SED(args, sample_dataset)
    predictions, ground_truths = rplmsed.detection()
    assert isinstance(predictions, np.ndarray)
    assert isinstance(ground_truths, np.ndarray)

def test_rplmsed_evaluate(sample_dataset):
    args = args_define().args
    rplmsed = RPLM_SED(args, sample_dataset)
    predictions = np.array([0, 1, 1])
    ground_truths = np.array([0, 1, 0])
    ars, ami, nmi = rplmsed.evaluate(predictions, ground_truths)
    assert isinstance(ars, float)
    assert isinstance(ami, float)
    assert isinstance(nmi, float)

# 测试其他辅助函数
def test_batch_to_tensor():
    from rplmsed import batch_to_tensor
    args = args_define().args
    batch = [
        (1, 1, 0, 1, [0, 1], [101, 102], [0, 1]),
        (0, 2, 1, 2, [1, 0], [103, 104], [1, 0])
    ]
    toks, typs, prefix, tags, events = batch_to_tensor(batch, args)
    assert isinstance(toks, torch.Tensor)
    assert isinstance(typs, torch.Tensor)
    assert isinstance(prefix, torch.Tensor)
    assert isinstance(tags, torch.Tensor)
    assert isinstance(events, torch.Tensor)

def test_create_trainer():
    from rplmsed import create_trainer, get_model
    args = args_define().args
    model = get_model(args)
    optimizer, lr_scheduler = initialize(model, args, 100)
    trainer = create_trainer(model, optimizer, lr_scheduler, args)
    assert isinstance(trainer, torch.nn.Module)

def test_create_evaluator():
    from rplmsed import create_evaluator, get_model
    args = args_define().args
    model = get_model(args)
    evaluator = create_evaluator(model, args)
    assert isinstance(evaluator, torch.nn.Module)

def test_create_tester():
    from rplmsed import create_tester, get_model
    args = args_define().args
    model = get_model(args)
    msg_feats = torch.zeros((10, model.feat_size()), device='cpu')
    ref_num = torch.zeros((10,), dtype=torch.long, device='cpu')
    tester = create_tester(model, args, msg_feats, ref_num)
    assert isinstance(tester, torch.nn.Module)

# 添加更多的测试用例来覆盖其他功能和方法