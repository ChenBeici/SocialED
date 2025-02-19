import pytest
import torch
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detector.clkd import *

# Create args object

args = args_define().args
# Test the CLKD class initialization
def test_clkd_initialization():
    clkd = CLKD(args)
    assert clkd.args == args

# Test the preprocess method
def test_preprocess():
    print(args)
    clkd = CLKD(args)
    clkd.preprocess()
    # Add assertions to check if the preprocessing steps were successful
    assert clkd.embedding_save_path is not None
    assert clkd.data_split is not None

# Test the fit method
def test_fit():
    clkd = CLKD(args)
    clkd.preprocess()
    clkd.fit()
    # Add assertions to check if the model was trained successfully
    assert clkd.model is not None

# Test the detection method
def test_detection():
    clkd = CLKD(args)
    clkd.preprocess()
    clkd.fit()
    predictions, ground_truths = clkd.detection()
    # Add assertions to check if the detection was successful
    assert isinstance(predictions, np.ndarray)
    assert isinstance(ground_truths, np.ndarray)

# Test the evaluate method
def test_evaluate():
    clkd = CLKD(args)
    clkd.preprocess()
    clkd.fit()
    predictions, ground_truths = clkd.detection()
    results = clkd.evaluate(predictions, ground_truths)
    # Add assertions to check if the evaluation was successful
    assert isinstance(results, tuple)
    assert len(results) == 3
    assert all(isinstance(val, float) for val in results)

# Test the evaluate2 method
def test_evaluate2():
    clkd = CLKD(args)
    clkd.preprocess()
    clkd.fit()
    predictions, ground_truths = clkd.detection()
    ars, ami, nmi = clkd.evaluate2(predictions, ground_truths)
    # Add assertions to check if the evaluation2 was successful
    assert isinstance(ars, float)
    assert isinstance(ami, float)
    assert isinstance(nmi, float)

# Test the generate_initial_features method
def test_generate_initial_features():
    preprocessor = CLKD(args).preprocess()
    preprocessor.generate_initial_features()
    # Add assertions to check if the initial features were generated successfully
    assert os.path.exists(args.file_path + '/features/features_69612_0709_spacy_lg_zero_multiclasses_filtered_English.npy')

# Test the construct_graph method
def test_construct_graph():
    preprocessor = CLKD(args).preprocess()
    preprocessor.construct_graph()
    # Add assertions to check if the graph was constructed successfully
    assert os.path.exists(args.file_path + '/English/0/s_bool_A_tid_tid.npz')

# Test the extract_time_feature method
def test_extract_time_feature():
    preprocessor = CLKD(args).preprocess()
    time_feature = preprocessor.extract_time_feature('2012-10-11 07:19:34')
    # Add assertions to check if the time feature was extracted successfully
    assert isinstance(time_feature, list)
    assert len(time_feature) == 2

# Test the documents_to_features method
def test_documents_to_features():
    preprocessor = CLKD(args).preprocess()
    df = DatasetLoader('event2012').load_data()
    features = preprocessor.documents_to_features(df, 'English')
    # Add assertions to check if the document features were generated successfully
    assert isinstance(features, np.ndarray)
    assert features.shape[1] == 300

# Test the get_word2id_emb method
def test_get_word2id_emb():
    preprocessor = CLKD(args).preprocess()
    word2id, embeddings = preprocessor.get_word2id_emb(args.wordpath, args.embpath)
    # Add assertions to check if the word2id and embeddings were loaded successfully
    assert isinstance(word2id, dict)
    assert isinstance(embeddings, np.ndarray)

# Test the nonlinear_transform_features method
def test_nonlinear_transform_features():
    preprocessor = CLKD(args).preprocess()
    df = DatasetLoader('event2012').load_data()
    features = preprocessor.nonlinear_transform_features(args.wordpath, args.embpath, df)
    # Add assertions to check if the nonlinear transformed features were generated successfully
    assert isinstance(features, np.ndarray)
    assert features.shape[1] == 300

# Test the getlinear_transform_features method
def test_getlinear_transform_features():
    preprocessor = CLKD(args).preprocess()
    df = DatasetLoader('event2012').load_data()
    features = preprocessor.documents_to_features(df, 'English')
    transformed_features = preprocessor.getlinear_transform_features(features, 'English', 'French')
    # Add assertions to check if the linear transformed features were generated successfully
    assert isinstance(transformed_features, np.ndarray)
    assert transformed_features.shape[1] == 300

# Test the construct_graph_from_df method
def test_construct_graph_from_df():
    preprocessor = CLKD(args).preprocess()
    df = DatasetLoader('event2012').load_data()
    G = preprocessor.construct_graph_from_df(df)
    # Add assertions to check if the graph was constructed successfully
    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() > 0

# Test the networkx_to_dgl_graph method
def test_networkx_to_dgl_graph():
    preprocessor = CLKD(args).preprocess()
    df = DatasetLoader('event2012').load_data()
    G = preprocessor.construct_graph_from_df(df)
    all_mins, message = preprocessor.networkx_to_dgl_graph(G)
    # Add assertions to check if the graph was converted successfully
    assert isinstance(all_mins, float)
    assert isinstance(message, str)

# Test the construct_incremental_dataset method
def test_construct_incremental_dataset():
    preprocessor = CLKD(args).preprocess()
    df = DatasetLoader('event2012').load_data()
    features = np.random.rand(df.shape[0], 300)
    nfeatures = np.random.rand(df.shape[0], 300)
    message, data_split, all_graph_mins = preprocessor.construct_incremental_dataset(args, df, args.file_path + '/English', features, nfeatures)
    # Add assertions to check if the incremental dataset was constructed successfully
    assert isinstance(message, str)
    assert isinstance(data_split, list)
    assert isinstance(all_graph_mins, list)

# Test the SocialDataset class
def test_social_dataset():
    dataset = SocialDataset(args.data_path, 0)
    # Add assertions to check if the dataset was loaded successfully
    assert isinstance(dataset.features, np.ndarray)
    assert isinstance(dataset.labels, np.ndarray)
    assert isinstance(dataset.matrix, sparse.csr_matrix)

# Test the graph_statistics function
def test_graph_statistics():
    preprocessor = CLKD(args).preprocess()
    df = DatasetLoader('event2012').load_data()
    G = preprocessor.construct_graph_from_df(df)
    num_isolated_nodes = graph_statistics(G, args.file_path + '/English/0')
    # Add assertions to check if the graph statistics were calculated successfully
    assert isinstance(num_isolated_nodes, int)

# Run the tests
if __name__ == '__main__':
    pytest.main()