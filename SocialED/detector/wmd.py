import os
import pandas as pd
import numpy as np
import datetime
from gensim.models import Word2Vec
from gensim.similarities import WmdSimilarity
from sklearn.model_selection import train_test_split
from sklearn import metrics
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.dataloader import Event2012


# event_id, filtered_words
class WMD:

    r"""The WMD model for social event detection that uses Word Mover's Distance
    to measure document similarity and detect events.

    .. note::
        This detector uses word embeddings and Word Mover's Distance to identify similar documents
        and detect events in social media data. The model requires a dataset object with a 
        load_data() method.

    Parameters
    ----------
    dataset : object
        The dataset object containing social media data.
        Must provide load_data() method that returns the raw data.
    vector_size : int, optional
        Dimensionality of word vectors. Default: ``100``.
    window : int, optional
        Maximum distance between current and predicted word. Default: ``5``.
    min_count : int, optional
        Minimum word frequency. Default: ``1``.
    sg : int, optional
        Training algorithm: Skip-gram (1) or CBOW (0). Default: ``1``.
    num_best : int, optional
        Number of best matches to return. Default: ``5``.
    threshold : float, optional
        Similarity threshold for event detection. Default: ``0.6``.
    batch_size : int, optional
        Batch size for processing. Default: ``1000``.
    n_workers : int, optional
        Number of worker processes. Default: ``CPU count - 1``.
    file_path : str, optional
        Path to save model files. Default: ``'../model/model_saved/WMD/'``.
    """
    def __init__(self,
                 dataset,
                 vector_size=100,
                 window=5,
                 min_count=1,
                 sg=1,
                 num_best=5,
                 threshold=0.6,
                 batch_size=1000,  # 新增：批处理大小
                 n_workers=None,   # 新增：进程数
                 file_path='../model/model_saved/WMD/'):
        self.dataset = dataset.load_data()
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.sg = sg
        self.num_best = num_best
        self.threshold = threshold
        self.batch_size = batch_size
        self.n_workers = n_workers or max(1, multiprocessing.cpu_count() - 1)
        self.file_path = file_path
        self.df = None
        self.train_df = None
        self.test_df = None
        self.word2vec_model = None
        self.model_path = os.path.join(self.file_path, 'word2vec_model.model')

    def preprocess(self):
        """
        优化的数据预处理
        """
        df = self.dataset[['filtered_words', 'event_id']].copy()
        # 使用列表推导式优化处理速度
        df['processed_text'] = [
            [str(word).lower() for word in x] if isinstance(x, list) else []
            for x in df['filtered_words']
        ]
        # 过滤掉空文档
        df = df[df['processed_text'].map(len) > 0]
        self.df = df
        return df

    def fit(self):
        """
        Train the Word2Vec model and save it to a file.
        """
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        train_df, test_df = train_test_split(self.df, test_size=0.2, random_state=42)
        self.train_df = train_df
        self.test_df = test_df

        sentences = train_df['processed_text'].tolist()

        print("Training Word2Vec model...")
        word2vec_model = Word2Vec(sentences=sentences, 
                                vector_size=self.vector_size,
                                window=self.window,
                                min_count=self.min_count,
                                sg=self.sg)
        print("Word2Vec model trained successfully.")

        word2vec_model.save(self.model_path)
        print(f"Word2Vec model saved to {self.model_path}")

        self.word2vec_model = word2vec_model.wv

    def detection(self):
        """
        优化的事件检测
        """
        if self.word2vec_model is None:
            word2vec_model = Word2Vec.load(self.model_path)
            self.word2vec_model = word2vec_model.wv

        test_corpus = self.test_df['processed_text'].tolist()
        train_corpus = self.train_df['processed_text'].tolist()

        print("Calculating WMD distances...")
        instance = WmdSimilarity(train_corpus, self.word2vec_model, num_best=self.num_best)

        # 使用多进程处理文档
        process_doc = partial(
            process_document,
            instance=instance,
            train_df=self.train_df,
            threshold=self.threshold,
            num_best=self.num_best
        )

        predictions = []
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # 批处理文档
            for i in tqdm(range(0, len(test_corpus), self.batch_size)):
                batch = test_corpus[i:i + self.batch_size]
                batch_predictions = list(executor.map(process_doc, batch))
                predictions.extend(batch_predictions)

        # 处理未分配事件的文档
        max_event_id = max(self.train_df['event_id'])
        new_event_counter = 1
        for i in range(len(predictions)):
            if predictions[i] == -1:
                predictions[i] = max_event_id + new_event_counter
                new_event_counter += 1

        ground_truths = self.test_df['event_id'].tolist()

        # 保存结果
        self._save_results(ground_truths, predictions)

        return ground_truths, predictions

    def _save_results(self, ground_truths, predictions):
        """
        保存结果的辅助方法
        """
        unique_ground_truths = list(set(ground_truths))
        unique_predictions = list(set(predictions))

        max_len = max(len(unique_ground_truths), len(unique_predictions))
        unique_ground_truths.extend([None] * (max_len - len(unique_ground_truths)))
        unique_predictions.extend([None] * (max_len - len(unique_predictions)))

        data = {
            'Unique Ground Truths': unique_ground_truths,
            'Unique Predictions': unique_predictions
        }
        df = pd.DataFrame(data)
        output_file = os.path.join(self.file_path, "unique_ground_truths_predictions.csv")
        df.to_csv(output_file, index=False)
        print(f"Unique ground truths and predictions have been saved to {output_file}")

    def evaluate(self, ground_truths, predictions):
        """
        Evaluate the model and save results.
        """
        nmi = metrics.normalized_mutual_info_score(ground_truths, predictions)
        print(f"Normalized Mutual Information (NMI): {nmi}")

        ami = metrics.adjusted_mutual_info_score(ground_truths, predictions)
        print(f"Adjusted Mutual Information (AMI): {ami}")

        ari = metrics.adjusted_rand_score(ground_truths, predictions)
        print(f"Adjusted Rand Index (ARI): {ari}")

        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.model_path + "_evaluation.txt", "a") as f:
            f.write(f"Date and Time: {current_datetime}\n")
            f.write(f"Normalized Mutual Information (NMI): {nmi}\n")
            f.write(f"Adjusted Mutual Information (AMI): {ami}\n")
            f.write(f"Adjusted Rand Index (ARI): {ari}\n")
            f.write("\n")

# 新增：计算单个文档的相似度
def process_document(doc, instance, train_df, threshold, num_best):
    sims = instance[doc]
    similar_events = []
    for idx, score in sims[:num_best]:  # 只处理前num_best个结果
        if score > threshold:
            similar_events.append(train_df.iloc[idx]['event_id'])
    
    if similar_events:
        prediction = max(set(similar_events), key=similar_events.count)
    else:
        prediction = -1  # 使用临时标记，后续处理
    
    return prediction




