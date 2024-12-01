import numpy as np
import os
import pandas as pd
import zipfile
import io
import requests
import os


class MAVEN_Dataset:
    default_path = './data/MAVEN/'
    default_file = 'all_df_words_ents_mids.npy'

    @staticmethod
    def load_data(file_paths=None):
        """
        Load and concatenate data from multiple .npy files.

        :param file_paths: List of file paths to load data from. If None, default paths will be used.
        :return: Concatenated numpy array.
        """

        file_path = str(MAVEN_Dataset.default_path) + str(MAVEN_Dataset.default_file)

        df_np = np.load(file_path, allow_pickle=True)
        print("Data loaded.")
        df = pd.DataFrame(data=df_np, \
            columns=['document_ids', 'sentence_ids', 'sentences', 'event_id', 'words', 'unique_words', 'entities', 'message_ids'])
        print("df_np converted to dataframe.")

        return df

class Event2012_Dataset:
    default_path = 'data/Event2012/'
    default_files = [
        '68841_tweets_multiclasses_filtered_0722_part1.npy',
        '68841_tweets_multiclasses_filtered_0722_part2.npy'
    ]

    @staticmethod
    def load_data(file_paths=None):
        """
        Load and concatenate data from multiple .npy files.

        :param file_paths: List of file paths to load data from. If None, default paths will be used.
        :return: Concatenated numpy array.
        """
        if file_paths is None:
            file_paths = [os.path.join(Event2012_Dataset.default_path, file) for file in Event2012_Dataset.default_files]

        data_list = []
        for file_path in file_paths:
            data = np.load(file_path, allow_pickle=True)
            data_list.append(data)

        if data_list:
            concatenated_data = np.concatenate(data_list, axis=0)
        else:
            concatenated_data = np.array([])

        df = pd.DataFrame(concatenated_data, columns=[
            "event_id", "tweet_id", "text", "user_id", "created_at", "user_loc",
            "place_type", "place_full_name", "place_country_code", "hashtags", 
            "user_mentions", "image_urls", "entities", "words", "filtered_words", 
            "sampled_words"
        ])    
        print("df_np converted to dataframe.")

        return df

class Event2018_Dataset:
    default_path = 'data/Event2018/'
    default_files = [
        'french_tweets.npy'
    ]

    @staticmethod
    def load_data(file_paths=None):
        """
        Load and concatenate data from multiple .npy files.

        :param file_paths: List of file paths to load data from. If None, default paths will be used.
        :return: Concatenated numpy array.
        """
        if file_paths is None:
            file_paths = [os.path.join(Event2018_Dataset.default_path, file) for file in Event2018_Dataset.default_files]

        data_list = []
        for file_path in file_paths:
            data = np.load(file_path, allow_pickle=True)
            data_list.append(data)

        if data_list:
            concatenated_data = np.concatenate(data_list, axis=0)
        else:
            concatenated_data = np.array([])

        df = pd.DataFrame(concatenated_data, columns=[
                "tweet_id", "user_id", "text", "time", "event_id", "user_mentions",
                "hashtags", "urls", "words", "created_at", "filtered_words", "entities",
                "sampled_words"])
        print("df_np converted to dataframe.")
        
        return df

class Arabic_Dataset:
    default_path = 'data/Arabic_Twitter/'
    default_files = [
        'All_Arabic.npy'
    ]

    @staticmethod
    def load_data(file_paths=None):
        """
        Load and concatenate data from multiple .npy files.

        :param file_paths: List of file paths to load data from. If None, default paths will be used.
        :return: Concatenated numpy array.
        """
        if file_paths is None:
            file_paths = [os.path.join(Arabic_Dataset.default_path, file) for file in Arabic_Dataset.default_files]

        data_list = []
        for file_path in file_paths:
            data = np.load(file_path, allow_pickle=True)
            data_list.append(data)

        if data_list:
            concatenated_data = np.concatenate(data_list, axis=0)
        else:
            concatenated_data = np.array([])

        df = pd.DataFrame(concatenated_data, columns=[
                "tweet_id", "user_id", "text", "time", "event_id", "user_mentions",
                "hashtags", "urls", "words", "created_at", "filtered_words", "entities",
                "sampled_words"])
        print("df_np converted to dataframe.")

        return df

class Dataset(object):
    def __init__(self, is_normalize: bool = False, root: str = "./dataset") -> None:


        self.root = root
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        self.path_name = ""

    def download(self, url: str, filename: str):
        r = requests.get(url)
        assert r.status_code == 200
        open(os.path.join(self.root, self.path_name, filename), "wb").write(r.content)

    def download_zip(self, url: str):
        r = requests.get(url)
        assert r.status_code == 200
        foofile = zipfile.ZipFile(io.BytesIO(r.content))
        foofile.extractall(os.path.join(self.root, self.path_name))

class DatasetLoader:
    def __init__(self, dataset):
        self.dataset = dataset.lower()
    
    def load_data(self):
        if self.dataset == 'maven':
            return MAVEN_Dataset.load_data()
        elif self.dataset == 'event2012':
            return Event2012_Dataset.load_data()
        elif self.dataset == 'event2018':
            return Event2018_Dataset.load_data()
        elif self.dataset == 'arabic_twitter':
            return Arabic_Dataset.load_data()
        else:
            raise ValueError(f"Unsupported language: {self.dataset}")


if __name__ == "__main__":
    loader = DatasetLoader(dataset='maven')
    df = loader.load_data()

    # Print the first few rows of the dataframe to verify
    print(df.head())


