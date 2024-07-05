# socialED/datasets/Event2012.py

import numpy as np
import os

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

        return concatenated_data

# 示例使用
if __name__ == "__main__":
    data = Event2012_Dataset.load_data()
    print(data.shape)
