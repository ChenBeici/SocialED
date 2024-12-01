import numpy as np
import os
from git import Repo, GitCommandError
import pandas as pd
import shutil
from uuid import uuid4
from datetime import datetime

def download_and_cleanup(repo_url, local_repo_path, target_folder, local_target_folder):
    try:
        # Clone the repository or pull the latest changes
        if not os.path.exists(local_repo_path):
            print(f"Cloning repository from {repo_url} to {local_repo_path}")
            Repo.clone_from(repo_url, local_repo_path)
        else:
            repo = Repo(local_repo_path)
            print(f"Pulling latest changes from {repo_url}")
            repo.remotes.origin.pull()

        # Ensure the target folder exists
        if not os.path.exists(local_target_folder):
            os.makedirs(local_target_folder)

        # Copy the target folder to the local target folder
        source_folder = os.path.join(local_repo_path, target_folder)
        if os.path.exists(source_folder):
            print(f"Copying {source_folder} to {local_target_folder}")
            shutil.copytree(source_folder, local_target_folder, dirs_exist_ok=True)
            print(f"Folder {target_folder} has been successfully downloaded to {local_target_folder}")

            # Clean up unnecessary files and folders
            for root, dirs, files in os.walk(local_repo_path, topdown=False):
                for name in files:
                    file_path = os.path.join(root, name)
                    if not file_path.startswith(source_folder):
                        os.remove(file_path)
                for name in dirs:
                    dir_path = os.path.join(root, name)
                    if not dir_path.startswith(source_folder) and dir_path != local_repo_path:
                        shutil.rmtree(dir_path)

            print("Cleanup completed, only the required folder is retained")
        else:
            print(f"Target folder {source_folder} does not exist")

    except GitCommandError as e:
        print(f"Git command error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


# 下载数据库
def download():
    # 使用示例
    repo_url = "git@gitee.com:fuoct/SocialED.git"  # Gitee 仓库 URL
    local_repo_path = "./datatmp"  # 本地git库
    target_folder = "data"  # 项目内文件夹路径
    local_target_folder = "../dataset/data"  # 目标文件夹路径，这样是和tttttest同级别

    download_and_cleanup(repo_url, local_repo_path, target_folder, local_target_folder)
    shutil.rmtree("./datatmp")


class DatasetLoader:
    def __init__(self, dataset, dir_path=None):
        self.dir_path = dir_path
        self.dataset = dataset
        self.default_root_path = "../dataset/data/"

    def load_data(self):
        # print(type(self))
        if self.dataset == 'MAVEN':
            default_path = self.default_root_path + 'MAVEN/' if self.dir_path is None else self.dir_path
            default_files = ['all_df_words_ents_mids.npy']
            columns = ['document_ids', 'sentence_ids', 'sentences', 'event_id', 'words', 'filtered_words',
                       'entities', 'message_ids']
            # print(os.path.abspath(default_path))
            if not os.path.exists(default_path):
                download()

        elif self.dataset == 'Event2012':
            # return Event2012_Dataset.load_data(self.dir_path)
            default_path = self.default_root_path + 'Event2012/' if self.dir_path is None else self.dir_path
            default_files = [
                '68841_tweets_multiclasses_filtered_0722_part1.npy',
                '68841_tweets_multiclasses_filtered_0722_part2.npy'
            ]
            columns = [
                "event_id", "tweet_id", "text", "user_id", "created_at", "user_loc",
                "place_type", "place_full_name", "place_country_code", "hashtags",
                "user_mentions", "urls", "entities", "words", "filtered_words",
                "sampled_words"
            ]
            # print(os.path.abspath(default_path))
            # print(os.path.dirname(default_path))
            if not os.path.exists(default_path):
                download()
        elif self.dataset == 'Event2018':
            # return Event2018_Dataset.load_data(self.dir_path)
            default_path = self.default_root_path + 'Event2018/' if self.dir_path is None else self.dir_path
            default_files = [
                'french_tweets.npy'
            ]
            columns = [
                "tweet_id", "user_name", "text", "time", "event_id", "user_mentions",
                "hashtags", "urls", "words", "created_at", "filtered_words", "entities",
                "sampled_words"]
            # print(os.path.abspath(default_path))
            if not os.path.exists(default_path):
                download()
        elif self.dataset == 'Arabic_Twitter':
            default_path = self.default_root_path + 'Arabic_Twitter/' if self.dir_path is None else self.dir_path
            default_files = [
                'All_Arabic.npy'
            ]
            columns = [
                "tweet_id", "user_name", "text", "time", "event_id", "user_mentions",
                "hashtags", "urls", "words", "created_at", "filtered_words", "entities",
                "sampled_words"]


        
        else:
            supported_datasets = ['MAVEN', 'Event2012', 'Event2018', 'Arabic_Twitter']
            print(f"Unsupported dataset: {self.dataset}. Supported datasets are: {', '.join(supported_datasets)}")
            raise ValueError(f"Unsupported language: {self.dataset}")


        file_paths = [os.path.join(default_path, file) for file in default_files]
        data_list = [np.load(file_path, allow_pickle=True) for file_path in file_paths]
        concatenated_data = np.concatenate(data_list, axis=0) if data_list else np.array([])

        df = pd.DataFrame(concatenated_data, columns=columns)

        if self.dataset in ['Event2018']:
            df['user_id'] = [[] for _ in range(len(df))]
        if self.dataset == 'Arabic_Twitter':
            df['user_id'] = [[] for _ in range(len(df))]
            df['event_id'], _ = pd.factorize(df['event_id'])
        if self.dataset == 'MAVEN':
            df['urls'] = [[] for _ in range(len(df))]
            df['created_at'] = [[] for _ in range(len(df))]   
            df['user_mentions'] = [[] for _ in range(len(df))]
            df['user_id'] = [[] for _ in range(len(df))]
            df['hashtags'] = [[] for _ in range(len(df))]
            df = df.rename(columns={'message_ids': 'tweet_id'})
            df = df.rename(columns={'sentences': 'text'})
        
        df = check_and_filter_columns(df)

        print("df_np converted to dataframe.")
        return df

    def get_dataset_language(self):
        """
        Determine the language based on the current dataset.
        
        Returns:
            str: The language of the dataset ('English', 'French', 'Arabic').
        """
        # 语言映射表
        dataset_language_map = {
            'MAVEN': 'English',
            'Event2012': 'English',
            'Event2018': 'French',
            'Arabic_Twitter': 'Arabic'
        }
        
        language = dataset_language_map.get(self.dataset)
        if not language:
            raise ValueError(f"Unsupported dataset: {self.dataset}. Supported datasets are: {', '.join(dataset_language_map.keys())}")
        return language


class DatasetLoader_mini(DatasetLoader):
    def __init__(self, dataset, dir_path=None):
        # 初始化时调用父类构造函数
        super().__init__(dataset, dir_path)
        
    def load_data(self):
        """
        Load data, but return only one-tenth of the data.
        """
        # 调用父类的 load_data 方法加载完整数据
        df = super().load_data()
        
        # 下采样：随机选择十分之一的数据
        df_mini = df.sample(frac=0.1, random_state=42).reset_index(drop=True)
        
        print(f"Data sample size: {len(df_mini)} (1/10th of original dataset)")
        
        return df_mini




def check_and_filter_columns(df):
    # Define the required columns in the exact order they should appear
    required_columns = [
        'tweet_id', 'text', 'event_id', 'words', 'filtered_words',
        'entities', 'user_id', 'created_at', 'urls', 'hashtags', 'user_mentions'
    ]
    
    # Check if the DataFrame contains all the required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if len(missing_columns) == 0:
        # If no columns are missing, filter the DataFrame to include only the required columns
        df_filtered = df[required_columns]
        return df_filtered
    else:
        # If there are missing columns, return a message with the missing columns
        return f"Missing columns: {', '.join(missing_columns)}"


def save_head_100_csv(datasets=None, output_dir=None):
    """
    Process the specified datasets and save the first 100 rows (processed and raw) as CSV files.

    Parameters:
        datasets (list): List of dataset names to process. Defaults to ['MAVEN', 'Event2012', 'Event2018', 'Arabic_Twitter'].
        output_dir (str): Directory to save the output CSV files. Defaults to the dataset's original directory.
    """
    if datasets is None:
        datasets = ['MAVEN', 'Event2012', 'Event2018', 'Arabic_Twitter']

    # Iterate through each dataset
    for dataset_name in datasets:
        print(f"Loading dataset: {dataset_name}")
        
        # Create an instance of DatasetLoader
        loader = DatasetLoader(dataset_name)
        
        # Load the processed data
        df = loader.load_data()

        # Keep only the first 100 rows
        df_head_100 = df.head(100)

        # Load the first 100 rows of the raw data (assume raw data is stored in the dataset's directory)
        # Here, raw data is directly read from the .npy file
        default_path = loader.default_root_path + dataset_name + "/"
        raw_data_file = {
            'MAVEN': 'all_df_words_ents_mids.npy',
            'Event2012': '68841_tweets_multiclasses_filtered_0722_part1.npy',
            'Event2018': 'french_tweets.npy',
            'Arabic_Twitter': 'All_Arabic.npy'
        }

        # Load raw data
        raw_data = np.load(os.path.join(default_path, raw_data_file[dataset_name]), allow_pickle=True)
        raw_df = pd.DataFrame(raw_data)

        # Keep only the first 100 rows of the raw data
        raw_df_head_100 = raw_df.head(100)

        # Ensure the output directory exists (default is the dataset's directory)
        if output_dir is None:
            output_dir = default_path
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Generate CSV file paths
        processed_csv_path = os.path.join(output_dir, f"{dataset_name}_head_100.csv")
        raw_csv_path = os.path.join(output_dir, f"{dataset_name}_head_100raw.csv")

        # Save the first 100 rows of processed data
        df_head_100.to_csv(processed_csv_path, index=False)
        print(f"Processed data for {dataset_name} saved as CSV: {processed_csv_path}")

        # Save the first 100 rows of raw data
        raw_df_head_100.to_csv(raw_csv_path, index=False)
        print(f"Raw data for {dataset_name} saved as CSV: {raw_csv_path}")


if __name__ == "__main__":
    
    loader = DatasetLoader(dataset='maven')
    df = loader.load_data()

    # Print the first few rows of the dataframe to verify
    print(df.head())
    