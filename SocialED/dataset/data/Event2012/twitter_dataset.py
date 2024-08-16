import pandas as pd
import numpy as np

# 加载数据部分
p_part1 = '/home/zhangkun/llm-project/KPGNN/KPGNN-main/datasets/Twitter/68841_tweets_multiclasses_filtered_0722_part1.npy'
p_part2 = '/home/zhangkun/llm-project/KPGNN/KPGNN-main/datasets/Twitter/68841_tweets_multiclasses_filtered_0722_part2.npy'
df_np_part1 = np.load(p_part1, allow_pickle=True)
df_np_part2 = np.load(p_part2, allow_pickle=True)
df_np = np.concatenate((df_np_part1, df_np_part2), axis = 0)
print("Loaded data.")

# 将 numpy 数组转换为 pandas DataFrame
df = pd.DataFrame(data=df_np, columns=["event_id", "tweet_id", "text", "user_id", "created_at", "user_loc",\
    "place_type", "place_full_name", "place_country_code", "hashtags", "user_mentions", "image_urls", "entities", 
    "words", "filtered_words", "sampled_words"])
print("Data converted to dataframe.")

# 打印 DataFrame 的形状和头部数据
print(df.shape)
print(df.head(10))

# 筛选 event_type_ids 为 1 到 10 的数据，并且每种类型只取前 300 条
filtered_data = []
for event_id in range(1, 800):  # 从 1 到 10
    filtered_data.append(df[df['event_id'] == event_id].head(10))

filtered_df = pd.concat(filtered_data)


# 保存 DataFrame 到 CSV 文件
output_path = '/home/zhangkun/llm-project/KPGNN/KPGNN-main/datasets/Twitter/output_data.csv'  # 指定输出文件的路径
filtered_df.to_csv(output_path, index=False)  # 只选择前 100 行并保存到 CSV
print(f"Data saved to {output_path}.")



