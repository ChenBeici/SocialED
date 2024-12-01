import pandas as pd
import numpy as np

# 加载数据
load_path = '/home/zhangkun/llm-project/KPGNN/KPGNN-main/datasets/MAVEN/all_df_words_ents_mids.npy'
df_np = np.load(load_path, allow_pickle=True)
print("Loaded data.")

# 设置显示的最大行数，None 表示无限制
pd.set_option('display.max_rows', None)

# 设置显示的最大列数，None 表示无限制
pd.set_option('display.max_columns', None)

df = pd.DataFrame(data=df_np, columns=['document_ids', 'sentence_ids', 'sentences', 'event_type_ids', 'words', 'unique_words', 'entities', 'message_ids'])
print("Data converted to dataframe.")

# 筛选 event_type_ids 为 1 到 10 的数据，并且每种类型只取前 300 条
filtered_data = []
for event_type_id in range(1, 50):  # 从 1 到 10
    filtered_data.append(df[df['event_type_ids'] == event_type_id].head(10))

# 将筛选后的数据合并成一个新的 DataFrame
filtered_df = pd.concat(filtered_data)

# 显示新数据集的形状和头部数据
print(filtered_df.shape)
print(filtered_df.head())
print(filtered_df['sentences'])

# 保存到 CSV 文件
output_path = '/home/zhangkun/llm-project/KPGNN/KPGNN-main/datasets/MAVEN/output_data.csv'  # 更新为实际输出路径
filtered_df.to_csv(output_path, index=False)
print(f"Data saved to {output_path}.")
