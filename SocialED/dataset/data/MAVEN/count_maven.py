import pandas as pd
import numpy as np

# 加载数据
load_path = '/home/zhangkun/llm-project/KPGNN/KPGNN-main/datasets/MAVEN/all_df_words_ents_mids.npy'
df_np = np.load(load_path, allow_pickle=True)
print("Loaded data.")
df = pd.DataFrame(data=df_np, columns=['document_ids', 'sentence_ids', 'sentences', 'event_type_ids', 'words', 'unique_words', 'entities', 'message_ids'])
print("Data converted to dataframe.")

# 初始化一个字典来存储每个 event_type_ids 的计数
event_type_counts = {}

# 遍历 event_type_ids 的每个值
for event_type_id in range(1, 169):  # 从 1 到 168
    count = df[df['event_type_ids'] == event_type_id].shape[0]
    if count == 0:
        event_type_counts[event_type_id] = count
        print(f"event_type_ids={event_type_id}: 该类型在KPGNN数据集中已经被删除，并没有此类数据。")
    else:
        event_type_counts[event_type_id] = count
        print(f"event_type_ids={event_type_id}: {count} entries")

# 保存到文本文件
output_path = '/home/zhangkun/llm-project/KPGNN/KPGNN-main/datasets/MAVEN/event_type_counts.txt'  # 更新为实际输出路径
with open(output_path, 'w') as f:
    for event_type_id, count in event_type_counts.items():
        if count == 0:
            f.write(f"event_type_ids={event_type_id}: 该类型在KPGNN数据集中已经被删除，并没有此类数据。\n")
        else:
            f.write(f"event_type_ids={event_type_id}: {count} entries\n")

# 输出数据集的总条数
total_entries = df.shape[0]
print(f"Total number of entries in the dataset: {total_entries}")

# 在文件中也记录总条数
with open(output_path, 'a') as f:
    f.write(f"\nTotal number of entries in the dataset: {total_entries}")


print(f"Data saved to {output_path}.")



