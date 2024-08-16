import pandas as pd
import numpy as np

load_path = '/home/zhangkun/llm-project/KPGNN/KPGNN-main/datasets/MAVEN/all_df_words_ents_mids.npy'
df_np = np.load(load_path, allow_pickle=True)
print("Loaded data.")
df = pd.DataFrame(data=df_np, \
    columns=['document_ids', 'sentence_ids', 'sentences', 'event_type_ids', 'words', 'unique_words', 'entities', 'message_ids'])
print("Data converted to dataframe.")

print(df.shape)
print(df.head(3000))
# 保存 DataFrame 到 CSV 文件
output_path = '/home/zhangkun/llm-project/KPGNN/KPGNN-main/datasets/MAVEN/output_data.csv'  # 指定输出文件的路径
df.head(3000).to_csv(output_path, index=False)  # 只选择前 100 行并保存到 CSV
print(f"Data saved to {output_path}.")

