import pandas as pd
import numpy as np

load_path = './datasets/FRENCH/All_French.npy'
df_np = np.load(load_path, allow_pickle=True)
print("Loaded data.")
df = pd.DataFrame(data=df_np, \
    columns=["tweet_id", "user_name", "text", "time", "event_id", "user_mentions","hashtags", 
    "urls", "words", "created_at", "filtered_words","entities","sampled_words"])
print("Data converted to dataframe.")

print(df.shape)
print(df.head(100))

# 保存 DataFrame 到 CSV 文件
output_path = '/home/zhangkun/KPGNN/KPGNN-main/datasets/FRENCH/output_data.csv'  # 指定输出文件的路径
df.head(100).to_csv(output_path, index=False)  # 只选择前 100 行并保存到 CSV
print(f"Data saved to {output_path}.")