import os

# 定义要处理的目录为当前目录
directory = "."

# 获取当前目录中的所有文件
files = os.listdir(directory)

# 遍历所有文件
for filename in files:
    # 检查是否为 Python 文件
    if filename.endswith(".py"):
        # 将文件名改为小写
        new_filename = filename.lower()
        
        # 生成完整的旧文件路径和新文件路径
        old_file = os.path.join(directory, filename)
        new_file = os.path.join(directory, new_filename)
        
        # 重命名文件
        os.rename(old_file, new_file)
        print(f'Renamed: {filename} -> {new_filename}')

print("All files have been renamed to lowercase.")
