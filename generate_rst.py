import os

# 你的examples目录路径
examples_dir = 'examples'
# 生成rst文件的目标目录
output_dir = 'docs\source'
print("1")
# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 模板内容
rst_template = """
{module_name} Module
===================

.. automodule:: {module_path}
    :members:
    :undoc-members:
    :show-inheritance:
"""

# 遍历examples目录
for filename in os.listdir(examples_dir):
    if filename.endswith('.py'):
        module_name = filename[:-3]  # 去掉'.py'后缀
        module_path = f'examples.{module_name}'
        
        # 生成rst内容
        rst_content = rst_template.format(module_name=module_name, module_path=module_path)
        
        # 写入rst文件
        rst_filename = os.path.join(output_dir, f'{module_name}.rst')
        with open(rst_filename, 'w', encoding='utf-8') as rst_file:
            rst_file.write(rst_content)
        
        print(f'Generated {rst_filename}')

print('All rst files have been generated.')
