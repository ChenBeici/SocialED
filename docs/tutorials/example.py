"""
This example shows how to plot a simple line chart using matplotlib with SocialED data.

Section 1: Data Preparation
---------------------------

We start by creating a simple dataset.
"""

import matplotlib.pyplot as plt

# 数据准备
x = [1, 2, 3, 4]
y = [10, 15, 13, 17]

# 绘制折线图
plt.plot(x, y)
plt.title("Simple Line Chart")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()  # 这个命令会生成并显示图表
