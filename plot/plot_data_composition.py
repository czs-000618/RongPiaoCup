import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""

    柱状图 数据组成

"""

plt.figure(figsize=(20, 15), dpi=100)
classes = ['limestone', 'mudstone', 'algal']
numbers = [1084, 586, 944]

plt.barh(classes, numbers, height=0.5, color=['lime', 'lightcoral', 'orange'])
plt.xticks(fontproperties="Arial", size=35, weight="normal")
plt.yticks(fontproperties="Arial", size=35, weight="normal")

plt.xlabel("Number", fontdict={'family': 'Times New Roman',
                               'weight': 'normal',
                               'size': 45, })

# plt.ylabel("Types", fontdict={'family': 'Times New Roman',
#                                'weight': 'normal',
#                                'size': 40, })
plt.savefig('Data_Compose.png')
plt.show()
