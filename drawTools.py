"""
专门用来绘图的函数
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 解决matplotlib中文问题
from pylab import mpl, Text, time
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# 绘制ROC曲线
def plot_roc_curve(fprs, tprs):
    plt.figure(figsize=(8, 6), dpi=80)
    plt.plot(fprs, tprs)
    plt.plot([0, 1], linestyle='--')
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.ylabel('TP rate', fontsize=15)
    plt.xlabel('FP rate', fontsize=15)
    plt.title('ROC曲线', fontsize=17)
    plt.show()


# 绘制混淆矩阵
def plot_cnf_matrix(cnf_matrix, description,path):
    class_names = [0, 1]
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap='OrRd',
                fmt='g')
    ax.xaxis.set_label_position('top')
    plt.tight_layout()
    plt.title(description, y=1.1, fontsize=16)
    plt.ylabel('实际值0/1', fontsize=12)
    plt.xlabel('预测值0/1', fontsize=12)
    plt.show()
    plt.savefig(path)