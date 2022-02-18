# -*- coding: utf-8 -*-
"""
Created on Wen Jan 26 18:58:39 2022

@author: ZLQ
"""

# 数据预处理
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#%% 加载数据
#%% data = pd.read_csv('E:/DOC_LIFE/DataViualize/data_source/alldays_stage2.csv')
data = pd.read_csv('E:/DOC_LIFE/DataViualize/data_source/alldays.csv')
y = data.target.values
X = data.drop(['target'], axis=1)

# 标准化
ss = StandardScaler()
X = ss.fit_transform(X)

# 数据不平衡问题  494 vs 760 还可以
# count = data[data['target']==1]

#%% 切分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
# stratify-保持两个数据集的比例

data_val = pd.DataFrame(X_val, columns=data.columns.values.tolist()[:-1])
data_val['target'] = y_val
data_train = pd.DataFrame(X_train, columns=data.columns.values.tolist()[:-1])
data_train['target'] = y_train

data_val.to_csv('E:/DOC_LIFE/DataViualize/data_output/valData.csv', index=False)
data_train.to_csv('E:/DOC_LIFE/DataViualize/data_output/trainData.csv', index=False)
