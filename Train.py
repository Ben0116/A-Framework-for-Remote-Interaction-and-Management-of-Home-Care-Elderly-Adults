# -*- coding: utf-8 -*-
"""
Created on Wen Jan 26 18:58:39 2022

@author: ZLQ
"""

# 模型参数训练
#%% 库
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import os
import pandas as pd

#%% 数据预处理
os.chdir(r"E:/DOC_LIFE/DataViualize")
data = pd.read_csv('E:/DOC_LIFE/DataViualize/data_output/trainData.csv')

y = data.target.values
X = data.drop(['target'], axis=1)

# 训练集和测试集
global X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4, stratify=y)

#%% 模型训练

def train(model, param_grid):
    # 寻找最佳参数
    grid_search = GridSearchCV(model, param_grid)
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)
    # 查看模型训练集和测试集的表现
    clf = grid_search.best_estimator_
    print('训练集准确率:',clf.score(X_train, y_train))
    print('测试集准确率:',clf.score(X_test, y_test))


#%% 决策树
DT = DecisionTreeClassifier(random_state=0)
param_grid = {
    'criterion':['entropy','gini'],
    'splitter': ['best','random'],
    'max_depth': [9, 10, 11, 12],
    'min_samples_leaf': [3, 4, 5, 6],
    'max_features':['auto','log2','sqrt']
        }

train(DT,param_grid)


'''
决策树：ourdata
{'criterion': 'gini', 'max_depth': 12, 'min_samples_leaf': 5, 'splitter': 'best'}
训练集：0.9854166666666667
测试集：0.9504132231404959
'''

#%% 随机森林
RF = RandomForestClassifier(random_state=0)

param_grid = {
    'n_estimators' : [8,9,10,11,12,13],
    'criterion':['entropy','gini'],
    'max_depth': [9, 10, 11, 12],
    'min_samples_leaf': [2, 3, 4, 5, 6],
    'bootstrap':[True, False],
    'max_features':['auto','log2','sqrt']
        }

train(RF,param_grid)

'''
随机森林：
{'bootstrap': False, 'criterion': 'gini', 'max_depth': 12, 'max_features': 'log2', 'min_samples_leaf': 2, 'n_estimators': 12}
训练集：1.0
测试集：0.9834710743801653
'''

#%% SVM
SVM = SVC(random_state=0)

param_grid = {
    'C' : [1,2,3,4],
    'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['auto', 0.3, 0.4,0.5,0.6,0.7,0.8]
        }

train(SVM,param_grid)

'''
SVM：
{'C': 3, 'gamma': 0.7, 'kernel': 'rbf'}
训练集：0.9854166666666667
测试集：0.9090909090909091
'''

#%% 逻辑回归

LR = LogisticRegression(random_state=0)

param_grid = {
    'penalty' : ['l2'],
    'C':[0.2,0.4,0.6]
        }

train(LR,param_grid)

'''
逻辑回归：
{'C': 0.4, 'penalty': 'l2'}
训练集：0.6479166666666667
测试集：0.6198347107438017
'''

#%% KNN

KNN = KNeighborsClassifier()

param_grid = {
    'n_neighbors' : [2,3,4],
    'weights':['uniform', 'distance'],
    'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size':[10,20,30]
        }

train(KNN,param_grid)

'''
KNN：
{'algorithm': 'auto', 'leaf_size': 10, 'n_neighbors': 2, 'weights': 'distance'}
训练集：1.0
测试集：0.9090909090909091
'''

#%% 贝叶斯
GNB = GaussianNB()

train(GNB,{})

'''
高斯朴素贝叶斯：无参数
训练集：0.46041666666666664
测试集：0.4132231404958678
'''

#%% 神经网络
MLP = MLPClassifier(random_state=0,max_iter=99999)
param_grid = {
    'hidden_layer_sizes':[(10,10),(5,5),(10),(10,10,10)],
    'learning_rate':['constant', 'invscaling', 'adaptive'],
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'solver':['lbfgs', 'sgd', 'adam']
    }
param_grid = {
    'hidden_layer_sizes':[(10,10)],
    'learning_rate':['constant'],
    'activation': ['tanh'],
    'solver':['lbfgs', 'sgd', 'adam']
    }
train(MLP, param_grid)

'''
{'activation': 'tanh', 'hidden_layer_sizes': (20, 20), 'learning_rate': 'constant', 'solver': 'adam'}
训练集准确率: 1.0
测试集准确率: 0.9090909090909091
'''


