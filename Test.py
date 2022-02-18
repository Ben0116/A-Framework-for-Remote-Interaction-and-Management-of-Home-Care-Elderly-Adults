# -*- coding: utf-8 -*-
"""
Created on Wen Jan 26 19:17:10 2022

@author: ZLQ
"""

# 验证集测试
#%% 库
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import pandas as pd
import os
from drawTools import plot_cnf_matrix

#%% 训练集数据
os.chdir(r"E:/DOC_LIFE/DataViualize")
data = pd.read_csv('E:/DOC_LIFE/DataViualize/data_output/trainData.csv')

y = data.target.values
X = data.drop(['target'], axis=1)

#%% 验证集数据
data = pd.read_csv('E:/DOC_LIFE/DataViualize/data_output/valData.csv')

y_val = data.target.values
X_val = data.drop(['target'], axis=1)

#%% 模型评估

def valuation(y_val,y_pred,model):
    accuracy = accuracy_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted') 
    confusion = confusion_matrix(y_val, y_pred)
    # plot_cnf_matrix(confusion, 'Confusion matrix -- {}'.format(model),model+'.jpg')
    result.loc[len(result)] = [model,accuracy,recall,f1,confusion]
    
#%% 模型测试
global result
result = pd.DataFrame(columns=['model','准确率','召回率','f1','confusion'])
#%% 决策树
DT = DecisionTreeClassifier(
                criterion='gini', 
                max_depth=11,
                min_samples_leaf= 6,
                splitter='best', 
                random_state=0
                )
DT = DT.fit(X, y)
y_pred = DT.predict(X_val)
valuation(y_val,y_pred,"Decision Tree")

#%% 随机森林
RF = RandomForestClassifier(
                bootstrap=False, 
                criterion='gini', 
                max_depth=12, 
                max_features='log2', 
                min_samples_leaf=2, 
                n_estimators=12,
                random_state=0
                )
RF = RF.fit(X, y)
y_pred = RF.predict(X_val)
valuation(y_val,y_pred,"Random Forest")

#%% SVM
SVM = SVC(
                C=3, 
                gamma=0.7, 
                kernel='rbf',
                random_state=0
                )
SVM = SVM.fit(X, y)
y_pred = SVM.predict(X_val)
valuation(y_val,y_pred,"SVM")

#%% 逻辑回归
LR = LogisticRegression(
                C=0.4,
                penalty='l2',
                random_state=0
                )
LR = LR.fit(X, y)
y_pred = LR.predict(X_val)
valuation(y_val,y_pred,"Logistic Regression")

#%% KNN
KNN = KNeighborsClassifier(
                algorithm='auto', 
                leaf_size=10, 
                n_neighbors=2, 
                weights='distance'
                )
KNN = KNN.fit(X, y)
y_pred = KNN.predict(X_val)
valuation(y_val,y_pred,"K Nearest Neighbours")

#%% 贝叶斯
GNB = GaussianNB()
GNB = GNB.fit(X, y)
y_pred = GNB.predict(X_val)
valuation(y_val,y_pred,"GaussianNB Bayes")

#%% BP神经网络
MLP = MLPClassifier(
                activation='tanh', 
                hidden_layer_sizes=(10,10), 
                learning_rate='constant', 
                solver='lbfgs',
                random_state=0,
                max_iter=99999
                )
MLP = MLP.fit(X, y)
y_pred = MLP.predict(X_val)
valuation(y_val,y_pred,"Back Process Neutrual Network")

#%%
result.to_csv('E:/DOC_LIFE/DataViualize/TrainModels/result.csv', index=False, encoding='gbk')




