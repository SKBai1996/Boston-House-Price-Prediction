#!usr/bin/env python
#-*- coding:-utf-8 -*-
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import pdb



def loadData():
    boston = load_boston()
    data = boston['data']
    target = boston['target']
    featureName = boston['feature_names']
    df_data = pd.DataFrame(data, index=range(len(target)), columns=featureName)
    df_target = pd.DataFrame(target, index=range(len(target)), columns=['MEDV'])

    return data,target

def train_test_split(data,label,testsize=0.2):
     ranges = range(data.shape[0])
     test_length = int(len(ranges)*testsize)
     test_index = random.sample(ranges,test_length)
     test_index.sort()
     train_index = list(set(ranges)^set(test_index))
     testset = data[test_index]
     test_label = label[test_index]
     trainset = data[train_index]
     train_label = label[train_index]

     return trainset,train_label,testset,test_label

def paint(true_value,pred_value):
    plt.figure(figsize=(20, 8), dpi=80)
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False
    x = range(len(true_value))
    plt.plot(x, true_value, linestyle='-')
    plt.plot(x, pred_value, linestyle='-.')
    plt.legend([u'真实房价', u'预测房价'],fontsize='xx-large')
    plt.title(u'波士顿房价走势图',fontsize='xx-large')
    plt.show()


###########################################################################
def simple_Regress(x_train,y_train):
    x_mean = np.mean(x_train,axis=0)
    y_mean = np.mean(y_train,axis=0)
    b1= np.sum((x_train-x_mean)*(y_train-y_mean))

def standard_Regress(x_train,y_train):
    x_Mat = np.mat(x_train);y_Mat = np.mat(y_train)
    xTx = x_Mat.T*x_Mat   #(405,13).T*(405,13)->(13,13)
    if np.linalg.det(xTx) == 0.0:
        return
    ws = xTx.I*(x_Mat.T*y_Mat.T)
    return ws

if __name__ == '__main__':
    data, target = loadData()
    x_train, y_train, x_test, y_test = train_test_split(data, target)
    ws = standard_Regress(x_train, y_train)
    y_pred = x_test * ws
    paint(y_test, y_pred)