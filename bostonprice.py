#!usr/bin/env python
#-*- coding:-utf-8 -*-
from scipy.io import loadmat
from sklearn.datasets import load_boston
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
import pandas as pd
import random
import pdb

def loadData():
    boston = load_boston()
    data = boston['data']
    target = boston['target']
    
    return data,target

def feature_choice(data):
    estimator = PCA(n_components=10)
    #estimator.fit(data)
    data_pca = estimator.fit_transform(data)
    #pdb.set_trace()
    return data_pca
def selectfeature(data, target):
    select_data = SelectKBest(chi2, k=10).fit_transform(data, target.astype('int'))
    return select_data
def feature_choice_lda(data,target):
    #pdb.set_trace()
    estimator = LinearDiscriminantAnalysis(n_components=3)
    data_lda = estimator.fit(data,target).transform(data)
    #pdb.set_trace()
    return data_lda

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


def paint_error(pred_error,pred_choice_error):
    #pdb.set_trace()
    plt.figure(figsize=(20, 8), dpi=80)
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False
    x = np.arange(1, len(pred_error) + 1)
    for i in range(len(x)):
        #print(x[i],pred_value[i],true_value[i])
        plt.text(x[i],pred_error[i]+0.05,'%.2f' % pred_error[i],ha='center',va='bottom')
        plt.text(x[i], -(pred_choice_error[i] + 0.05), '%.2f' % pred_choice_error[i], ha='center', va='top')

    plt.plot(x, pred_error)
    plt.plot(x, pred_choice_error * (-1))
    plt.plot(x,np.zeros(len(pred_error)))

    plt.legend([u'特征选择前误差', u'特征选择后误差',u'标准'], fontsize='xx-large')
    plt.title(u'特征选择误差对比', fontsize='xx-large')
    plt.show()


def paint(true_value,pred_value):
    plt.figure(figsize=(20, 8), dpi=80)
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False
    x = np.arange(1,len(pred_value)+1)
    plt.subplot(121)
    for i in range(len(x)):
        #print(x[i],pred_value[i],true_value[i])
        plt.text(x[i],pred_value[i]+0.05,'%.2f' % pred_value[i],ha='center',va='bottom')
        #plt.text(x[i], -(true_value[i] + 0.05), '%.2f' % true_value[i], ha='center', va='top')
    plt.bar(x, pred_value)
    #plt.bar(x, true_value*(-1))

    plt.legend([u'预测房价'], fontsize='xx-large')
    plt.title(u'特征选择前房价预测结果',fontsize='xx-large')
    #plt.show()
    #///////////////////////////////////////////////////////////////
    plt.subplot(122)
    #plt.figure(figsize=(20, 8), dpi=80)
    for i in range(len(x)):
        #print(x[i],pred_value[i],true_value[i])
        plt.text(x[i],pred_value[i]+0.05,'%.2f' % pred_value[i],ha='center',va='bottom')
        plt.text(x[i], -(true_value[i] + 0.05), '%.2f' % true_value[i], ha='center', va='top')
    plt.bar(x, pred_value)
    plt.bar(x, true_value*(-1))

    plt.legend([ u'预测房价',u'实际房价'],fontsize='xx-large')
    plt.title(u'特征选择前房价对比结果',fontsize='xx-large')
    plt.show()

def paint_choice(true_value,pred_value):
    plt.figure(figsize=(20, 8), dpi=80)
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False
    x = np.arange(1,len(pred_value)+1)
    plt.subplot(121)
    for i in range(len(x)):
        #print(x[i],pred_value[i],true_value[i])
        plt.text(x[i],pred_value[i]+0.05,'%.2f' % pred_value[i],ha='center',va='bottom')
        #plt.text(x[i], -(true_value[i] + 0.05), '%.2f' % true_value[i], ha='center', va='top')
    plt.bar(x, pred_value)
    #plt.bar(x, true_value*(-1))

    plt.legend([u'预测房价'], fontsize='xx-large')
    plt.title(u'特征选择后房价预测结果',fontsize='xx-large')
    #plt.show()
    #///////////////////////////////////////////////////////////////
    plt.subplot(122)
    #plt.figure(figsize=(20, 8), dpi=80)
    for i in range(len(x)):
        #print(x[i],pred_value[i],true_value[i])
        plt.text(x[i],pred_value[i]+0.05,'%.2f' % pred_value[i],ha='center',va='bottom')
        plt.text(x[i], -(true_value[i] + 0.05), '%.2f' % true_value[i], ha='center', va='top')
    plt.bar(x, pred_value)
    plt.bar(x, true_value*(-1))

    plt.legend([ u'预测房价',u'实际房价'],fontsize='xx-large')
    plt.title(u'特征选择后房价生产总值对比结果',fontsize='xx-large')
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
    #pdb.set_trace()
    ws = xTx.I*(x_Mat.T*y_Mat.T)
    return ws

if __name__ == '__main__':
    data, target = loadData()
    x_train, y_train, x_test, y_test = train_test_split(data, target)

    ws = standard_Regress(x_train, y_train)
    # pdb.set_trace()
    print(ws)
    y_pred = x_test * ws
    paint(y_test, np.array(y_pred))

    x_train_choice = selectfeature(x_train,y_train)
    x_test_choice = selectfeature(x_test, y_test)
    ws = standard_Regress(x_train_choice, y_train)
    print(ws)
    y_pred_choice = x_test_choice * ws
    paint_choice(y_test, np.array(y_pred_choice))
    #pdb.set_trace()
    paint_error(np.squeeze(np.array(y_pred))-y_test,np.squeeze(np.array(y_pred_choice))-y_test)
