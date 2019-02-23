# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 19:18:56 2018

@author: markli
"""
import numpy as np;
'''
kmeans 算法实现
算法原理
1、随机选择k个点作为聚类中心点，进行聚类
2、求出聚类后的各类的 中心点
3、由中心点作为新的聚类中心点，再次进行聚类
4、比较前后两次的聚类中心点是否发生变化，若没有变化则停止，否则重复2,3,4
'''

def Kmeans(X,k,maxiter):
    '''
    使用Kmeans均值聚类对数据集Data进行聚类
    X 数据集
    k 聚类中心个数
    maxiter 最大迭代次数
    '''
    m,n = X.shape;
    #向数据集中添加一列，用来存放类别号
    Dataset = np.zeros((m,n+1));
    Dataset[:,:-1] = X;
    
     #随机选取k 个聚类中心
    randomCenterIndex = np.random.randint(m,size=k);
    center = Dataset[randomCenterIndex];
    center[:,-1] = range(1,k+1);
    
    #初始聚类
    oldCenter = np.copy(center);
    DataClass(Dataset,center);
    center = getCenter(Dataset,k);
    
    itertor = 1;
    while not isStop(oldCenter,center,itertor,maxiter):
        oldCenter = np.copy(center);
        DataClass(Dataset,center);
        center = getCenter(Dataset,k);
        itertor = itertor + 1;
    print("数据集聚类结果",Dataset);
    print("聚类中心点",center);
        

def DataClass(Dataset,center):
    '''
    对数据集进行聚类或者类标签更新
    Dataset 数据集
    center 聚类中心点 最后一列为聚类中心点的分类号
    '''
    n = Dataset.shape[0];
    k = center.shape[0];
    for i in range(n):
        lable = center[0,-1];
        mindistance = np.linalg.norm(Dataset[i,:-1]-center[0,:-1],ord=2);
        for j in range(1,k):
            distance = np.linalg.norm(Dataset[i,:-1]-center[j,:-1],ord=2);
            if(distance < mindistance):
                mindistance = distance;
                lable = center[j,-1];
        Dataset[i,-1] = lable;

def getCenter(Dataset,k):
    '''
    获得数据集的k个聚类中心,数据集的最后一列是当前的分类号
    Dataset 数据集
    k 聚类中心点个数
    '''
    center = np.ones((k,Dataset.shape[1]));
    for i in range(1,k+1):
        DataSubset = Dataset[Dataset[:,-1] == i,:];
        center[i-1] = np.mean(DataSubset,axis=0);
    return center;

def isStop(oldCenter,newCenter,itertor,maxiter):
    '''
     判断是否停止
     oldCenter 前一次聚类的聚类中心
     newCenter 新产生的聚类中心
     itertor 当前迭代次数
     maxitor 最大迭代次数
    '''

    if(itertor >= maxiter):
        return True;
    
    return np.array_equal(oldCenter,newCenter);


X = np.array([[1,1],[2,1],[4,3],[5,4]]);
print(X.shape);
Kmeans(X,2,10);