# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 13:13:44 2017

@author: markli
"""
import numpy as np;
#两点之间的距离采用欧式几何距离
'''
采用欧式距离进行K最小临近分类
x 未知分类点 m*1 向量
y n个测试样本点 m*n 维向量
'''
def ComputeDistance(x,y):
    m = len(x); #获取维度数量
    #print(m);
    tempeye = -np.eye(m);
    tempone = np.ones((1,m));
    C = np.vstack((tempone,tempeye));#中间过渡矩阵 m+1 * m 按列合并，列数不变扩张行
    translateMatrix = np.hstack((x,y)); #按行合并，行数不变，扩张列
    tempresult = np.dot(translateMatrix,C);
    result = np.multiply(tempresult,tempresult);
    #result = [d**2 for d in np.array(tempresult)];
    result = np.sum(result,axis=0)
    distance = [pow(d,1/m) for d in np.array(result)];
    return distance;

'''
k 选取点的个数
distance 带预测点与每个样本点的距离
labels 每个样本点的类别标记
return 返回距离最近的k的样本点的类别标记
'''
def KNN(k,distance,labels):
    dis_label = [];
    for i in range(len(labels)):
        tup = (distance[i],labels[i]);
        dis_label.append(tup);
    dis_label = sorted(dis_label,lambda x:x[0]);
    Kmin = [];
    for i in range(k-1):
        label = dis_label[i][1];
        if label not in Kmin:
            Kmin.append(label);
    return Kmin;
    
    
    
