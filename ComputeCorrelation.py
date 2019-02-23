# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 19:36:48 2018

@author: markli
"""

import numpy as np;
import math;

'''
计算矩阵A的相关系数矩阵
'''
def Correlation(A):
    #得到A的形状 m 是行数 n 是列数
    m,n = A.shape;
    #存放每一列的均值
    means = [];
    #存放每一列的方差
    var = [];
    for i in range(n):
        me = np.mean(A[:,i]);
        means.append(me);
        temp = A[:,i] - me;
        #计算方差，除以m-1 与np.corrcoef有误差，除以m则基本没有误差
        v = np.sum([p**2 for p in temp]) / m  ;
        var.append(v);
        
    #存放相关系数
    r = np.ones((n,n));
    #离差矩阵
    deviation = A - np.atleast_2d(means);
    
    for i in range(n):
        for j in range(n):
            cov = np.dot(np.atleast_2d(deviation[:,i]),np.atleast_2d(deviation[:,j]).T)/ (m);
            va = math.sqrt(var[i] * var[j]);
            r[i,j] = cov / va;
            if(i==j):
                r[i,j]=1;
            
    return r;
#coeffs = np.ployfit(X,Y,degree) #得到一组回归方程的系数 X为一维，Y一般为一维，最多二维 degree指定自变量的次数
#np.ploy1d(coeffs) 产生一个多项式 多项式次数由高到低
A = np.random.random((10,3));
r1 = Correlation(A);
r2 = np.corrcoef(A,rowvar=False);
print(r1);
print('numpy 计算值：');
print(r2);
            
    