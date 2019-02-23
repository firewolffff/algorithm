# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 20:11:07 2018

@author: markli
"""
import numpy as np;
import pandas as pd;

"""
贝叶斯分类
训练数据格式
X = [x1,x2,...xm]; n*m
xi = [xi1,xi2,...xin].T
Y = [y1,y2,...ym];1*m
"""

class Bayes:
    def __init__(self,lamda,region):
        """
        lamda 贝叶斯修正参数
        region 特征属性取值域，类标签取值域
        例如两个特征取值范围分别为，A1=[1,2,3],A2=['S','M','L']
        类标签取值：C=[1,-1]，region=[A1,A2,C]
        """
        self.lamda = lamda;
        #存放类标签域
        self.Y = region[-1];
        #存放特征取值域
        self.X = region[:-1];

        #存放先验概率 P(Y = Ck)
        self.PrioPro = np.zeros((1,len(region[-1])));

        #存放条件概率 P(Xj = ajl | Y = Ck)
        self.ConditionalPro = [];
        for i in range(len(region)-1):
            cp = np.zeros((len(region[-1]),len(region[i])));
            self.ConditionalPro.append(cp);

    def fit(self,TrainData):
        """
        计算先验概率和条件概率，建立模型
        TrainData 为二维数组
        TrainData 列的顺序与region中特征属性顺序一致
        TrainData 最后一列为类别
        """
        N = len(TrainData);
        K = len(self.Y);
        TrainData = TrainData.astype(np.str);

        NumofCk = pd.value_counts(TrainData[:,-1], sort=True); #Series 类型
        CountOfCk = [NumofCk[ck] for ck in self.Y]; #list类型
        self.PrioPro = [(ck+self.lamda) / (N + K * self.lamda) for ck in CountOfCk];

        j=0;
        for ck in self.Y:
            #选出类别为Ck的数据
            DataofCk = TrainData[np.where(TrainData[:,-1]==ck)];
            n = len(DataofCk);
            #选出第i个特征的数据
            for i in range(len(self.X)):
                DataofCkandXi = DataofCk[:,i];
                Numofaj = pd.value_counts(DataofCkandXi,sort=True); #为第i个特征的每个特征值计数
                Countofaj = [Numofaj[aj] for aj in self.X[i]];
                S = len(self.X[i]);
                self.ConditionalPro[i][j] = [(aj+self.lamda) / (n+S * self.lamda) for aj in Countofaj];
            j = j+1;

    def predict(self,TestData):
        """
        预测实例，为其分类
        测试数据没有类别列，其余数据格式与训练数据格式一致
        """
        predictY = [];
        for i in range(len(TestData)):
            x = TestData[i];
            y = self.GetLable(x);
            predictY.append(y);
        return predictY;

    def GetLable(self,x):
        """
        输入一个测试实例x ，输出使其后验概率最大的类别y
        """
        pro = [];
        n = len(x);
        for j in range(len(self.Y)):
            p = 1;
            for i in range(n):
                feature = self.ConditionalPro[i];
                fi = self.X[i] #获得第i个特征的值域
                index = fi.index(x[i]);
                p = p * feature[j][index];

            p = p * self.PrioPro[j];
            pro.append(p);

        y = self.Y[np.argmax(pro)];

        return y;
