# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 19:01:40 2018

@author: markli

采用信息增益作为特征选择原则构建决策树
"""
import numpy as np;
import pandas as pd;

class DecisionTree(object):
    def __init__(self,features):
        """
        features 样本具有的特征的取值范围，例如具有A1，A2两个特征，features=[A1,A2]
        """
        self.features = features;
    
    def fit(self,TrainData):
        """
        TrainData 训练样本，数据格式为二维数组m*n
        m 个样本，n-1 个特征，最后一列是类别标签
        """
        tree = self.GetDicTree(TrainData,self.features);
        return tree;
        
        
    def SetEntropy(self,Data):
        """
        获取数据集的经验熵
        Data 数据集的最后一列，类别
        """
        N = len(Data[:,-1]); #获取数据集的大小
        Numoflabel = pd.value_counts(Data[:,-1]); #得数据集中到每一类别的数量
        classlabel = list(set(Data[:,-1]));
        entropy = 0;# 数据集的信息熵
        for c in classlabel:
            try:
                Ck = Numoflabel[c]; #得到类别为c的样例的数量
                entropy = entropy - Ck/N * np.log2(Ck/N);
            except KeyError:
                Ck = 0;
                entropy = entropy - Ck;
            
        return entropy;
    
    def ConditionEntropy(self,Data,index):
        """
        获取某一特征的条件经验熵
        Data 数据集与TrainData格式一致
        feature 特征的取值范围 例如 A1=[1,2,3]
        feature_index 该特征在数据集中属于第几个特征,从0开始
        """
        ConEntropy = 1;
        feature_value = list(set(Data[:,index]));
        N = len(Data[:,0]);
        for a in feature_value:
            d = Data[np.where(Data[:,index]==a)];
            d_n = len(d);
            if(d_n == 0):
                return 0;
            #计算特征取a值时的数据集的经验熵
            d_entropy = self.SetEntropy(d);
            ConEntropy = ConEntropy * (d_n / N) * d_entropy;
        
        return -ConEntropy;
    
    def SelectBestFeature(self,Data):
        """
        选出数据集中最大信息增益的特征及最优特征
        Data 数据集与TrainData格式一致
        """
        AddEntropy = [];
        entropy = self.SetEntropy(Data); #求得数据集的经验熵
        feature_num = len(Data[0])-1; #获得数据集中特征数量
        for i in range(feature_num):
            ConEntropy = self.ConditionEntropy(Data,i); #求得每个特征的条件熵
            adden = entropy - ConEntropy;
            AddEntropy.append(adden);
        
        index = np.argmax(AddEntropy);
        return index;
      
    
    def VoteClass(self,classlist):
        """
        当特征被选完，但还是无法准确判断哪一类时，采用投票的方式确定其类
        """
        classlabel = list(set(classlist));
        dic = {};
        for c in classlabel:
            if(c not in dic.keys()):
                dic[c] = 0;
            else:
                dic[c] += 1;
        return max(dic);
    
    def GetDicTree(self,TrainData,features):
        """
        构造字典树
        TrainData 训练数据集
        """
        classlabel = list(set(TrainData[:,-1])); #获得数据集的类别标签
        #classlabel = [row[-1] for row in TrainData];
        
        if(len(classlabel) == 1):
            return classlabel[0];
        
        if(len(TrainData[0]) == 1):
            return self.VoteClass(TrainData[:,-1]);
        
        bestfeature_index = self.SelectBestFeature(TrainData);
        bestfeature = features[bestfeature_index]; #选出最优的特征
        dictree = {bestfeature:{}}; #以最优特征为节点构建子树
        del(features[bestfeature_index]) #删除已选过的特征
        
        #根据最优特征的取值拆分数据集，递归上述选最优特征过程
        feature_attr = list(set(TrainData[:,bestfeature_index]));
        for value in feature_attr:
            sub_features = features[:];
            subdata = self.SplitData(TrainData,bestfeature_index,value);
            dictree[bestfeature][value] = self.GetDicTree(subdata,sub_features);
            
        return dictree;
    
    def SplitData(self,Data,feature_index,feature_value):
        subdata = Data[np.where(Data[:,feature_index] == feature_value)];
        n = len(Data[0]);
        subdata = [[row[i] for i in range(n) if i != feature_index] for row in subdata];
        return np.array(subdata);
            
        
           
            
            
            

