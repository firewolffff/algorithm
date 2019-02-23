# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 13:47:54 2018

@author: markli
"""
import numpy as np;
import random;

def tanh(x):  
    return np.tanh(x);

def tanh_derivative(x):  
    return 1.0 - np.tanh(x)*np.tanh(x);

def logistic(x):  
    return 1/(1 + np.exp(-x));

def logistic_derivative(x):  
    return logistic(x)*(1-logistic(x));

def ReLU(x,a=1):
    return max(0,a * x);

def ReLU_derivative(x,a=1):
    return 0 if x < 0 else a;


class NeuralNetwork:
    '''
    Z = W * x + b
    A = sigmod(Z)
    Z 净输入
    x 样本集合 m * n n 个特征 m 个样本数量
    b 偏移量
    W 权重
    A 净输出
    '''
    def __init__(self,layers,active_function=[logistic],active_function_der=[logistic_derivative],learn_rate=0.9):
        self.weights = [2*np.random.randn(x,y)-1 for x,y in zip(layers[1:],layers[:-1])]; #weight 取值范围（-1,1）
        self.B = [2*np.random.randn(x,1)-1 for x in layers[1:]]; #b 取值范围（-1,1）
        self.learnRate = learn_rate;
        self.size = len(layers);
        self.sigmoids = [];
        self.sigmoids_der = [];
        for i in range(len(layers)-1):
            if(len(active_function) == self.size-1):
                self.sigmoids = active_function;
            else:
                self.sigmoids.append(active_function[0]);
            if(len(active_function_der)== self.size-1):
                self.sigmoids_der = active_function_der;
            else:
                self.sigmoids_der.append(active_function_der[0]);
    
   
    '''后向传播算法'''
    def BackPropgation(self,X,Y): 
        """
        X size*n 维，size大小为Mini_Batch_size 值大小,n 个特征
        Y size*l 维，size大小为Mini_Batch_sieze 值大小，l 个类标签
        一次计算size个样本带来的w,b的变化量
        """
        deltb = [np.zeros(b.shape) for b in self.B];
        deltw = [np.zeros(w.shape) for w in self.weights];
        
        active = np.transpose(X);
        actives = [active];
        zs = [];
        i=0;
        #前向传播
        for w,b in zip(self.weights,self.B):
            z = np.dot(w,active) + b;
            zs.append(z);
            active = self.sigmoids[i](z);
            actives.append(active);
            i = i+1;
        
        Y = np.transpose(Y); #转置
        cost = self.cost(actives[-1], Y) #成本函数 计算对a的一阶导数
        z = zs[-1];
        delta = np.multiply(cost,self.sigmoids_der[-1](z)); #计算输出层(最后一层)的变化量
        deltb[-1] = np.sum(delta,axis=1,keepdims=True); #计算输出层(最后一层)b的size次累计变化量 l*1 维
        deltw[-1] = np.dot(delta, np.transpose(actives[-2]));#计算输出层(最后一层)w的size次累计变化量 x*l 维
        for i in range(2,self.size):
            z = zs[-i]; #当前层的z值
            sp = self.sigmoids_der[-i](z); #对z的偏导数值
            delta = np.multiply(np.dot(np.transpose(self.weights[-i+1]), delta), sp); #求出当前层的误差
            #deltb = delta;
            deltb[-i] = np.sum(delta,axis=1,keepdims=True); #当前层b的size次累计变化量 l*1 维
            deltw[-i] = np.dot(delta, np.transpose(actives[-i-1])); # 当前层w的size次累计变化量 x*l
            
        return deltw,deltb;
            
    def fit(self,X,Y,mini_batch_size,epochs=1000):
        
        N = len(Y);
        for i in range(epochs):
            randomlist = np.random.randint(0,N-mini_batch_size,int(N/mini_batch_size));
            batch_X = [X[k:k+mini_batch_size] for k in randomlist];
            batch_Y = [Y[k:k+mini_batch_size] for k in randomlist];
            for m in range(len(batch_Y)):
                deltw,deltb = self.BackPropgation(batch_X[m],batch_Y[m]);
                self.weights = [w - (self.learnRate / mini_batch_size) * dw for w,dw in zip(self.weights,deltw)];
                self.B = [b - (self.learnRate / mini_batch_size) * db for b,db in zip(self.B,deltb)];
#        path = sys.path[0];
#        with open(path,'w',encoding='utf8') as f:
#            for j in range(len(self.weights)-1):
#                f.write(self.weights[j+1]);
#                f.write(self.activeFunction[j+1]);
#                f.write(self.activeFunctionDer[j+1]);
#        f.close();
        
            
                
    
    def predict(self,x):
        """前向传播"""
        i = 0;
        for b, w in zip(self.B, self.weights):
            x = self.sigmoids[i](np.dot(w, x)+b);
            i = i + 1;
        return x
    
    def cost(self,a,y):
        """
        损失函数对z的偏导数的除输出层对z的导数的因子部分
        完整表达式 为 （a - y）* sigmod_derivative(z)
        由于此处不知道输出层的激活函数故不写出来，在具体调用位置加上
        """
        return a-y;
    
        
        
