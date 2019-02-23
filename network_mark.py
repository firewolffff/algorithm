# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 15:27:24 2018

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
    x 样本集合 n * m n 个特征 m 个样本数量
    b 偏移量
    W 权重
    A 净输出
    '''
    def __init__(self,layers,active_function=[logistic],active_function_der=[logistic_derivative],learn_rate=0.9):
        """
        初始化神经网络
        layer中存放每层的神经元数量，layer的长度即为网络的层数
        active_function 为每一层指定一个激活函数，若长度为1则表示所有层使用同一个激活函数
        active_function_der 激活函数的导数
        learn_rate 学习速率 
        """
        self.weights = [np.random.randn(x,y) for x,y in zip(layers[1:],layers[:-1])];
        self.biases = [np.random.randn(x,1) for x in layers[1:]];
        self.size = len(layers);
        self.rate = learn_rate;
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
        
    def fit(self,TrainData,epochs=1000,mini_batch_size=32):
        """
        运用后向传播算法学习神经网络模型
        TrainData 是（X,Y）值对
        X 输入特征矩阵 m*n 维 n 个特征，m个样本
        Y 输入实际值 t*m 维 t个类别标签，m个样本
        epochs 迭代次数
        mini_batch_size mini_batch 一次的大小，不使用则mini_batch_size = 1
        """
        n = len(TrainData);
        for i in range(epochs):
            random.shuffle(TrainData)
            mini_batches = [
                TrainData[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)];
            for mini_batch in mini_batches:
                self.BP(mini_batch, self.rate);
        
        
        
        
    def predict(self, x):
        """前向传播"""
        i = 0;
        for b, w in zip(self.biases, self.weights):
            x = self.sigmoids[i](np.dot(w, x)+b);
            i = i + 1;
        return x
    
    def BP(self,mini_batch,rate):
        """
        BP 神经网络算法
        """
        size = len(mini_batch);

        nabla_b = [np.zeros(b.shape) for b in self.biases]; #存放每次训练b的变化量
        nabla_w = [np.zeros(w.shape) for w in self.weights]; #存放每次训练w的变化量
        #一个一个的进行训练  
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y);
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]; #累加每次训练b的变化量
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]; #累加每次训练w的变化量
        self.weights = [w-(rate/size)*nw
                        for w, nw in zip(self.weights, nabla_w)];
        self.biases = [b-(rate/size)*nb
                       for b, nb in zip(self.biases, nabla_b)];
            
    def backprop(self, x, y):
        """
        x 是一维 的行向量
        y 是一维行向量
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases];
        nabla_w = [np.zeros(w.shape) for w in self.weights];
        # feedforward
        activation = np.atleast_2d(x).reshape((len(x),1)); #转换为列向量
        activations = [activation]; # 存放每层a
        zs = []; # 存放每z值
        i = 0;
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b;
            zs.append(z);
            activation = self.sigmoids[i](z);
            activations.append(activation);
            i = i + 1;
        # backward pass
        y = np.atleast_2d(y).reshape((len(y),1)); #将y转化为列向量
        #delta cost对z的偏导数
        delta = self.cost_der(activations[-1], y) * \
            self.sigmoids_der[-1](zs[-1]);
        nabla_b[-1] = delta;
        nabla_w[-1] = np.dot(delta, np.transpose(activations[-2]));
        
        for l in range(2, self.size):
            z = zs[-l]; #当前层的z
            sp = self.sigmoids_der[-l](z); #对z的偏导数值
            delta = np.multiply(np.dot(np.transpose(self.weights[-l+1]), delta), sp); #求出当前层的误差
            nabla_b[-l] = delta;
            nabla_w[-l] = np.dot(delta, np.transpose(activations[-l-1]));
        return (nabla_b, nabla_w)
    
    """
    损失函数
    cost_der 差的平方损失函数对a 的导数
    cost_cross_entropy_der 交叉熵损失函数对a的导数
    """
    def cost_der(self,a,y):
        return a - y;
    
    def cost_cross_entropy_der(self,a,y):
        return (a-y)/(a * (1-a));
        
        
