# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 13:48:31 2017

@author: bharg
"""
#class Classifier:
#    def __init__(self,X,y):
#        pass
#    def get_name(self):
#        raise NotImplementedError()
#    def fit(self,X,y):
#        raise NotImplementedError()
#    def predict(self,X):
#        raise NotImplementedError()
import numpy as np
from GradientDescent import GradientDescent
from cross_validation import Kcross
from sigmoid import sigmoid
class Classifier:
    def __init__(self,X,t):
        self.X=X
        self.t=t
        self.m=np.size(self.X,0)
        self.n=np.size(self.X,1)
        
    def get_name(self):
        return "Logistic regression"
    def HomeVal50(self):
        med=np.median(self.t)
        for i in range(self.m):
            if self.t[i]<med:
                self.t[i]=0
            else:
                self.t[i]=1
        return self.t
    def fit(self,X,t):
        self.W=np.random.randn(self.n)
        #self.J=(self.t).dot(np.log(self.y))+(1-self.t).dot(np.log(1-self.y))
        self.W,y=GradientDescent(self.X,self.t,self.W)
        return(self.W,y) 
    
    def predict(self,X,W):
        self.prob=sigmoid(np.matmul(self.X,W))
        self.t_pred=np.zeros((self.m))
        for i in range(self.m):
            if self.prob[i]>=0.5:
                self.t_pred[i]=1
            else:
                self.t_pred[i]=0
        return(self.t_pred)

from sklearn.datasets import load_boston
boston=load_boston().data
target=load_boston().target
indices=np.arange(np.size(boston,0))
np.random.shuffle(indices)
lst=[]
K=10
err_rate=np.zeros((K))
lst=np.array_split(indices,K)
for k in range(K):
    train_data,test_data,train_target,test_target=Kcross(boston,target,K)
    clf_train=Classifier(train_data,train_target)
    train_target=clf_train.HomeVal50()
    W,y=clf_train.fit(clf_train.X,train_target)
    clf_test=Classifier(test_data,test_target)
    test_target=clf_test.HomeVal50()
    t_pred=clf_test.predict(clf_test.X,W)
    count=0
    for i in range(np.size(t_pred)):
        if t_pred[i] != test_target[i]:
            count=count+1
    err_rate[k]=count/np.size(t_pred)
    
print("Mean error rate: {} and std deviation:{} ".format(np.mean(err_rate),np.std(err_rate)))   




            
        
        
        
        
    
        