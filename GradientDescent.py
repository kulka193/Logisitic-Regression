# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 22:23:53 2017

@author: bharg
"""
import numpy as np
from sigmoid import sigmoid
def GradientDescent(X,t,W):
    m=np.size(X,0)
    lr=0.001
    for _iter in range(1000):
        y=sigmoid(np.matmul(X,W))
        W=W-(lr/m)*np.matmul(X.T,(y-t))
    return W,y
    