# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 23:37:16 2017

@author: bharg
"""
import numpy as np
def Kcross(X,y,indexlist):
    test_data=X[indexlist]
    test_target=y[indexlist]
    train_data=np.delete(X,indexlist,axis=0)
    train_target=np.delete(y,indexlist,axis=0)
    return(train_data,test_data,train_target,test_target)


        
        
    
    
    
    
    
    
    
    
#    m=np.size(X,0)
#    n=np.size(X,1)
#    f=int(train_percent*m)
#    indices=sample(range(m),f)
#    train_data=X[indices]
#    test_data=np.delete(X,indices,0)
#    train_target=t[indices]
#    test_target=np.delete(t,indices)
#    return(train_data,test_data,train_target,test_target)
