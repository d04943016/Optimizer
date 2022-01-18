#!/usr/bin/env python3
# Copyright (c) 2020 Wei-Kai Lee. All rights reserved

# coding=utf-8
# -*- coding: utf8 -*-

# reference [1]: https://zhuanlan.zhihu.com/p/22252270
# reference [2]: https://zh.d2l.ai/chapter_optimization/adadelta.html

### python module
import sys, os
import numpy as np

OptPath = os.path.dirname(os.path.abspath(__file__))
srcPath = os.path.dirname(OptPath)
try:
    from . import myStatistics 
except:
    import myStatistics
if srcPath not in sys.path:
    sys.path.append(srcPath)
from Help.myXYClass import myXYClass

_DATA_LEN_ = 5
def LinearFittor(A, yi):
    [n, f] = A.shape # n : number of data/ f : number of free variables
    x = np.dot( np.linalg.pinv(A), yi )
    yh = np.dot(A, x)
    # loss
    chisquare = myStatistics.MeanSuqaredError(y_test=yi, y_predict=yh, N=n-f)
    return x, yh, chisquare
def diffFun(a,b):
    return np.max(np.abs(a-b))
def MsgPre(a_init, f=None, varStrList=None, varList=None, MSGOPENBOOL=False):
    countList, diffList, a_history, LossList = [], [], [], []
    # Information
    if not MSGOPENBOOL:
        return countList, diffList, a_history, LossList
    # Msg Box Title
    if varStrList!=None and varList!=None:
        for ii in np.arange(len(varStrList)):
            print( '{0:>20s} : {1:>10.5e}'.format(varStrList[ii], varList[ii]) )
    # Msg Box Content
    if f==None:
        if ( len(a_init)<=_DATA_LEN_ ):
            print('{0:>6s} {1:>15s}   {2}'.format('Iter.', 'da_diff', 'a'))
            print('='*30)
        else:
            print('{0:>6s} {1:>15s}'.format('Iter.', 'da_diff'))
            print('='*22)
    else:
        if ( len(a_init)<=_DATA_LEN_ ):
            print('{0:>6s} {1:>15s} {2:>15s}   {3}'.format('Iter.', 'Loss', 'da_diff', 'a'))
            print('='*45)
        else:
            print('{0:>6s} {1:>15s} {2:>15s}'.format('Iter.', 'Loss', 'da_diff'))
            print('='*40)
    return countList, diffList, a_history, LossList
def MsgRunning(Info, count, diff, a_new, fa_new=None, MSGCount=100, MSGOPENBOOL=False):
    countList, diffList, a_history, LossList = Info
    # Information
    countList += [count] 
    diffList  += [diff]
    a_history += [a_new]
    LossList = LossList if fa_new==None else LossList+[fa_new]
    if not MSGOPENBOOL:
        return countList, diffList, a_history, LossList
    # Msg Box Title
    if (count%MSGCount==0):
        if fa_new==None:
            if (len(a_new)<=_DATA_LEN_):
                print('{0:>6d} {1:>15.6e}   {2}'.format(count, diff, a_new))
            else:
                print('{0:>6d} {1:>15.6e}'.format(count, diff))
        else:
            if (len(a_new)<=_DATA_LEN_):
                print('{0:>6d} {1:>15.6e} {2:>15.6e}   {3}'.format(count, LossList[-1], diff, a_new))
            else:
                print('{0:>6d} {1:>15.6e} {2:>15.6e}'.format(count, LossList[-1], diff))
    return countList, diffList, a_history, LossList
def MsgPost(Info, count, diff, a_new, fa_new=None, MSGOPENBOOL=False):
    countList, diffList, a_history, LossList = Info
    # Information
    countList += [count] 
    diffList  += [diff]
    a_history += [a_new]
    LossList = LossList if fa_new==None else LossList+[fa_new]

    countList = np.array( countList, dtype=np.int32)
    diffList  = np.array(  diffList, dtype=np.float64)
    a_history = np.array( a_history, dtype=np.float64)
    LossList  = np.array(  LossList, dtype=np.float64)

    if not MSGOPENBOOL:
        return countList, diffList, a_history, LossList
    # Msg Box Title
    if fa_new==None:
        print('{0:>6d} {1:>15.6e}'.format(count, diff) )
    else:
        print('{0:>6d} {1:>15.6e} {2:>15.6e}'.format(count, LossList[-1], diff))
    return countList, diffList, a_history, LossList
def ParameterSavor(fpath, fname, varDict):
    if not os.path.isdir(fpath):
        os.makedirs(fpath)
    file = open(os.path.join(fpath, fname+'_Leaner_Parameter.txt'),'w')
    for key in varDict:
        file.write( '{0:>20s} : {1:>10.5e}'.format(key, varDict[key])+'\n' )
    file.close()
def ResultsSavor(a_new, count, countList, diffList, a_history, LossList, savefilepath, savefilename):
    # Optimizer Information
    appList = [ a_history[:,ii] for ii in range(a_history.shape[1]) ]
    apptagsList = ['x{0}'.format(ii+1) for ii in range(a_history.shape[1]) ]
    if len(LossList)==len(countList):
        yList = np.transpose( [LossList, diffList]+appList )
        tagsList = ['Loss', 'Max(abs(da))'] + apptagsList
        Appendix = myXYClass( countList, yList, xStr='count', tags=tagsList, xdtype=np.float64, ydtype=np.float64)
    else:
        yList = np.transpose( [diffList]+appList )
        tagsList = ['Max(abs(da))'] + apptagsList
        Appendix = myXYClass( countList, yList, xStr='count', tags=tagsList, xdtype=np.float64, ydtype=np.float64)
    Appendix.saveData(savefilepath, savefilename+'_Leaner_Path', fnameExtenstion='.txt')
    Appendix.plotData(savefilepath, savefilename+'_Leaner_Path', SeparateBool=True, figshowBool=True, closeBool=True)
    return Appendix
def StochasticGradientDescent(dfda_f_fun, a_init, learning_rate=0.01, tol=1e-5, iter_TOL=5e4, MSGOPENBOOL=True, difffun=diffFun, MSGCount=100):
    # initialize
    a_init = np.array(a_init, dtype=np.float64)
    learning_rate = np.abs(learning_rate)
    count = 0
    diff = np.inf
    a_new = a_init
    # message 
    varStrList = ('learning_rate', 'tol', 'iter_TOL' )
    varList    = [  learning_rate,   tol,  iter_TOL]

    dfda, f = dfda_f_fun(a_new)
    Info = MsgPre( a_new, f=f, varStrList=varStrList, varList=varList, MSGOPENBOOL=MSGOPENBOOL)
    # for loop
    while ( (count<iter_TOL) and diff>tol):
        # algorithm
        a_old = a_new.copy()
        dfda, f = dfda_f_fun(a_old)
        da = learning_rate * np.array(dfda, dtype=np.float64)
        a_new = a_old - da
        diff = difffun(a_new, a_old)
        # Information 
        Info = MsgRunning(Info, count, diff, a_old, fa_new=f, MSGCount=MSGCount, MSGOPENBOOL=MSGOPENBOOL)
        count += 1
    # Information
    dfda, f = dfda_f_fun(a_new)
    diff = 0
    countList, diffList, a_history, LossList = MsgPost(Info, count, diff, a_new, fa_new=f, MSGOPENBOOL=MSGOPENBOOL)
    return a_new, count, countList, diffList, a_history, LossList
def Momentum(dfda_f_fun, a_init, v_init=0.0, learning_rate=0.001, beta=0.9, tol=1e-5, iter_TOL=5e4, MSGOPENBOOL=True, difffun=diffFun, MSGCount=100):
    # initialize
    a_init = np.array(a_init, dtype=np.float64)
    learning_rate = np.abs(learning_rate)
    count = 0
    diff = np.inf
    a_new = a_init
    vt = v_init
    # message 
    varStrList = ( 'learning_rate', 'v_init',  'beta', 'tol', 'iter_TOL' )
    varList    = [  learning_rate,    v_init,    beta,  tol,   iter_TOL]

    dfda, f = dfda_f_fun(a_new)
    Info = MsgPre( a_new, f=f, varStrList=varStrList, varList=varList, MSGOPENBOOL=MSGOPENBOOL)
    # for loop
    while ( (count<iter_TOL) and diff>tol):
        # algorithm
        a_old = a_new.copy()
        dfda, f = dfda_f_fun(a_old)
        vt = beta*vt - learning_rate * np.array(dfda, dtype=np.float64)
        a_new = a_old + vt
        diff = difffun(a_new, a_old)
        # Information 
        Info = MsgRunning(Info, count, diff, a_old, fa_new=f, MSGCount=MSGCount, MSGOPENBOOL=MSGOPENBOOL)
        count += 1
    # Information
    dfda, f = dfda_f_fun(a_new)
    diff = 0
    countList, diffList, a_history, LossList = MsgPost(Info, count, diff, a_new, fa_new=f, MSGOPENBOOL=MSGOPENBOOL)
    return a_new, count, countList, diffList, a_history, LossList
def AdaGrad(dfda_f_fun, a_init, epsilon=1e-8, learning_rate=0.5, tol=1e-5, iter_TOL=5e4, MSGOPENBOOL=True, difffun=diffFun, MSGCount=100):
    # initialize
    a_init = np.array(a_init, dtype=np.float64)
    learning_rate = np.abs(learning_rate)
    count = 0
    diff = np.inf
    a_new = a_init
    sigma = np.zeros( a_init.size, dtype=np.float64 )
    # message 
    varStrList = ('learning_rate', 'epsilon', 'tol', 'iter_TOL' )
    varList    = [  learning_rate,  epsilon,   tol,  iter_TOL]

    dfda, f = dfda_f_fun(a_new)
    Info = MsgPre( a_new, f=f, varStrList=varStrList, varList=varList, MSGOPENBOOL=MSGOPENBOOL)
    # for loop
    while ( (count<iter_TOL) and diff>tol):
        # algorithm
        a_old = a_new.copy()
        dfda, f = dfda_f_fun(a_old)
        grad = np.array(dfda, dtype=np.float64)
        sigma = sigma+grad**2
        da = learning_rate * grad/np.sqrt( sigma+epsilon )
        a_new = a_old - da
        diff = difffun(a_new, a_old)
        # Information 
        Info = MsgRunning(Info, count, diff, a_old, fa_new=f, MSGCount=MSGCount, MSGOPENBOOL=MSGOPENBOOL)
        count += 1
    # Information
    dfda, f = dfda_f_fun(a_new)
    diff = 0
    countList, diffList, a_history, LossList = MsgPost(Info, count, diff, a_new, fa_new=f, MSGOPENBOOL=MSGOPENBOOL)
    return a_new, count, countList, diffList, a_history, LossList
def AdaDelta(dfda_f_fun, a_init, rho=0.9, epsilon=1e-8, tol=1e-5, iter_TOL=5e4, MSGOPENBOOL=True, difffun=diffFun, MSGCount=100):    # initialize
    a_init = np.array(a_init, dtype=np.float64)
    count = 0
    diff = np.inf
    a_new = a_init
    sigma = np.zeros( a_init.size, dtype=np.float64 )
    deltaX = np.zeros( a_init.size, dtype=np.float64 )
    # message 
    varStrList = ('rho', 'epsilon', 'tol', 'iter_TOL' )
    varList    = [  rho,   epsilon,   tol,  iter_TOL]

    dfda, f = dfda_f_fun(a_new)
    Info = MsgPre( a_new, f=f, varStrList=varStrList, varList=varList, MSGOPENBOOL=MSGOPENBOOL)
    # for loop
    while ( (count<iter_TOL) and diff>tol):
        # algorithm
        a_old = a_new.copy()
        dfda, f = dfda_f_fun(a_old)

        grad = np.array(dfda, dtype=np.float64)
        sigma = rho*sigma+(1-rho)*grad**2
        da = - np.sqrt( deltaX+epsilon ) * grad/np.sqrt( sigma+epsilon )
        a_new = a_old + da
        deltaX = rho*deltaX+(1-rho)*da**2
        diff = difffun(a_new, a_old)
        # Information 
        Info = MsgRunning(Info, count, diff, a_old, fa_new=f, MSGCount=MSGCount, MSGOPENBOOL=MSGOPENBOOL)
        count += 1
    # Information
    dfda, f = dfda_f_fun(a_new)
    diff = 0
    countList, diffList, a_history, LossList = MsgPost(Info, count, diff, a_new, fa_new=f, MSGOPENBOOL=MSGOPENBOOL)
    return a_new, count, countList, diffList, a_history, LossList
def RMSprop(dfda_f_fun, a_init, rho=0.9, epsilon=1e-8, learning_rate=0.01, tol=1e-5, iter_TOL=5e4, MSGOPENBOOL=True, difffun=diffFun, MSGCount=100):
    # initialize
    a_init = np.array(a_init, dtype=np.float64)
    learning_rate = np.abs(learning_rate)
    count = 0
    diff = np.inf
    a_new = a_init
    sigma = np.zeros( a_init.size, dtype=np.float64 )
    # message 
    varStrList = ('learning_rate', 'rho', 'epsilon', 'tol', 'iter_TOL' )
    varList    = [  learning_rate,   rho,   epsilon,   tol,  iter_TOL]

    dfda, f = dfda_f_fun(a_new)
    Info = MsgPre( a_new, f=f, varStrList=varStrList, varList=varList, MSGOPENBOOL=MSGOPENBOOL)
    # for loop
    while ( (count<iter_TOL) and diff>tol):
        a_old = a_new.copy()
        dfda, f = dfda_f_fun(a_old)
        grad = np.array(dfda, dtype=np.float64)
        sigma = rho*sigma+(1-rho)* grad**2
        da = learning_rate * grad/np.sqrt( sigma+epsilon )
        a_new = a_old - da
        diff = difffun(a_new, a_old)
        # Information 
        Info = MsgRunning(Info, count, diff, a_old, fa_new=f, MSGCount=MSGCount, MSGOPENBOOL=MSGOPENBOOL)
        count += 1
    # Information
    dfda, f = dfda_f_fun(a_new)
    diff = 0
    countList, diffList, a_history, LossList = MsgPost(Info, count, diff, a_new, fa_new=f, MSGOPENBOOL=MSGOPENBOOL)
    return a_new, count, countList, diffList, a_history, LossList
def Adam(dfda_f_fun, a_init, beta1=0.9, beta2=0.999, epsilon=1e-8, learning_rate=0.1, tol=1e-5, iter_TOL=5e4, MSGOPENBOOL=True, difffun=diffFun, MSGCount=100):
    # initialize
    a_init = np.array(a_init, dtype=np.float64)
    learning_rate = np.abs(learning_rate)
    count = 0
    diff = np.inf
    a_new = a_init
    vt = np.zeros( a_init.size, dtype=np.float64 )
    sigma = np.zeros( a_init.size, dtype=np.float64 )
    # message 
    varStrList = ('learning_rate', 'beta1', 'beta2', 'epsilon', 'tol', 'iter_TOL' )
    varList    = [  learning_rate,   beta1,   beta2,   epsilon,   tol,  iter_TOL]


    dfda, f = dfda_f_fun(a_new)
    Info = MsgPre( a_new, f=f, varStrList=varStrList, varList=varList, MSGOPENBOOL=MSGOPENBOOL)
    # for loop
    while ( (count<iter_TOL) and diff>tol):
        # algorithm
        count += 1
        a_old = a_new.copy()
        dfda, f = dfda_f_fun(a_old)
        grad = np.array(dfda, dtype=np.float64)
        vt = beta1*vt + (1-beta1)*grad
        sigma = beta2*sigma+(1-beta2)* grad**2
        v_bias_corr = vt/(1-beta1**count)
        sigma_bias_corr = sigma/(1-beta2**count)
        da = learning_rate * v_bias_corr/np.sqrt( sigma_bias_corr+epsilon )
        a_new = a_old - da
        diff = difffun(a_new, a_old)
        # Information 
        Info = MsgRunning(Info, count-1, diff, a_old, fa_new=f, MSGCount=MSGCount, MSGOPENBOOL=MSGOPENBOOL)
    # Information
    dfda, f = dfda_f_fun(a_new)
    diff = 0
    countList, diffList, a_history, LossList = MsgPost(Info, count, diff, a_new, fa_new=f, MSGOPENBOOL=MSGOPENBOOL)
    return a_new, count, countList, diffList, a_history, LossList
if __name__ == '__main__':
    import doctest
    doctest.testmod()

    def dfda_f_fun(x):
        dfda = np.array( [(x[0]-2)**1, (x[1]-4)**3], dtype=np.float64 )
        f = (x[0]-2)**2/2 + (x[1]-2)**4/4
        return dfda, f

    MSGOPENBOOL = False
    tol = 1e-8
    a_init = [0.0,0.0]
    
    print( 'StochasticGradientDescent' )
    a_new, count, countList, diffList, a_history, LossList = StochasticGradientDescent(dfda_f_fun, a_init=[0,0], tol=tol, MSGOPENBOOL=MSGOPENBOOL)
    print( a_new, count, '\n' )
   
    print( 'Momentum' )
    a_new, count, countList, diffList, a_history, LossList = Momentum(dfda_f_fun, a_init=a_init, tol=tol, MSGOPENBOOL=MSGOPENBOOL)
    print( a_new, count, '\n' )
    
    print( 'AdaGrad' )
    a_new, count, countList, diffList, a_history, LossList = AdaGrad(dfda_f_fun, a_init=a_init, tol=tol, MSGOPENBOOL=MSGOPENBOOL)
    print( a_new, count, '\n' )
    
    print( 'AdaDelta' )
    a_new, count, countList, diffList, a_history, LossList = AdaDelta(dfda_f_fun, a_init=a_init, tol=tol, MSGOPENBOOL=MSGOPENBOOL)
    print( a_new, count, '\n' )
    
    print( 'RMSprop' )
    a_new, count, countList, diffList, a_history, LossList = RMSprop(dfda_f_fun, a_init=a_init, tol=tol, MSGOPENBOOL=MSGOPENBOOL)
    print( a_new, count, '\n' )
   
    print( 'Adam' )
    a_new, count, countList, diffList, a_history, LossList = Adam(dfda_f_fun, a_init=a_init, tol=tol, MSGOPENBOOL=MSGOPENBOOL)
    print( a_new, count, '\n' )




