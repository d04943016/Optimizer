#!/usr/bin/env python3
# Copyright (c) 2019 Wei-Kai Lee. All rights reserved

# coding=utf-8
# -*- coding: utf8 -*-
### python module
import sys, os
import numpy as np

### my module
HelpPath = os.path.dirname(os.path.abspath(__file__))
srcPath = os.path.dirname(HelpPath)

def error(y_test, y_predict):
    return y_predict-y_test
def MeanSuqaredError(y_test, y_predict, N=1):
    return np.sum( error(y_test, y_predict)**2 )/2/N
def dMeanSuqaredError_dy(y_test, y_predict, N=1):
    return (y_test-y_predict) *(-1/N)
def R(yi_ori, yh_ori):
    # analysis of variance
    yi = yi_ori.reshape( yi_ori.size )
    yh = yh_ori.reshape( yh_ori.size )
    yibar, yhbar = np.mean(yi), np.mean(yh)
    SSReg = np.sum( (yh-yhbar)**2 ) 
    SSTot = np.sum( (yi-yibar)**2 )
    if SSReg != 0 and SSTot != 0:
        R = np.sum( (yi-yibar) * (yh-yhbar) )/(SSReg*SSTot)**0.5
    else :
        R = 0
    return R