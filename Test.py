#!/usr/bin/env python3
# Copyright (c) 2022 Wei-Kai Lee. All rights reserved

# coding=utf-8
# -*- coding: utf8 -*

import numpy as np
from Optimizer.GradientDescentOptimizer import GradientDescent, Momentum, AdaGrad, AdaDelta, RMSprop, Adam
from Optimizer.ScipyOptimizer import Scipy_LBFGS_B

""" timer """
def timer(optimier, x_init, N=1):
    import time
    t1 = time.time()
    for ii in range(N):
        optimier.optimize(x_init)
    t2 = time.time()
    print( 'average time = ', (t2-t1)/N )
""" test function """
def test(optimizer, x_init, N=1, savefilepath='./Example', save_history=True):
    print('Now testing {0}'.format(optimizer.name))
    print('='*50)
    optimizer.save_parameter(savefilepath=savefilepath)

    print('optimizer print parameter (before optimization): ')
    optimizer.print_parameter()
    print('')

    print('optimizer print state (before optimization): ')
    optimizer.print_state()
    print('')

    print('optimization')
    timer(optimizer, x_init, N)
    print('')

    print('optimizer print parameter (after optimization): ')
    optimizer.print_parameter()
    print('')

    print('optimizer print state (after optimization): ')
    optimizer.print_state()
    print('')

    if save_history:
        optimizer.save_history(savefilepath=savefilepath).plot_history(savefilepath=savefilepath)
if __name__ == '__main__':
    """ example """
    min1, min2 = 1, 0.5
    y = lambda x: (x[0]-min1)**2 + (x[1]-min2)**2
    grad = lambda x: np.array( [2*(x[0]-min1), 2*(x[1]-min2)] )
    y_grad = lambda x: ( y(x), grad(x) )
    x_init = np.array( [1.5, 1.0] )
    
    #y = lambda x: (x-1)**2
    #grad = lambda x: 2*(x-1)
    #y_grad = lambda x: ( y(x), grad(x) )
    #x_init = np.array( 1.5 )
        
    y_grad = y_grad
    grad_flag = True

    """ example gradient descent """
    gd = GradientDescent( target_fun=y_grad, fun_gradient=grad_flag, 
                          print_iter_flag=True, print_every_iter=100, save_history_flag=True,
                          max_iter=5e4, dx=1e-10, tol=1e-5, lr=1e-2)
    test(gd, x_init, N=1)

    """ example momentum """
    momentum = Momentum( target_fun=y_grad, fun_gradient=grad_flag, 
                         print_iter_flag=True, print_every_iter=100, save_history_flag=True,
                         max_iter= 5e4, dx=1e-10, tol=1e-5, v_init=0.0, lr=0.001, beta=0.9 )
    test(momentum, x_init, N=1)

    """ example AdaGrad """
    adagrad = AdaGrad( target_fun=y_grad, fun_gradient=grad_flag, 
                       print_iter_flag=True, print_every_iter=100, save_history_flag=True,
                       max_iter= 5e4, dx=1e-10, tol=1e-5, epsilon=1e-8, lr=0.5 )
    test(adagrad, x_init, N=1)

    """ example AdaDelta """
    adadelta = AdaDelta( target_fun=y_grad, fun_gradient=grad_flag, 
                         print_iter_flag=True, print_every_iter=100, save_history_flag=True,
                         max_iter= 5e4, dx=1e-10, tol=1e-5, rho=0.9, epsilon=1e-8 )
    test(adadelta, x_init, N=1)


    """ example AdaDelta """
    rmsprop = RMSprop( target_fun=y_grad, fun_gradient=grad_flag, 
                       print_iter_flag=True, print_every_iter=100, save_history_flag=True,
                       max_iter= 5e4, dx=1e-10, tol=1e-5, rho=0.9, epsilon=1e-8, lr=0.01 )
    test(rmsprop, x_init, N=1)

    """ example AdaDelta """
    adam = Adam( target_fun=y_grad, fun_gradient=grad_flag, 
                       print_iter_flag=True, print_every_iter=100, save_history_flag=True,
                       max_iter= 5e4, dx=1e-10, tol=1e-5, 
                       beta1=0.9, beta2=0.999, epsilon=1e-8, lr=0.1 )
    test(adam, x_init, N=1)

    """ example Scipy_LBFGS_B """
    scipy_optimizer = Scipy_LBFGS_B(target_fun=y_grad, fun_gradient=grad_flag, max_iter= 5e4, 
                                    print_iter_flag=True, print_every_iter=1, save_history_flag=True,
                                    dx=1e-10, ftol=1e-11, gtol=1e-08, eps=1e-08, bnd=(0,None))
    test(scipy_optimizer, x_init, N=1)
