#!/usr/bin/env python3
# Copyright (c) 2022 Wei-Kai Lee. All rights reserved

# coding=utf-8
# -*- coding: utf8 -*

from scipy import optimize
import copy
import numpy as np
try:
    from GradientDescentOptimizer import numerical_gradient
    from GradientDescentOptimizer import GradientDescentOptimizer
except:
    from .GradientDescentOptimizer import numerical_gradient
    from .GradientDescentOptimizer import GradientDescentOptimizer
class Scipy_LBFGS_B(GradientDescentOptimizer):
    def __init__(self,target_fun=None, fun_gradient=None, name:str='Scipy_LBFGS_B', max_iter:int= 5e4, 
                 print_iter_flag:bool=True, print_every_iter:int=1, save_history_flag:bool=False,
                 dx:float=1e-10, ftol:float=1e-11, gtol:float=1e-08, eps:float=1e-08, bnd:tuple=(0,None) ):
        super().__init__( target_fun=target_fun, fun_gradient=fun_gradient, name=name, max_iter=max_iter, 
                          print_iter_flag=print_iter_flag, print_every_iter=print_every_iter, save_history_flag=save_history_flag,
                          dx=dx, tol=None )
        if fun_gradient is None:
            self.__fun_gradient = lambda x: numerical_gradient( self.target_fun, x, dx=self._parameter['dx_num_grad'])
        else:
            self.__fun_gradient = fun_gradient
        self._parameter['ftol'] = ftol
        self._parameter['gtol'] = gtol 
        self._parameter['eps']  = eps
        self._parameter['bnd']  = bnd
    """ execution (virtual function) """
    def optimize(self, x_init:np.ndarray)->np.ndarray:
        """ optimization """

        # prepare jacobian function 
        options = {'ftol': self._parameter['ftol'], 
                   'gtol': self._parameter['gtol'], 
                   'eps' : self._parameter['eps'] }
        # pre-iteration
        self.reset_history()
        self._pre_iteration(x_init)
        if self._parameter['print_iter_flag']:
            self._print_one(self._parameter['counter'])
        if self._parameter['save_history_flag']:
            self._add_one_to_history()

        # iteration
        def callbackF(Xi):
            self._parameter['counter'] += 1
            self._dx = Xi - self._x
            self._x = copy.deepcopy(Xi)
            self._y = self.target_fun(self._x)
            if ( self._parameter['print_iter_flag'] and (self._parameter['counter']%self._parameter['print_every_iter']==0)):
                self._print_one(self._parameter['counter'])
            if self._parameter['save_history_flag']:
                self._add_one_to_history()
                
        results = optimize.minimize( 
                      fun=self.gradient, 
                      x0=x_init, 
                      method="L-BFGS-B", 
                      jac=True, 
                      callback=callbackF, 
                      bounds=[self._parameter['bnd'] ] * x_init.size, 
                      options=options)

        # post iteration
        self._post_iteration()
        return self.optimized_x
    """ property """
    @property
    def ftol(self):
        return self._parameter['ftol']
    @property
    def gtol(self):
        return self._parameter['gtol']
    @property
    def eps(self):
        return self._parameter['eps']
    @property
    def bnd(self):
        return self._parameter['bnd']
    """ setter """
    @ftol.setter
    def ftol(self, ftol:float):
        self._parameter['ftol'] = ftol
    @gtol.setter
    def gtol(self, gtol:float):
        self._parameter['gtol'] = gtol
    @eps.setter
    def eps(self, eps:float):
        self._parameter['eps'] = eps
    @bnd.setter
    def bnd(self, bnd:tuple):
        self._parameter['bnd'] = bnd

if __name__ == '__main__':
    from GradientDescentOptimizer import timer, test 
    """ example """
    min1, min2 = 1, 0.5
    y = lambda x: (x[0]-min1)**2 + (x[1]-min2)**2
    grad = lambda x: np.array( [2*(x[0]-min1), 2*(x[1]-min2)] )
    y_grad = lambda x: ( y(x), grad(x) )
    x_init = np.array( [1.5, 1.0] )

    """ example gradient descent """
    scipy_optimizer = Scipy_LBFGS_B(target_fun=y, fun_gradient=y_grad, max_iter= 5e4, 
                                    print_iter_flag=True, print_every_iter=1, save_history_flag=True,
                                    dx=1e-10, ftol=1e-11, gtol=1e-08, eps=1e-08, bnd=(0,None))
    test(scipy_optimizer, x_init, N=1)
