#!/usr/bin/env python3
# Copyright (c) 2022 Wei-Kai Lee. All rights reserved

# coding=utf-8
# -*- coding: utf8 -*

# reference [1]: https://zhuanlan.zhihu.com/p/22252270
# reference [2]: https://zh.d2l.ai/chapter_optimization/adadelta.html

import copy
import numpy as np

try:
    from OptimizerClass import IterationOptimizer
except:
    from .OptimizerClass import IterationOptimizer

def numerical_gradient(fun, x, dx=1e-10, dtype=np.float_):
    y = fun( x )
    if x.ndim ==0:
        x = np.array( [x], dtype=x.dtype )
    dy_dx = np.zeros( (len(x),), dtype=dtype) 
    for ii in range( len(x) ):
        # x + dx
        x_tmp = copy.deepcopy( x )
        x_tmp[ii] += dx
        # derivative
        dy_dx[ii] = ( fun(x_tmp)-y )/dx
    if dy_dx.size==1:
        return dy_dx[0]
    return y, dy_dx
class GradientDescentOptimizer(IterationOptimizer):
    def __init__(self, target_fun=None, fun_gradient=None, name:str='', max_iter:int= 5e4, 
                 print_iter_flag:bool=True, print_every_iter:int=10, save_history_flag:bool=False,
                 dx:float=1e-10, tol:float=1e-5):
        super().__init__( target_fun=target_fun, name=name, max_iter=max_iter, 
                          print_iter_flag=print_iter_flag, print_every_iter=print_every_iter,
                          save_history_flag=save_history_flag )
        self._parameter['dx_num_grad'] = dx
        self._parameter['tol'] = tol
        if fun_gradient is None:
            self.__fun_gradient = lambda x: numerical_gradient( self.target_fun, x, dx=self._parameter['dx_num_grad'])
        else:
            self.__fun_gradient = fun_gradient
        self.state_reset()
    """ untility (overload) """
    def print_state(self):
        """ print parameter's information """
        super().print_state()
        if self.optimized_flag:
            print('dx : ',  self._dx)
        return self
    def reset_history(self):
        """ reset history """
        super().reset_history()
        self.dx_history = []
        return self
    def state_reset(self):
        super().state_reset()
        self._dx = np.inf
        return self
    """ calculation """
    def gradient(self, x:np.ndarray):
        """ gradient of target function """
        if (self.fun_gradient is not None):
            tmp = self.fun_gradient(x)
            if isinstance(tmp, tuple):
                return tmp # y, dy_dx
            y = self.target_fun(x)
            return y, tmp
        assert False
    """ protected (overload) """
    def _pre_iteration(self, x_init:np.ndarray):
        """ preparation before iteration """
        super()._pre_iteration(x_init)
        self._dx = -np.ones( x_init.shape, dtype=x_init.dtype)
        return self
    def _add_one_to_history(self):
        """ add the current state into history list """
        super()._add_one_to_history()
        self.dx_history.append( copy.deepcopy( self._dx))
        return self
    def _print_one(self):
        """ print the information in one step """
        if self._parameter['counter'] == 0:
            print( '{0:>10s}  {1:>10s}  {2:>10s} {3:<10s}'.format('count', 'y', 'max(abs(dx))', 'x') )
            print( '-'*47 )
        str1 = '{0}'.format(self._x) 
        print( '{0:>10d}  {1:>10.5e}  {2:>10.5e} {3:<10s}'.format(self._parameter['counter'], self._y, np.max( np.abs(self._dx) ), str1) )
        return self
    def _termination(self):
        return ( np.max( np.abs(self._dx) ) > self._parameter['tol'] ) & (self._parameter['counter']<self._parameter['max_iter'])
    """ property (overload) """
    @property
    def history_dict(self):
        """ 
        return a history dict 
        
        Usage
        -----
        key_tuple, his_dict = optimizer.history_dict
        
        Returns
        -------
        key_tuple : a tuple with key of his_dict (with correct output order)
                    / 'count_list', 'y_history', ''x_history', 'dx_history'
        his_dict  : a dictory of histroy 
        """
        if (not self.optimized_flag) or (not self._parameter['save_history_flag']):
            return [], {}
        key_tuple, tmp_dict = super().history_dict
        key_list = list(key_tuple)
        # if dx_history is a list or array not a single value
        # change dx_history to dx[0]_history, dx[1]_history, ...
        if ((not isinstance( self.dx_history[0], list )) and 
            ( (not isinstance( self.dx_history[0], np.ndarray )) or self.dx_history[0].ndim==0 ) ):
            tmp_dict['dx_history'] = copy.deepcopy(self.dx_history)
            key_list.append( 'dx_history' )
        else:
            for ii in range( len(self.dx_history[0])  ):
                tmp_key = 'dx[{0}]_history'.format(ii)
                key_list.append(tmp_key)
                tmp_dict[tmp_key]= [ l1[ii] for l1 in self.dx_history]
        tmp_dict['dx_history'] = copy.deepcopy(self.dx_history)
        key_tuple = key_tuple + ('dx_history',)
        return tuple(key_list), tmp_dict
    @property
    def fun_gradient(self):
        return self.__fun_gradient
    @property
    def dx_num_grad(self):
        return self._parameter['dx_num_grad']
    @property
    def tol(self):
        return self._parameter['tol']
    """ setter """
    @fun_gradient.setter
    def fun_gradient(self, fun_gradient):
        """ gradient of target function """
        self.__fun_gradient = fun_gradient
        self.state_reset()
    @dx_num_grad.setter
    def dx_num_grad(self, dx_num_grad:float):
        self._parameter['dx_num_grad'] = dx_num_grad
    @tol.setter
    def tol(self, tol:float):
        self._parameter['tol'] = tol
class GradientDescent(GradientDescentOptimizer):
    def __init__(self, target_fun=None, fun_gradient=None, name:str='GradientDescent', max_iter:int= 5e4, 
                 print_iter_flag:bool=True, print_every_iter:int=10, save_history_flag:bool=False,
                 dx:float=1e-10, tol:float=1e-5, lr:float=1e-2):
        super().__init__( target_fun=target_fun, fun_gradient=fun_gradient, 
                          name=name, max_iter= max_iter, 
                          print_iter_flag=print_iter_flag,print_every_iter=print_every_iter, 
                          save_history_flag=save_history_flag, 
                          dx=dx, tol=tol )
        self._parameter['learning_rate'] = lr
    """ protected (overload) """
    def _updata_one(self):
        """ update the parameter in one step"""
        self._parameter['counter'] += 1
        self._x += self._dx
        self._y, dy_dx = self.gradient( self._x )
        self._dx = -self._parameter['learning_rate']*dy_dx
    """ property """
    @property
    def learning_rate(self):
        return self._parameter['learning_rate']
    """ setter """
    @property
    def learning_rate(self, learning_rate:float):
        self._parameter['learning_rate'] = learning_rate
class Momentum(GradientDescentOptimizer):
    def __init__(self, target_fun=None, fun_gradient=None, name:str='Momentum', max_iter:int= 5e4, 
                 print_iter_flag:bool=True, print_every_iter:int=10, save_history_flag:bool=False,
                 dx:float=1e-10, tol:float=1e-5, v_init:float=0.0, lr:float=0.001, beta:float=0.9):
        super().__init__( target_fun=target_fun, fun_gradient=fun_gradient, 
                          name=name, max_iter= max_iter, 
                          print_iter_flag=print_iter_flag,print_every_iter=print_every_iter, 
                          save_history_flag=save_history_flag, 
                          dx=dx, tol=tol )
        self._parameter['v_init'] = v_init
        self._parameter['learning_rate'] = lr
        self._parameter['beta'] = beta
    def _pre_iteration(self, x_init):
        super()._pre_iteration(x_init)
        self._dx = np.ones( x_init.shape, dtype=x_init.dtype) * self._parameter['v_init']
        return self
    def _updata_one(self):
        """ update the parameter in one step"""
        self._parameter['counter'] += 1
        self._x += self._dx
        self._y, dy_dx = self.gradient( self._x )
        self._dx = self._parameter['beta'] * self._dx - self._parameter['learning_rate'] * dy_dx
    """ property """
    @property
    def v_init(self):
        return self._parameter['v_init']
    @property
    def learning_rate(self):
        return self._parameter['learning_rate']
    @property
    def beta(self):
        return self._parameter['beta']
    """ setter """
    @v_init.setter
    def v_init(self, v_init:float):
        self._parameter['v_init'] = v_init
    @learning_rate.setter
    def learning_rate(self, learning_rate:float):
        self._parameter['learning_rate'] = learning_rate
    @beta.setter
    def beta(self, beta:float):
        self._parameter['beta'] = beta
class AdaGrad(GradientDescentOptimizer):
    def __init__(self, target_fun=None, fun_gradient=None, name:str='AdaGrad', max_iter:int= 5e4, 
                 print_iter_flag:bool=True, print_every_iter:int=10, save_history_flag:bool=False,
                 dx:float=1e-10, tol:float=1e-5, epsilon:float=1e-8, lr:float=0.5):
        super().__init__( target_fun=target_fun, fun_gradient=fun_gradient, 
                          name=name, max_iter= max_iter, 
                          print_iter_flag=print_iter_flag,print_every_iter=print_every_iter, 
                          save_history_flag=save_history_flag, 
                          dx=dx, tol=tol )
        self._parameter['epsilon'] = epsilon
        self._parameter['learning_rate'] = lr
        self.state_reset()
    def state_reset(self):
        super().state_reset()
        self.__sigma = None
        return self
    def _pre_iteration(self, x_init):
        super()._pre_iteration(x_init)
        self.__sigma = np.zeros( x_init.shape, dtype=np.float64 )
        return self
    def _updata_one(self):
        """ update the parameter in one step"""
        self._parameter['counter'] += 1
        self._x += self._dx
        self._y, dy_dx = self.gradient( self._x )
        self.__sigma += dy_dx**2
        self._dx = - self._parameter['learning_rate'] * dy_dx/np.sqrt( self.__sigma+self._parameter['epsilon'] )
        return self
    """ property """
    @property
    def epsilon(self):
        return self._parameter['epsilon']
    @property
    def learning_rate(self):
        return self._parameter['learning_rate']
    """ setter """
    @epsilon.setter
    def epsilon(self, epsilon):
        self._parameter['epsilon'] = epsilon
    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self._parameter['learning_rate'] = learning_rate
class AdaDelta(GradientDescentOptimizer):
    def __init__(self, target_fun=None, fun_gradient=None, name:str='AdaDelta', max_iter:int= 5e4, 
                 print_iter_flag:bool=True, print_every_iter:int=10, save_history_flag:bool=False,
                 dx:float=1e-10, tol:float=1e-5, rho:float=0.9, epsilon:float=1e-8):
        super().__init__( target_fun=target_fun, fun_gradient=fun_gradient, 
                          name=name, max_iter= max_iter, 
                          print_iter_flag=print_iter_flag,print_every_iter=print_every_iter, 
                          save_history_flag=save_history_flag, 
                          dx=dx, tol=tol )
        self._parameter['rho'] = rho
        self._parameter['epsilon'] = epsilon
        self.state_reset()
    def state_reset(self):
        super().state_reset()
        self.__sigma = None
        self.__deltaX = None
        return self
    def _pre_iteration(self, x_init):
        super()._pre_iteration(x_init)
        self.__sigma = np.zeros( x_init.shape, dtype=np.float64 )
        self.__deltaX = np.zeros( x_init.shape, dtype=np.float64 )
        return self
    def _updata_one(self):
        """ update the parameter in one step"""
        self._parameter['counter'] += 1
        self._x += self._dx
        self._y, dy_dx = self.gradient( self._x )
        self.__sigma = self._parameter['rho']*self.__sigma+(1-self._parameter['rho'])*(dy_dx**2)
        self._dx = - np.sqrt( self.__deltaX+self._parameter['epsilon'] ) * dy_dx/np.sqrt( self.__sigma+self._parameter['epsilon'] )
        self.__deltaX = self._parameter['rho']*self.__deltaX+(1-self._parameter['rho'])*(self._dx**2)
        return self
    """ property """
    @property
    def rho(self):
        return self._parameter['rho']
    @property
    def epsilon(self):
        return self._parameter['epsilon']
    """ setter """
    @rho.setter
    def rho(self, rho:float):
        self._parameter['rho'] = rho
    @epsilon.setter
    def epsilon(self, epsilon:float):
        self._parameter['epsilon'] = epsilon
class RMSprop(GradientDescentOptimizer):
    def __init__(self, target_fun=None, fun_gradient=None, name:str='RMSprop', max_iter:int= 5e4, 
                 print_iter_flag:bool=True, print_every_iter:int=10, save_history_flag:bool=False,
                 dx:float=1e-10, tol:float=1e-5, rho:float=0.9, epsilon:float=1e-8, lr:float=0.01):
        super().__init__( target_fun=target_fun, fun_gradient=fun_gradient, 
                          name=name, max_iter= max_iter, 
                          print_iter_flag=print_iter_flag,print_every_iter=print_every_iter, 
                          save_history_flag=save_history_flag, 
                          dx=dx, tol=tol )
        self._parameter['rho'] = rho
        self._parameter['epsilon'] = epsilon
        self._parameter['learning_rate'] = lr
        self.state_reset()
    def state_reset(self):
        super().state_reset()
        self.__sigma = None
        return self
    def _pre_iteration(self, x_init):
        super()._pre_iteration(x_init)
        self.__sigma = np.zeros( x_init.size, dtype=np.float64 )
        return self
    def _updata_one(self):
        """ update the parameter in one step"""
        self._parameter['counter'] += 1
        self._x += self._dx
        self._y, dy_dx = self.gradient( self._x )
        self.__sigma = self._parameter['rho']*self.__sigma+(1-self._parameter['rho'])* (dy_dx**2)
        self._dx = - self._parameter['learning_rate'] * dy_dx/np.sqrt( self.__sigma+self._parameter['epsilon'] ) 
        return self
    """ property """
    @property
    def rho(self):
        return self._parameter['rho']
    @property
    def epsilon(self):
        return self._parameter['epsilon']
    @property
    def learning_rate(self):
        return self._parameter['learning_rate']
    """ setter """
    @rho.setter
    def rho(self, rho:float):
        self._parameter['rho'] = rho
    @epsilon.setter
    def epsilon(self, epsilon:float):
        self._parameter['epsilon'] = epsilon
    @learning_rate.setter
    def learning_rate(self, learning_rate:float):
        self._parameter['learning_rate'] = learning_rate
class Adam(GradientDescentOptimizer):
    def __init__(self, target_fun=None, fun_gradient=None, name:str='Adam', max_iter:int= 5e4, 
                 print_iter_flag:bool=True, print_every_iter:int=10, save_history_flag:bool=False,
                 dx:float=1e-10, tol:float=1e-5, 
                 beta1:float=0.9, beta2:float=0.999, epsilon:float=1e-8, lr:float=0.1):
        super().__init__( target_fun=target_fun, fun_gradient=fun_gradient, 
                          name=name, max_iter= max_iter, 
                          print_iter_flag=print_iter_flag,print_every_iter=print_every_iter, 
                          save_history_flag=save_history_flag, 
                          dx=dx, tol=tol )
        self._parameter['beta1'] = beta1
        self._parameter['beta2'] = beta2
        self._parameter['epsilon'] = epsilon
        self._parameter['learning_rate'] = lr
        self.state_reset()
    def state_reset(self):
        super().state_reset()
        self.__vt = None
        self.__sigma = None
        return self
    def _pre_iteration(self, x_init):
        super()._pre_iteration(x_init)
        self.__vt = np.zeros( x_init.size, dtype=np.float64 )
        self.__sigma = np.zeros( x_init.size, dtype=np.float64 )
        return self
    def _updata_one(self):
        """ update the parameter in one step"""
        self._parameter['counter'] += 1
        self._x += self._dx
        self._y, dy_dx = self.gradient( self._x )
        # vt and sigma
        self.__vt = self._parameter['beta1']*self.__vt + (1-self._parameter['beta1'])*dy_dx
        self.__sigma = self._parameter['beta2']*self.__sigma+(1-self._parameter['beta2'])* (dy_dx**2)
        v_bias_corr = self.__vt/(1-self._parameter['beta1']**self._parameter['counter'])
        sigma_bias_corr = self.__sigma/(1-self._parameter['beta2']**self._parameter['counter'])
        # update
        self._dx = - self._parameter['learning_rate'] * v_bias_corr/np.sqrt( sigma_bias_corr+self._parameter['epsilon'] )
        return self
    """ property """
    @property
    def beta1(self):
        return self._parameter['beta1']
    @property
    def beta2(self):
        return self._parameter['beta2']
    @property
    def epsilon(self):
        return self._parameter['epsilon']
    @property
    def learning_rate(self):
        return self._parameter['learning_rate']
    """ setter """
    @beta1.setter
    def beta1(self, beta1:float):
        self._parameter['beta1'] = beta1
    @beta2.setter
    def beta2(self, beta2:float):
        self._parameter['beta2'] = beta2
    @epsilon.setter
    def epsilon(self, epsilon:float):
        self._parameter['epsilon'] = epsilon
    @learning_rate.setter
    def learning_rate(self, learning_rate:float):
        self._parameter['learning_rate'] = learning_rate
""" timer """
def timer(optimier, x_init, N=1):
    import time
    t1 = time.time()
    for ii in range(N):
        optimier.optimize(x_init)
    t2 = time.time()
    print( 'average time = ', (t2-t1)/N )
""" test function """
def test(optimizer, x_init, N=1, savefilepath='./test', save_history=True):
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
        optimizer.save_history(savefilepath=savefilepath)
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
    
    """ example gradient descent """
    gd = GradientDescent( target_fun=y, fun_gradient=y_grad, 
                          print_iter_flag=True, print_every_iter=100, save_history_flag=True,
                          max_iter=5e4, dx=1e-10, tol=1e-5, lr=1e-2)
    test(gd, x_init, N=1)

    """ example momentum """
    momentum = Momentum( target_fun=y, fun_gradient=y_grad, 
                         print_iter_flag=True, print_every_iter=100, save_history_flag=True,
                         max_iter= 5e4, dx=1e-10, tol=1e-5, v_init=0.0, lr=0.001, beta=0.9 )
    test(momentum, x_init, N=1)

    """ example AdaGrad """
    adagrad = AdaGrad( target_fun=y, fun_gradient=y_grad, 
                       print_iter_flag=True, print_every_iter=100, save_history_flag=True,
                       max_iter= 5e4, dx=1e-10, tol=1e-5, epsilon=1e-8, lr=0.5 )
    test(adagrad, x_init, N=1)

    """ example AdaDelta """
    adadelta = AdaDelta( target_fun=y, fun_gradient=y_grad, 
                         print_iter_flag=True, print_every_iter=100, save_history_flag=True,
                         max_iter= 5e4, dx=1e-10, tol=1e-5, rho=0.9, epsilon=1e-8 )
    test(adadelta, x_init, N=1)


    """ example AdaDelta """
    rmsprop = RMSprop( target_fun=y, fun_gradient=y_grad, 
                       print_iter_flag=True, print_every_iter=100, save_history_flag=True,
                       max_iter= 5e4, dx=1e-10, tol=1e-5, rho=0.9, epsilon=1e-8, lr=0.01 )
    test(rmsprop, x_init, N=1)

    """ example AdaDelta """
    adam = Adam( target_fun=y, fun_gradient=y_grad, 
                       print_iter_flag=True, print_every_iter=100, save_history_flag=True,
                       max_iter= 5e4, dx=1e-10, tol=1e-5, 
                       beta1=0.9, beta2=0.999, epsilon=1e-8, lr=0.1 )
    test(adam, x_init, N=1)