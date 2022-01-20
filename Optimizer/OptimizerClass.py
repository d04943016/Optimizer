#!/usr/bin/env python3
# Copyright (c) 2022 Wei-Kai Lee. All rights reserved

# coding=utf-8
# -*- coding: utf8 -*

import os
import copy
import json
import numpy as np
import matplotlib.pyplot as plt

class Optimizer:
    """
    Optimizer is a virtual class for optimization method. 
    """
    def __init__(self, target_fun=None, name:str = ''):
        self.name = name # the name of optimizer
        self._target_fun = target_fun # the target function 
        self._parameter = {} # the parameter's that needs to be saved
        self.state_reset() 
    """ untility """
    def __repr__(self):
        """ oprimizer string """
        tmp = '' if (self.name is None) else '('+self.name+')'
        return 'Optimizer' + tmp    
    def save_parameter(self, savefilename:str =None, savefilepath:str ='./'):
        """ save optimizer's parameters """
        if savefilename is None:
            savefilename = self.name
        if not os.path.isdir(savefilepath):
            os.makedirs( savefilepath )
        json.dump(self._parameter, open( os.path.join(savefilepath, savefilename+'_parameter.txt'),'w'))
        return self
    def read_parasmeter(self, filename:str, filepath:str = './'):
        """ read optimizer parameters """
        self._parameter = json.load(open( os.path.join(filepath, filename ) ))
        return self
    def print_parameter(self):
        """ print parameter's information """
        print('Optimizer : '+self.name)
        print('-'*50 )
        for key in self._parameter:
            print('{0:>20s} : {1}'.format(key, self._parameter[key]))
        return self
    def print_state(self):
        """ print the optimizer's state """
        if not self.optimized_flag:
            print('Opitimization is not started.')
        else:
            print('Opitimization is finished.')
            print('optimized y : ', self.optimized_y)
            print('optimized x : ', self.optimized_x)
        return self
    def state_reset(self):
        """ reset the state """
        self._x = None
        self._y = None
        return self
    """ execution (virtual function) """
    def optimize(self, x_init:np.ndarray)->np.ndarray:
        """ optimization """
        raise NotImplementedError
    """ property """
    @property
    def target_fun(self):
        return self._target_fun 
    @property
    def optimized_flag(self)->bool:
        """ whether the target function is optimized or not """
        return (self._x is not None)
    @property
    def optimized_x(self)->np.ndarray:
        """ the optimized x """
        if self._x.size == 1:
            return self._x
        return tuple( self._x )
    @property
    def optimized_y(self)->np.ndarray:
        """ optimized y """
        return self._y
    """ setter """
    @target_fun.setter
    def target_fun(self, target_fun):
        """ target function """
        self._target_fun = target_fun
        self.state_reset()
class IterationOptimizer(Optimizer):
    def __init__(self, target_fun=None, name:str='', max_iter:int= 1e3, print_iter_flag:bool=True, 
                 print_every_iter:int=10, save_history_flag:bool=False):
        super().__init__( target_fun=target_fun, name=name )
        self._parameter['max_iter'] = int(max_iter)
        self._parameter['counter'] = -1
        self._parameter['print_iter_flag'] = print_iter_flag
        self._parameter['print_every_iter'] = int(print_every_iter)
        self._parameter['save_history_flag'] = save_history_flag
        self.reset_history()
        self.state_reset()
    """ untility """
    def state_reset(self):
        super().state_reset()
        self._parameter['counter'] = -1
        return self
    def print_state(self):
        """ print parameter's information """
        super().print_state()
        if self.optimized_flag:
            print('iteration number : ',  self._parameter['counter'])
        return self
    def reset_history(self):
        """ reset history """
        self.count_list = []
        self.x_history = []
        self.target_history = []
        return self
    def save_history(self, savefilename:str=None, savefilepath:str='./'):
        """ save history """
        if not self._parameter['save_history_flag']:
            return
        if savefilename is None:
            savefilename = self.name
        if not os.path.isdir(savefilepath):
            os.makedirs( savefilepath )
        keytuple, hist_dict = self.history_dict
        with open( os.path.join( savefilepath, savefilename+'_history.txt'), 'w') as file:
            # first line
            for key in keytuple:
                file.write('{0:>15}'.format(key))
            file.write('\n')
            # data
            count = len( hist_dict[keytuple[0]] )
            for ii in range(count):
                for key in keytuple:
                    data = hist_dict[key][ii]
                    if isinstance( data, int ):
                        file.write('{0:>15d}'.format( data ))
                    else:
                        file.write('{0:>15.5e}'.format( data ))
                file.write('\n')
        return self
    def plot_history(self, savefilename:str=None, savefilepath:str='./', showbool=False):
        """ plot history """
        if not self._parameter['save_history_flag']:
            return
        if savefilename is None:
            savefilename = self.name
        if not os.path.isdir(savefilepath):
            os.makedirs( savefilepath )
        keytuple, hist_dict = self.history_dict
        count_list = hist_dict['count_list']
        for key in keytuple:
            if key=='count_list':
                continue
            plt.plot(count_list, hist_dict[key])
            plt.xlabel('count')
            plt.ylabel(key)
            plt.savefig( os.path.join( savefilepath, savefilename+'_'+key+'_history.png'), bbox_inches='tight' )
            if showbool:
                plt.show()
            plt.close()
        return self
    """ execution (workflow) """
    def optimize(self, x_init:np.ndarray)->np.ndarray:
        """ optimization workflow """
        # pre-iteration
        self.reset_history()
        self._pre_iteration(x_init)
        if self._parameter['print_iter_flag']:
            self._print_one()
        if self._parameter['save_history_flag']:
            self._add_one_to_history()
        # iteration
        self._updata_one()
        while self._termination():
            if ( self._parameter['print_iter_flag'] and (self._parameter['counter']%self._parameter['print_every_iter']==0)):
                self._print_one()
            if self._parameter['save_history_flag']:
                self._add_one_to_history()
            self._updata_one()
        # post iteration
        self._post_iteration()
        if self._parameter['print_iter_flag']:
            self._print_one()
        if self._parameter['save_history_flag']:
            self._add_one_to_history()
        return self.optimized_x
    """ protected function """
    def _pre_iteration(self, x_init:np.ndarray):
        """ preparation before iteration """
        self._parameter['counter'] = 0
        self._x = copy.deepcopy( x_init )
        self._y = self.target_fun(self._x)
        return self
    def _add_one_to_history(self):
        """ add the current state into history list """
        self.count_list.append( self._parameter['counter'] ) 
        self.x_history.append( copy.deepcopy( self._x )) 
        self.target_history.append( self._y ) 
        return self
    def _print_one(self):
        """ print the information in one step """
        raise NotImplementedError
        return self
    def _updata_one(self):
        """ update the parameter in one step (virtual) """
        raise NotImplementedError
        return self
    def _termination(self):
        raise NotImplementedError
        return True
    def _post_iteration(self):
        """ after iteration """
        return self
    """ property """
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
                    / 'count_list', 'y_history', ''x_history'
        his_dict  : a dictory of histroy 
        """
        if (not self.optimized_flag) or (not self._parameter['save_history_flag']):
            return [], {}
        tmp_dict = {'count_list': copy.deepcopy( self.count_list ) ,
                    'y_history': np.array( self.target_history) }
        key_list = ['count_list', 'y_history']
        # if x_history is a list or array not a single value
        # change x_history to x[0]_history, x[1]_history, ...
        if ((not isinstance( self.x_history[0], list )) and 
            ( (not isinstance( self.x_history[0], np.ndarray )) or self.x_history[0].ndim==0 ) ):
            tmp_dict['x_history'] = np.array( self.x_history )
            key_list.append( 'x_history' )
        else:
            for ii in range( len(self.x_history[0])  ):
                tmp_key = 'x_{0}_history'.format(ii)
                key_list.append(tmp_key)
                tmp_dict[tmp_key]= np.array( [ l1[ii] for l1 in self.x_history] )
        return tuple(key_list), tmp_dict
    @property
    def max_iter(self):
        return self._parameter['max_iter']
    @property
    def counter(self):
        return self._parameter['counter']
    @property
    def print_iter_flag(self):
        return self._parameter['print_iter_flag']
    @property
    def print_every_iter(self):
        return self._parameter['print_every_iter']
    @property
    def save_history_flag(self):
        return self._parameter['save_history_flag']
    """ setter """
    @max_iter.setter
    def max_iter(self, max_iter:int):
        self._parameter['max_iter'] = int(max_iter)
    @print_iter_flag.setter
    def print_iter_flag(self, print_iter_flag:bool):
        self._parameter['print_iter_flag'] = bool(print_iter_flag)
    @print_every_iter.setter
    def print_every_iter(self, print_every_iter:int):
        self._parameter['print_every_iter'] = int(print_every_iter)
    @save_history_flag.setter
    def save_history_flag(self, save_history_flag:bool):
        self._parameter['save_history_flag'] = int(save_history_flag)
class DummyOptimizer(Optimizer):
    def __init__(self, target_fun=None, name:str='Dummy' ):
        super().__init__( target_fun=target_fun, name=name )
    def optimize(self, x_init:np.ndarray)->np.ndarray:
        self._x = copy.deepcopy(x_init)
        self._y = self.target_fun(x_init)
        return self.optimized_x
class MultiOptimizer(Optimizer):
    def __init__(self, target_fun=None, name:str='MultiOptimizer', optimizer_list = None ):
        self._optimizer_list = optimizer_list
        if self._optimizer_list is None:
            self._optimizer_list = None
        super().__init__( target_fun=target_fun, name=name )
    def __iter__(self):
        for optimizer in self._optimizer_list:
            yield optimizer
    def optimize(self, x_init:np.ndarray)->np.ndarray:
        self._x = copy.deepcopy(x_init)
        print('Optimizer : ', self.name)
        for ii, optimizer in enumerate(self.optimizer_list):
            print('Optimizer [{0}] : '.format(ii), optimizer.name)
            self._x = np.asarray( optimizer.optimize(self._x) )
        self._y = self.target_fun(self._x)
        return self.optimized_x
    def save_parameter(self, savefilename:str =None, savefilepath:str ='./'):
        if savefilename is None:
            savefilename = self.name
        super().save_parameter(savefilename, savefilepath)
        for ii,optimizer in enumerate(self.optimizer_list):
            tmp = savefilename+'_idx_{0}_'.format(ii)+optimizer.name
            optimizer.save_parameter(tmp, savefilepath)
        return self
    def print_parameter(self):
        """ print parameter's information """
        super().print_parameter()
        for ii,optimizer in enumerate(self.optimizer_list):
            print('')
            print('- sub-optimizer[{0}]'.format(ii))
            optimizer.print_parameter()
        return self
    def print_state(self):
        """ print the optimizer's state """
        super().print_parameter()
        for ii, optimizer in enumerate(self.optimizer_list):
            print('')
            print('- sub-optimizer[{0}]'.format(ii))
            optimizer.print_state()
        return self
    def state_reset(self):
        super().state_reset()
        for optimizer in self.optimizer_list:
            optimizer.state_reset()
        return self
    """ property """
    @property
    def optimizer_list(self):
        return self._optimizer_list
    @property
    def target_fun(self):
        return self._target_fun 
    """ setter """
    @optimizer_list.setter
    def optimizer_list(self, optimizer_list):
        self._optimizer_list = optimizer_list
        if (self._optimizer_list) is None:
            self._optimizer_list = []
        self.state_reset()
    @target_fun.setter
    def target_fun(self, target_fun):
        """ target function """
        self._target_fun = target_fun
        self.state_reset()
        for optimizer in self.optimizer_list:
            optimizer.target_fun = target_fun
    @property
    def optimized_flag(self)->bool:
        """ whether the target function is optimized or not """
        flag = super().optimized_flag
        for optimizer in self.optimizer_list:
            flag = ( flag and optimizer.optimized_flag )
        return flag
class MultiIterationOptimizer(MultiOptimizer):
    def __init__(self, target_fun=None, name:str='MultiIterationOptimizer', optimizer_list = None, 
                 print_iter_flag:bool=True, save_history_flag:bool=False ):
        super().__init__( target_fun=target_fun, name=name, optimizer_list=optimizer_list )
        self.print_iter_flag = print_iter_flag
        self.save_history_flag = save_history_flag
    """ utility """
    def reset_history(self):
        """ reset history """
        for optimizer in self.optimizer_list:
            optimizer.reset_history()
        return self
    def save_history(self, savefilename:str=None, savefilepath:str='./'):
        """ save history """
        if not self._parameter['save_history_flag']:
            return
        if savefilename is None:
            savefilename = self.name
        if not os.path.isdir(savefilepath):
            os.makedirs( savefilepath )
        keytuple, hist_dict = self.history_dict
        with open( os.path.join( savefilepath, savefilename+'_history.txt'), 'w') as file:
            # first line
            for key in keytuple:
                file.write('{0:>15}'.format(key))
            file.write('\n')
            # data
            count = len( hist_dict[keytuple[0]] )
            for ii in range(count):
                for key in keytuple:
                    data = hist_dict[key][ii]
                    if isinstance( data, int ):
                        file.write('{0:>15d}'.format( data ))
                    else:
                        file.write('{0:>15.5e}'.format( data ))
                file.write('\n')
        return self
    def plot_history(self, savefilename:str=None, savefilepath:str='./', showbool=False):
        """ plot history """
        if not self._parameter['save_history_flag']:
            return
        if savefilename is None:
            savefilename = self.name
        if not os.path.isdir(savefilepath):
            os.makedirs( savefilepath )
        keytuple, hist_dict = self.history_dict
        count_list = hist_dict['count_list']
        for key in keytuple:
            if key=='count_list':
                continue
            plt.plot(count_list, hist_dict[key])
            plt.xlabel('count')
            plt.ylabel(key)
            plt.savefig( os.path.join( savefilepath, savefilename+'_'+key+'_history.png'), bbox_inches='tight' )
            if showbool:
                plt.show()
            plt.close()
        return self
    """ property """
    @property 
    def print_iter_flag(self):
        return self._parameter['print_iter_flag']
    @property
    def save_history_flag(self):
        return self._parameter['save_history_flag']
    @property
    def counter(self):
        counter = 0
        for optimizer in self.optimizer_list:
            counter += optimizer.counter
        return counter
    @property
    def history_dict(self):
        # get histories
        key_list, histories = self.histories_list
        # count_list and y_history 
        tmp_dict = {}
        for key in key_list[1::]:
            tmplist = np.asarray([])
            for ii, history in enumerate( histories ):
                tmplist = np.append( tmplist, history[key] if (ii==0) else history[key][1::] )
            tmp_dict[key] = tmplist
        key_list = ['count_list'] + list(key_list)
        tmp_dict['count_list'] = np.arange( len(tmp_dict['y_history']) )
        return tuple(key_list), tmp_dict
    @property
    def histories_list(self):
        key_list, histories = None, [None] * len(self.optimizer_list)
        for ii, optimizer in enumerate( self.optimizer_list ):
            key_list, histories[ii] = optimizer.history_dict
        return key_list, histories
    """
    @property
    def history_dict(self):
        # get histories
        keys_list, histories = self.histories_list
        # count_list and y_history # 1st elemnet:count_list, 2nd: y_history
        y_history = np.asarray( [] )
        for ii, history in enumerate( histories ):
            y_history = np.append( y_history, history['y_history'] if (ii==0) else history['y_history'][1::] )
        count_list = np.arange( len(y_history) )
        tmp_dict = {'count_list': count_list, 'y_history': np.asarray( y_history ) }
        # others
        key_list = set( keys_list[0][2::] ).intersection( *keys_list[1::] ) 
        for key in key_list:
            tmp_list = np.asarray( [] )
            for ii, history in enumerate( histories ):
                tmp_list = np.append( tmp_list, history[key] if ii==0 else history[key][1::] )
            tmp_dict[key] = tmp_list
        key_list = ['count_list', 'y_history'] + list(key_list)
        return tuple(key_list), tmp_dict
    @property
    def histories_list(self):
        keys_list, histories = [None] * len(self.optimizer_list), [None] * len(self.optimizer_list)
        for ii, optimizer in enumerate( self.optimizer_list ):
            keys_list[ii], histories[ii] = optimizer.history_dict
        return keys_list, histories
    """
    """ setter """
    @print_iter_flag.setter
    def print_iter_flag(self, print_iter_flag:bool):
        for optimizer in self.optimizer_list:
            optimizer.print_iter_flag = print_iter_flag
        self._parameter['print_iter_flag'] = print_iter_flag
    @save_history_flag.setter
    def save_history_flag(self, save_history_flag:bool):
        for optimizer in self.optimizer_list:
            optimizer.save_history_flag = save_history_flag
        self._parameter['save_history_flag'] = save_history_flag
