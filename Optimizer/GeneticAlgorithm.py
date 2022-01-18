#!/usr/bin/env python3
# Copyright (c) 2022 Wei-Kai Lee. All rights reserved

# coding=utf-8
# -*- coding: utf8 -*

import copy
import numpy as np

from OptimizerClass import IterationOptimizer

class GeneticAlgorithm(IterationOptimizer):
    def __init__(self, target_fun=None, fun_gradient=None, name:str='', max_iter:int= 5e4, 
                 print_iter_flag:bool=True, print_every_iter:int=10, save_history_flag:bool=False):
        super().__init__( target_fun=target_fun, name=name, max_iter=max_iter, 
                          print_iter_flag=print_iter_flag, print_every_iter=print_every_iter,
                          save_history_flag=save_history_flag )
    def _pre_iteration(self, x_init:np.ndarray):
        """ preparation before iteration """
        super()._pre_iteration(x_init)
        pass
    def _add_one_to_history(self):
        """ add the current state into history list """
        super()._add_one_to_history()
        pass
    def _print_one(self, counter:int):
        """ print information in one step"""
        pass
    def _updata_one(self):
        """ update the parameter in one step"""
        pass