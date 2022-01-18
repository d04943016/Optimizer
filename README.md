# Optimizer
**Optimizer** is a module related to optimization method. 
There are two different types of optimizer. 
* class-based    : OptimizerClass.py, GradientDescentOptimizer.py, ScipyOptimizer.py, GeneticAlgorithm.py
* function-based : myLearner.py 


Copyright (c) 2020-2022 Wei-Kai Lee. All rights reserved<br/>
## Requirements
* numpy
* scipy

## references
[1]: https://zhuanlan.zhihu.com/p/22252270 <br/>
[2]: https://zh.d2l.ai/chapter_optimization/adadelta.html <br/>

## class based
(A) `Optimizer(target_fun=None, name:str=""): virtual class` <br/>
base class of optimizer <br/>

(a) parameters<br/>
* `target_fun`: target function<br/>
* `name`: the name of optimizer<br/>

(b) utility function<br/>
* `.save_parameter(savefilename:str =None, savefilepath:str ='./') ->self`: save parameters into json file<br/>
* `.read_parasmeter(filename:str, filepath:str = './')->self`: read parameters from json file<br/>
* `.print_parameter()->self`: print the parameter information<br/>
* `.print_state()->self`: print the state of optimizer<br/>
* `.state_reset()->self`: reset the optimizer. Because the optimizer would save the results when optimizing, the data would be clear in this step<br/>

execution function<br/>
* `.optimize(x_init:np.ndarray)->np.ndarray`: start optimization <br/>

(c) property
* `.target_fun (setter)`: target function, is a callabele function with a single value (i.e. y = obj.target_fun(x) )<br/>
* `.optimized_flag -> bool`: whether the optimizer has been optimized or not<br/>
* `.optimized_x -> numpy.ndarray`: the x value in current state (default: None)<br/>
* `.optimized_y -> numpy.ndarray`: the y value in current state (default: None)<br/>

(B) `IterationOptimizer(target_fun=None, name:str='', max_iter:int= 1e3, print_iter_flag:bool=True, print_every_iter:int=10 save_history_flag:bool=False): virtual class`<br/>
a derived class of **Optimizer**, a virtual class related to iteration optimizing algorithm<br/>

(a) parameters<br/>
* `target_fun`: target function<br/>
* `name`: the name of optimizer<br/>
* `max_iter`: maximum iteration<br/>
* `print_iter_flag`: whether to print the optimizing information when optimzing<br/>
* `print_every_iter`: print the optimizing information in every N step<br/>
* `save_history_flag`: whether to save the history when optimizing<br/>

(b) utility<br/>
* `.reset_history()->self`<br/>: reset the history of optimization
* `.save_history(savefilename:str=None, savefilepath:str='./')->self`: save the history of optimization<br/>
* `.plot_history(savefilename:str=None, savefilepath:str='./')->self`: plot the history of optimization<br/>
* 
(c) property<br/>
* `.history_dict -> dict`: history dictionary<br/>
* `.max_iter (setter) -> int`: maximum iteration<br/>
* `.counter (setter) -> int`: counter<br/>
* `.print_iter_flag (setter) -> bool`: whether to print the optimizing information when optimzing<br/>
* `.print_every_iter (setter) -> int`: print the optimizing information in every N step<br/>
* `.save_history_flag (setter) -> bool`: save_history_flag: whether to save the history when optimizing<br/>

(d) protected function while optimizing<br/>
* `._pre_iteration(x_init:np.ndarrat) -> self`: to prepare the parameters before iteration start<br/>
* `._add_one_to_history() -> self`: to add the information into history in one step<br/>
* `._print_one() -> self (virtual)`: to print the information in one step<br/>
* `._updata_one() -> self (virtual)`: to update the self._x and self._y value<br/>
* `._termination() -> bool (virtual)`: whether the optimizer is terminated or not<br/>
* `._post_iteration() -> self`: function after the iteration<br/>
        

(C) `GradientDescentOptimizer(target_fun=None, fun_gradient=None, name:str='', max_iter:int= 5e4, print_iter_flag:bool=True, print_every_iter:int=10, save_history_flag:bool=False, dx:float=1e-10, tol:float=1e-5)): virtual class`<br/>
a derived class of **IterationOptimizer**, a virtual class related to gradient descent method<br/>

(a) parameter<br/>
* `target_fun`: target function<br/>
* `fun_gradient`: the gradient of target function if fun_gradient is None, the optimizer would use -two-point- method to do the numerical gradient.<br/>
There are two types of output<br/>
a. dy_dx = fun_gradient(x)<br/>
b. y, dy_dx = fun_gradient(x) (faster)<br/>
, where dy_dx is the gradient of target_function with the same size of x and y is the output of target_function.<br/>
* `name:` the name of optimizer<br/>
* `max_iter`: maximum iteration<br/>
* `print_iter_flag`: whether to print the optimizing information when optimzing<br/>
* `print_every_iter`: print the optimizing information in every N step<br/>
* `save_history_flag`: whether to save the history when optimizing<br/>
* `dx`: delta x when numerical gradient calculation<br/>
* `tol`: tolerarance<br/>
the optimizer would stop when it achieve maximum iteration or the maximum absolute x difference in a step is smaller than tol<br/>

(b) utility<br/>
* `.gradient(x:np.ndarray)`: the gradient of target function<br/>

(c) property<br/>
* `.fun_gradient (setter)`: the gradient of target function<br/>
If fun_gradient is None, the optimizer would use -two-point- method to do the numerical gradient. <br/>
There are two types of output<br/>
a. dy_dx = fun_gradient(x)<br/>
b. y, dy_dx = fun_gradient(x) (faster)<br/>
, where dy_dx is the gradient of target_function with the same size of x and y is the output of target_function.<br/>
* `.dx_num_grad (setter) -> float`: delta x when numerical gradient calculation<br/>
* `.tol (setter) -> float`: tolerarance<br/>


(D) `DummyOptimizer(target_fun=None, name:str='Dummy')`<br/>
a derived class of Optimizer that do nothing, <br/>
optimizer.optimze(x) would directly return x and set the optimizer's state<br/>

(a) parameters<br/>
* `target_fun`: target function<br/>
* `name`: the name of optimizer<br/>
    
(E) `MultiOptimizer(target_fun=None, name:str='MultiOptimizer', optimizer_list = None)`<br/>
a derived calss of Optimizer that can combine different optimizer in the order in the optimizer_list<br/>

(a) parameters<br/>
* `target_fun`: target function<br/>
* `name`: the name of optimizer<br/>
* `optimizer_list`: a list of optimizer, when .optimize(x) the target function, MultiOptimizer would follow the same order in optimizer_list to optimize the target function <br/>

(b) utility<br/>
* `.__iter__`: when iterate the MultiOptimizer, it would return the optimizer stored optimizer_list<br/>

(c) property<br/>
* `.optimizer_list (setter)` -> list[Optimizer]<br/>



(F) `GradientDescent(target_fun=None, fun_gradient=None, name:str='GradientDescent', max_iter:int= 5e4, print_iter_flag:bool=True, print_every_iter:int=10, save_history_flag:bool=False, dx:float=1e-10, tol:float=1e-5, lr:float=1e-2)`<br/>
a derived class of GradientDescentOptimizer, using normal gradient descent method<br/>

(a) parameter<br/>
see the input of GradientDescentOptimizer<br/>
* `lr`:  learning_rate<br/>

(b) property<br/>
* `.learning_rate (setter) -> float`: learning rate<br/>

(G) `Momentum(target_fun=None, fun_gradient=None, name:str='Momentum', max_iter:int= 5e4, print_iter_flag:bool=True, print_every_iter:int=10, save_history_flag:bool=False,dx:float=1e-10, tol:float=1e-5, v_init:float=0.0, lr:float=0.001, beta:float=0.9))`<br/>
a derived class of GradientDescentOptimizer, using momentum method<br/>

(a) parameter<br/>
see the input of GradientDescentOptimizer<br/>
* `v_init`: initial velocity<br/>
* `lr`:  learning_rate<br/>
* `beta`: attenuating factor<br/>

(b) property<br/>
* `.v_init (setter) -> float`: initial velocity<br/>
* `.learning_rate (setter) -> float`: learning rate<br/>
* `.beta (setter) -> float`: attenuating factor<br/>

(H) `AdaGrad(target_fun=None, fun_gradient=None, name:str='AdaGrad', max_iter:int= 5e4, print_iter_flag:bool=True, print_every_iter:int=10, save_history_flag:bool=False, dx:float=1e-10, tol:float=1e-5, epsilon:float=1e-8, lr:float=0.5)`<br/>
a derived class of GradientDescentOptimizer, using AdaGrad method<br/>

(a) parameter<br/>
see the input of GradientDescentOptimizer<br/>
* `epsilon`: deniminator factor<br/>
* `lr`:  learning_rate<br/>

(b) property<br/>
* `.epsilon (setter) -> float`: deniminator factor<br/>
* `.learning_rate (setter) -> float`: learning rate<br/>

(I) `AdaDelta(target_fun=None, fun_gradient=None, name:str='AdaDelta',max_iter:int= 5e4, print_iter_flag:bool=True, print_every_iter:int=10, save_history_flag:bool=False, dx:float=1e-10, tol:float=1e-5, rho:float=0.9, epsilon:float=1e-8)`<br/>
a derived class of GradientDescentOptimizer, using AdaDelta method<br/>

(a) parameter<br/>
see the input of GradientDescentOptimizer<br/>
* `rho`: rho<br/>
* `epsilon`:  epsilon<br/>

(b) property<br/>
* `.rho (setter)` -> float: rho<br/>
* `.epsilon (setter)` -> float: epsilon<br/>

(J) `RMSprop(target_fun=None, fun_gradient=None, name:str='RMSprop', max_iter:int= 5e4, print_iter_flag:bool=True, print_every_iter:int=10, save_history_flag:bool=False,dx:float=1e-10, tol:float=1e-5, rho:float=0.9, epsilon:float=1e-8, lr:float=0.01)`<br/>
a derived class of GradientDescentOptimizer, using RMSprop method<br/>

(a) parameter<br/>
see the input of GradientDescentOptimizer<br/>
* `rho`: rho<br/>
* `epsilon`:  epsilon<br/>
* lr: learning rate<br/>

(b) property<br/>
* `.rho (setter) -> float`: rho<br/>
* `.epsilon (setter) -> float`: epsilon<br/>
* `.learning_rate (setter) -> float`: learning rate<br/>

(K) `Adam(target_fun=None, fun_gradient=None, name:str='Adam', max_iter:int= 5e4, print_iter_flag:bool=True, print_every_iter:int=10, save_history_flag:bool=False, dx:float=1e-10, tol:float=1e-5, beta1:float=0.9, beta2:float=0.999, epsilon:float=1e-8, lr:float=0.1)`<br/>
a derived class of GradientDescentOptimizer, using Adam method<br/>

(a) parameter<br/>
see the input of GradientDescentOptimizer<br/>
* `beta1`: beta1<br/>
* `beta2:`  beta2<br/>
* `epsilon`:  epsilon<br/>
* `lr`: learning rate<br/>

(b) property<br/>
* `.beta1 (setter) -> float`: beta1<br/>
* `.beta2 (setter) -> float`: beta2<br/>
* `.epsilon (setter) -> float`: epsilon<br/>
* `.learning_rate (setter) -> float`: learning rate<br/>

(L) `Scipy_LBFGS_B(target_fun=None, fun_gradient=None, name:str='Scipy_LBFGS_B', max_iter:int= 5e4, print_iter_flag:bool=True, print_every_iter:int=1, save_history_flag:bool=False, dx:float=1e-10, ftol:float=1e-11, gtol:float=1e-08, eps:float=1e-08, bnd:tuple=(0,None)`<br/>
a derived class of GradientDescentOptimizer, using scipy.optimize and L-BFGS-B to do the optimization<br/>

(a) parameter<br/>
* `dx`, `ftol`, `gtol`, `eps`: optimizing parameters in [scipy optimze L-BFGS-B](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html)<br/>

(b) property<br/>
* `.dx (setter) -> float`: dx<br/>
* `.ftol (setter) -> float`: ftol<br/>
* `.gtol (setter) -> float`: gtol<br/>
* `.eps (setter) -> float`: eps<br/>


(M) `GeneticAlgorithm (TODO)`<br/>


### Hierarchy
<p align="center">
<img src="./Graph/Hierarchy.png" width="1000">
</p>
<center><strong>FIG. A. Hierarchy.</strong></center>


### example<br/>
in main in GradientDescentOptimizer.py, ScipyOptimizer.py<br/>

## function based<br/>
(A) Utility<br/>
- `LinearFittor(A, yi)` : pseudoinverse<br/>
- `ParameterSavor(fpath, fname, varDict)`
- `ResultsSavor(a_new, count, countList, diffList, a_history, LossList, savefilepath, savefilename)`<br/>
    
(B) Optimization Function<br/>

(a) usage<br/>
`a_new, count, countList, diffList, a_history, LossList = fun(dfda_f_fun, a_init, ..., difffun, MSGCount)`<br/>

- `StochasticGradientDescent(dfda_f_fun, a_init, learning_rate=0.01, tol=1e-5, iter_TOL=5e4, MSGOPENBOOL=True, difffun=diffFun, MSGCount=100)`<br/>
- `Momentum(dfda_f_fun, a_init, v_init=0.0, learning_rate=0.001, beta=0.9, tol=1e-5, iter_TOL=5e4, MSGOPENBOOL=True, difffun=diffFun, MSGCount=100)`<br/>
- `AdaGrad(dfda_f_fun, a_init, epsilon=1e-8, learning_rate=0.5, tol=1e-5, iter_TOL=5e4, MSGOPENBOOL=True, difffun=diffFun, MSGCount=100)`<br/>
- `AdaDelta(dfda_f_fun, a_init, rho=0.9, epsilon=1e-8, tol=1e-5, iter_TOL=5e4, MSGOPENBOOL=True, difffun=diffFun, MSGCount=100)`<br/>
- `RMSprop(dfda_f_fun, a_init, rho=0.9, epsilon=1e-8, learning_rate=0.01, tol=1e-5, iter_TOL=5e4, MSGOPENBOOL=True, difffun=diffFun, MSGCount=100)`<br/>
- `Adam(dfda_f_fun, a_init, beta1=0.9, beta2=0.999, epsilon=1e-8, learning_rate=0.1, tol=1e-5, iter_TOL=5e4, MSGOPENBOOL=True, difffun=diffFun, MSGCount=100)`<br/>
    
(b) parameters
`dfda_f_fun` is a function with input a and the output is (dfda, f), where f is the target function and dfda is the gradient of the target function.<br/>
*** `f` is a single value<br/>
*** `dfda` is a numpy ndarray with the same size of a<br/>
`a_init`   : the initial condition of a<br/>
`...`      : parameters related to the algorithm<br/>
`diffun`   : difference function (default: diffun = lambda a,b: np.max(np.abs(a-b)))<br/>
`MSGCount` : print iteration information in every MSGCount count<br/>

(c) returns<br/>
`a_new`     : the optimized a <br/>
`count`     : total iteration<br/>
`countList` : iteration number<br/>
`diffList`  : difference in each step<br/>
`a_history` : history of a<br/>
`LossList`  : history of loss function<br/>

(C) example<br/>
in Test.py<br/>
