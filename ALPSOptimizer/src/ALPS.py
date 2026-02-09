import numpy as np
import matplotlib.pyplot as plt
import joblib
import subprocess

from sklearn.ensemble import RandomForestRegressor, StackingRegressor, VotingRegressor
from sklearn.cluster import KMeans
from sklearn.multioutput import MultiOutputRegressor
from sklearn_extra.cluster import KMedoids
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import Ridge
from sklearn.gaussian_process.kernels import *
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

import xgboost as xgb
import lightgbm as lgb
import scipy.stats.qmc as qmc
from scipy.optimize import minimize
from scipy.interpolate import splev, splrep
from scipy.stats import norm
from random import randint
from copy import copy

from indago import PSO

import sys
# sys.path.insert(0, 'src')
# sys.path.insert(1, 'models_benchmarks')

# import postprocess as pp
# import benchmarks_functions as bf


class ALPS:
    
    def __init__(self, target, 
                 max_evals, 
                 initial_sample_size,
                 model_batch_size, 
                 lb, ub,
                 experiment, 
                 sample_size=1000, 
                 method='model', 
                 hyperparameters=[],
                 
                 qbc=False,
                 warm_start=False, 
                 warm_start_model=None,
                 ws_target=None,
                 ws_experiment=None,
                 ws_bounds=None):
        
        self.target = target
        self.max_evals = max_evals
        self.initial_sample_size = initial_sample_size
        self.model_batch_size = model_batch_size
        self.warm_start = warm_start
        self.lb = lb
        self.ub = ub
        self.experiment = experiment
        self.sample_size = sample_size
        self.method = method
        self.hyperparameters = hyperparameters
        self.qbc = qbc

        
        self.warm_start = warm_start
        self.warm_start_model = warm_start_model
        self.ws_target = ws_target
        self.ws_experiment = ws_experiment
        self.ws_bounds = ws_bounds
    

    
    def update_model(self, X, y):
        
        # param_dist = {
        #     'n_estimators': randint(50, 200),      
        #     'max_depth': randint(3, 20),           
        #     'min_samples_split': randint(2, 11),  
        #     'min_samples_leaf': randint(1, 11),
        # }
        
        # model = RandomForestRegressor(n_estimators=params['n_estimators'],
        #                               max_depth=params['max_depth'],
        #                               min_samples_split=params['min_samples_split'],
        #                               min_samples_leaf=params['min_samples_leaf'])
        
        if len(self.hyperparameters) == 0: 
            model = RandomForestRegressor()
        else:
            model = RandomForestRegressor(n_estimators=self.hyperparameters[0], max_depth=self.hyperparameters[0])
        # model = RandomForestRegressor(n_estimators=200, max_depth=5)



        model.fit(X,y)
        
        return model       
        
        
    def update_mean_model(self, X, y):
                
        y_mean = np.mean(y, axis=1)  
        y_std = np.std(y, axis=1)
        y_range = np.ptp(y, axis=1)
        y_stack = np.hstack((y_mean.reshape(-1,1), y_std.reshape(-1,1), y_range.reshape(-1,1)))
        
        #model = RandomForestRegressor()
        model = RandomForestRegressor(n_estimators=450, max_depth=10)

        # model.fit(X,y_mean)
        model.fit(X,y_stack) 

        return model

    
    def initialize_warm_start(self):
        
        X = qmc.scale(qmc.LatinHypercube(d=len(self.lb)).random(n=self.sample_size), self.lb, self.ub)
        ws_model = joblib.load(self.warm_start_model)
        
        means = ws_model.predict(X)
        target = self.target
        y_index = np.argsort([np.sqrt(mean_squared_error(target, i)) for i in means])

        X_batch = X[y_index[:self.model_batch_size], :]
        
        
        f_, y_ = self.evaluate(X_batch)


        # X_batch = []
        # f_ = []
                      
        # for s in range(len(sample)):
            
        #     def objective_function(x):
        #         prediction = ws_model.predict([x])
        #         rmse_ = np.sqrt(mean_squared_error(target, prediction[0]))
        #         return rmse_ 
            
        #     dim = len(lb)
        #     min_f = 1
        #     min_x = None
    
        #     bounds = np.array([[self.lb[i], self.ub[i]] for i in range(len(self.lb))])
            
        #     res = minimize(objective_function, x0=sample[s], bounds=bounds, method='L-BFGS-B')
        #     if res.fun < min_f:
        #         min_f = res.fun
        #         min_x = res.x

        #     f_.append(min_f)
        #     X_batch.append(min_x)
            
            
        # ind = np.argsort(f_)[:self.model_batch_size]
        # X_batch = np.array([X_batch[i] for i in ind])
        # f_, y_ = self.evaluate(X_batch)
        
        return f_, X_batch, y_

    def get_qbc_samples(self, X, y):
        
         """experimental"""

         sample = qmc.scale(qmc.LatinHypercube(d=len(self.lb)).random(n=self.model_batch_size*10), self.lb, self.ub)
         
         X_batch = []
         f_ = []

         model = RandomForestRegressor()
         model.fit(X,y)
         predictions = model.predict(X)
         # eps_mean = np.array([np.sqrt(mean_squared_error(self.target, pred)) for pred in predictions])     
     
         # model = RandomForestRegressor(n_estimators=20)
         # model.fit(X, eps_mean)
         
         for s in range(len(sample)):
             
            def uncertainty(x):
   
                predictions = np.array([tree.predict([x]) for tree in model.estimators_])
                std_predictions = np.sum(np.std(predictions,axis=0))
                # std_predictions = np.std(predictions,axis=0)

   
                return -1*std_predictions
                 
            dim = len(self.lb)
            min_f = 1
            min_x = None
   
            bounds = np.array([[self.lb[i], self.ub[i]] for i in range(len(self.lb))])
            res = minimize(uncertainty, x0=sample[s], bounds=bounds, method='L-BFGS-B')
            if res.fun < min_f:
                min_f = res.fun
                min_x = res.x
            # opt = PSO()
            # opt.evaluation_function = uncertainty
            # # opt.params['swarm_size'] = 10
            # opt.max_evaluations = 20
            # opt.lb = self.lb
            # opt.ub = self.ub
            # res = opt.optimize()
            # min_f = res.f
            # min_x = res.X
  
            f_.append(min_f)
            X_batch.append(min_x)
             
         ind = np.argsort(f_)[:self.model_batch_size-1]
         X_batch = np.array([X_batch[i] for i in ind])
         
         
         return X_batch  

     
    def get_optim_samples(self, model):
        
         """experimental"""

         sample = qmc.scale(qmc.LatinHypercube(d=len(self.lb)).random(n=self.model_batch_size*10), self.lb, self.ub)
         
         X_batch = []
         f_ = []
         
         for s in range(len(sample)):
             
             def obj(x):
    
                 prediction = model.predict([x])
                 eps = np.sqrt(mean_squared_error(self.target, prediction[0]))

                 return eps
                 
             opt = PSO()
             opt.evaluation_function = obj
             opt.params['swarm_size'] = 10
             opt.max_evaluations = 40
             opt.lb = self.lb
             opt.ub = self.ub
             res = opt.optimize()
             min_f = res.f
             min_x = res.X
    
             f_.append(min_f)
             X_batch.append(min_x)
             
         ind = np.argsort(f_)[:self.model_batch_size]
         X_batch = np.array([X_batch[i] for i in ind])
         
         
         return X_batch      
 

    def get_model_samples(self, model):
                       
        X = qmc.scale(qmc.LatinHypercube(d=len(self.lb)).random(n=self.sample_size), self.lb, self.ub)
        # X = qmc.scale(qmc.Sobol(d=len(self.lb)).random(n=self.sample_size), self.lb, self.ub)

        means = model.predict(X)
        target = self.target#[::40]
        y_index = np.argsort([np.sqrt(mean_squared_error(target, i)) for i in means])
        # if self.qbc == True:
        #     X_batch = X[y_index[0], :]
        #     X_batch = X_batch.reshape(1,-1)
        # else:
        X_batch = X[y_index[:self.model_batch_size], :]


        return X_batch



    def get_mean_model_samples(self, model):
        
        """experimental"""
        
        X = qmc.scale(qmc.LatinHypercube(d=len(self.lb)).random(n=self.sample_size), self.lb, self.ub)
        means = model.predict(X)
        # target = self.target
        target = np.array([np.mean(self.target), np.std(self.target), np.ptp(self.target)])
        # y_index = np.argsort(np.abs(means - np.mean(target))/np.mean(target))
        y_index = np.argsort([np.sqrt(mean_squared_error(target, i)) for i in means])
        X_batch = X[y_index[:self.model_batch_size], :]
        
        return X_batch

    def get_eps(self, X, mean_model):
        
        predictions = mean_model.predict(X)
        eps_mean = np.array([np.abs(pred - np.mean(target))/np.mean(target) for pred in predictions])

        return eps_mean


    def evaluate(self, samples):
            
        responses = [self.experiment(i) for i in samples]
        f_ = []
        y_ = []
        
        for i in responses:
            a, b = (i)
            f_.append(a)
            y_.append(b)
            
        y_ = np.concatenate(y_)
        
        return f_, y_
    
    def initialize(self):
        
        X = qmc.scale(qmc.LatinHypercube(d=len(self.lb)).random(n=self.initial_sample_size), self.lb, self.ub)       
        f_, y_ = self.evaluate(X)

        return f_, X, y_
  
    def run_parameter_search(self, save_model=False, warm_start=False):
        
        fitness = []
        evals_counter = 0
        
        if warm_start == True:
            
            f_init, X_init, y_init = self.initialize_warm_start() 
        else:
            f_init, X_init, y_init = self.initialize() 
        
        X = X_init
        y = y_init
        
        # print (np.shape(X))
        # print (np.shape(y))

        
        evals_counter += len(X_init)
        
        for i in f_init: fitness.append(i)
        
        
        model = self.update_model(X, y)

        while evals_counter <= self.max_evals - self.initial_sample_size:
            
            #print (evals_counter)
            if self.qbc == True:
                
                X_qbc = self.get_qbc_samples(X, y)    
                evals_counter += len(X_qbc)
                f_qbc, y_qbc = self.evaluate(X_qbc)
                for i in f_qbc: fitness.append(i)
     
                X = np.vstack((X, X_qbc))
                y = np.vstack((y, y_qbc))
                
                model = self.update_model(X, y)
            

                
            if self.method == 'model':
                
                X_model_samples = self.get_model_samples(model)
                #X_model_samples = self.get_mean_model_samples(model)

            
            elif self.method == 'optim':
                
                X_model_samples = self.get_optim_samples(model)

            elif self.method == 'mean':
                
                X_model_samples_ = self.get_model_samples(model)
                X_mean_model_samples = self.get_mean_model_samples(mean_model)
                # print (np.shape(X_model_samples_))
                # print (np.shape(X_mean_model_samples))
                X_model_samples = np.vstack((X_model_samples_, X_mean_model_samples))

            f_model, y_model = self.evaluate(X_model_samples)
            evals_counter += len(X_model_samples)

            for i in f_model: fitness.append(i)


            X = np.vstack((X, X_model_samples))
            y = np.vstack((y, y_model))
            # params = self.get_hyperparameters(X, y)

            model = self.update_model(X, y)
            # mean_model = self.update_mean_model(X, y)




        if save_model == True:
            joblib.dump(model, 'ws_model.pkl')

        indx = np.argmin(fitness)
        while len(fitness) < self.max_evals:
            fitness.append(fitness[-1])
        
        while len(X) < self.max_evals:
            last = X[-1,:]
            X = np.vstack((X, last.reshape(1, -1)))


        return fitness[:self.max_evals], X[:self.max_evals]

    def run(self, n_runs):
        
           sample_set_values = []
           candidates_values = []
           fitness_values = []
           best_solutions = []
           
           orig_target = np.copy(self.target)
           orig_lb = np.copy(self.ws_bounds)
           orig_experiment = copy(self.experiment)
           orig_lb = self.lb
           orig_ub = self.ub
           
           for n in range(n_runs):
               print ('Running optimization:', n)
               if self.warm_start == True:
                   print ('Deleting previous model')
                   subprocess.run([f'rm -rf ./*.pkl'], shell=True)
                   print ('Running to generate model')
                   self.target = self.ws_target
                   self.experiment = self.ws_experiment
                   self.lb = self.ws_bounds[0]
                   self.ub = self.ws_bounds[1]
                   print (self.experiment)
                   _, _ = self.run_parameter_search(save_model=True, warm_start=False)
                   print ('Warm start run')
                   self.target = orig_target
                   self.experiment = orig_experiment
                   print (self.experiment)
                   self.lb = orig_lb
                   self.ub = orig_ub

                   fitness, candidates = self.run_parameter_search(warm_start=self.warm_start)
                   
               else:
                   fitness, candidates = self.run_parameter_search()
                   
               
               candidates_values.append(candidates)
               fitness_values.append(fitness)
           
           
           candidates_values = np.array(candidates_values)
           fitness_values = np.array(fitness_values)
           
           results = { 'solutions': candidates_values,
                       'fitness': fitness_values,
                       #'optimizer':'ALPS-[%s,%s]' % (self.hyperparameters[0], self.hyperparameters[1]),
                       #'optimizer':'ALPS$_{%s-%s}$' % (self.model_batch_size, self.sample_size),
                       'optimizer':'ALPS$_{ws=%s}$' % self.warm_start,
                       'evals': int(self.max_evals),
                       'n_runs': n_runs}
           
           
           # print(fitness_values)
           # print(candidates_values)
           # print(np.shape(fitness_values))
           # print(np.shape(candidates_values))
           
           fitness_value_best = np.argmin(fitness_values[0])
           candidate_value_best = candidates_values[0, fitness_value_best]
           
           # print (candidate_value_best)
           
           return results, candidate_value_best, fitness_values[0, fitness_value_best]

        
        
        
        
        
        
        
        
        
