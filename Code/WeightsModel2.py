"""

...


"""

# Import utils
import numpy as np
import pandas as pd
import copy
import time
import datetime as dt
import pickle
import json
import joblib
from joblib import dump, load



## Import ML tools
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.multiclass import type_of_target
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


class PreProcessing:
    
    """
    
    Description ...
    
    """
        
        
    def __init__(self, *args, **kwargs):
        
        return
    
    

    #### Reshape data to match multi-period rolling horizon
    def reshape_data(self, data, timePeriods, maxTimePeriod, tau):

        """

        Function that creates data slices to match the periods of the rolling horizon. It reshapes the data as consecutive vectors
        whose length equals the length of the rolling horizon and such that consecutive vectors do not overlap.

        Inputs:

            data: data frame to reshape
            timePeriods: series providing the according timePeriod of each row in data
            maxTimePeriod: last timePeriod that is allowed to be included in slices
            tau: look-ahead

        Outputs: 

            sliced_data: iid type reshaped data frame

        """

        ## Get slices of timePeriods
        maxTimePeriod = min(max(timePeriods), maxTimePeriod-tau)

        slices = []
        factor = 0
        step = 0

        while maxTimePeriod - step > 0:
            factor = factor + 1
            slices = slices + [maxTimePeriod - step]
            step = (tau+1) * factor        

        # Apply slices
        sliced_data = data.loc[timePeriods.isin(slices)]

        # Return 
        return sliced_data


    #### Function to scale response and response dependent features
    def scale_variables(self, vars_to_scale, vars_to_scale_with, vars_to_scale_groups, vars_to_scale_with_groups, scaler):

        """

        Function scales a set of variables based on provided data and scaling method.

        Inputs:

            vars_to_scale: np.array of variables that should be scaled with shape (n_samples, n_variables)
            vars_to_scale_with: np.array of variables whose data should be used to fit the scaler with shape (n_samples, n_variables)
            vars_to_scale_groups: grouping of variables to scale
            vars_to_scale_with_groups: grouping of variables to scale with (unqiue groups need to match groups of variables to scale)
            scaler: sklearn scaling method, e.g., MinMaxScaler()

        Outputs:

            vars_scaled: dict of np.arrays of scaled variables for each group in vars_to_scale_groups
            scaler_fitted: dict of scalers fitted on data provided by vars_to_scale_with for each group in vars_to_scale_groups

        """
        scaler_fitted = {}
        vars_scaled = {}

        for group in set(vars_to_scale_groups):

            scaler_fitted[group] = scaler.fit(vars_to_scale_with[vars_to_scale_with_groups==group])

            vars_scaled[group] = scaler_fitted[group].transform(vars_to_scale[vars_to_scale_groups==group])

        return vars_scaled, scaler_fitted


    #### Function to generate training and validation data splits
    def split_timeseries_cv(self, n_splits, timePeriods):

        """

        Function creates n_splits "rolling" cv folds of the timePeriods provided. The output is a list of n_splits folds,
        each of which is a series of True/False indicators for train/validation sets, respectively.
        
        Inputs:
        
            n_splits: number of training/validation splits (folds) to create
            timePeriods: series of the timePeriods to be split
            
        Outputs: 
        
            cv_folds: list of tuples of training and validation folds, each of which is a list of True/False 
                      indicators with length of timePeriods

        """

        # Range of sale yearweeks
        timePeriods_range = np.array(range(int(min(timePeriods)),
                                             int(max(timePeriods))))

        # Initailize rolling time series cross validation splits
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_folds = list()

        # For each fold
        for timePeriods_train_idx, timePeriods_val_idx in tscv.split(range(0,len(timePeriods_range))):

            idx_train = ((timePeriods >= min(timePeriods_range[timePeriods_train_idx])) & 
                         (timePeriods <= max(timePeriods_range[timePeriods_train_idx])))

            idx_val = ((timePeriods >= min(timePeriods_range[timePeriods_val_idx])) & 
                      (timePeriods <= max(timePeriods_range[timePeriods_val_idx])))

            cv_folds.append((idx_train, idx_val))

        # Return list of (train, val) splits
        return cv_folds
    
    
    
    
    
    
    
    
    
    
    
    
#### Random Forest Weights Model
class RandomForestWeightsModel:
    
    """
    
    Description ...
    
    """
        
    #### Initialize
    def __init__(self, **kwargs):
           
        """
    
        Initializes random forest weights model and sets parmeters if provided (meta paramaters 
        for model and tuning via 'model_params' and 'tuning_params', respectively, model 
        hyper parameters via 'hyper_params', and tuning hyper paramater search grid via 
        'hyper_params_grid'.
        
        Inputs:
            
            model_params: dictionary of meta parmaters accepted by sklearn RandomForestRegressor
            hyper_params: dictionary of hyper parmaters accepted by sklearn RandomForestRegressor
            hyper_params_grid: dictionary of lists of of hyper parmaters to try and accepted by sklearn RandomForestRegressor
            tuning_params: disctionary of meta parameters for sklearn RandomSearchCV or GridSearchCV
        



        """

        # Set default model params and update with provided model params
        self.model_params = {

            'oob_score': True,
            'random_state': 12345,
            'n_jobs': 32,
            'verbose': 0

        }

        self.model_params.update(kwargs.get('model_params', {}))

        # Set default hyper params and update with provided hyper params
        self.hyper_params = {

            'n_estimators': 1000,
            'max_depth': None,
            'min_samples_split': 20,
            'min_samples_leaf': 10,
            'max_features': 'auto',
            'max_leaf_nodes': None,
            'min_impurity_decrease': 0.0,
            'bootstrap': True,
            'max_samples': 0.80   

        }

        self.hyper_params.update(kwargs.get('hyper_params', {}))

        # Set default hyper params grid and update with provided hyper params grid
        self.hyper_params_grid = {

            'n_estimators': [1000],
            'max_depth': [None],
            'min_samples_split': [x for x in range(20, 1000, 20)],  
            'min_samples_leaf': [x for x in range(10, 1000, 10)],  
            'max_features': [x for x in range(8, 256, 8)],   
            'max_leaf_nodes': [None],
            'min_impurity_decrease': [0.0],
            'bootstrap': [True],
            'max_samples': [0.70, 0.80, 0.90]

        }    

        self.hyper_params_grid.update(kwargs.get('hyper_params_grid', {}))
        
        
        # Set default tuning params and update with provided tuning params
        self.tuning_params = {
            
            'random_search': True,
            'n_iter': 100,
            'scoring': {'MSE': 'neg_mean_squared_error'},
            'return_train_score': True,
            'refit': 'MSE',
            'random_state': 12345,
            'n_jobs': 8,
            'verbose': 2

        }
        
        self.tuning_params.update(kwargs.get('tuning_params', {}))
        
    
        # Initialize base model 
        self.weightsmodel = RandomForestRegressor(**{**self.model_params, **self.hyper_params})  


    
    
    
    
    
    
    
    #### Function to tune a weights model
    def tune(self, X, y, cv_folds, **kwargs):

        """

        Tunes a weights model on training data X and y, using cross-validation with
        folds provided by input 'cv_folds'
        
        If no hyper parameter search grid (hyper_params_grid) is provided, uses default 
        hyper paramater search grid or previously specified hyper params search grid. 
        The same applies to tuning parameters (tuning_params) and model parameters 
        (model_params).
        
        Inputs:
            
            - self: initialized weights kernel including default or provided parameters
            - X: np.array of feature training data
            - y: np.array of response training data
            - cv_folds: time series splitted np.arrays of train/val index sets for each fold
            - kwargs: further keyword arguments (hyper_params_grid, tuning_params, model_params, ...)
        
        Outputs:
        
            - cv_result: cross-validation results including best hyper parameters


        """
        
        # Update model params if provided
        self.model_params.update(kwargs.get('model_params', {}))
        
        # Update tuning params if provided
        self.tuning_params.update(kwargs.get('tuning_params', {}))
                
        # Update hyper params grid if provided
        self.hyper_params_grid.update(kwargs.get('hyper_params_grid', {}))
                
        # Get meta info
        n_samples = X.shape[0]
        n_features = X.shape[1]
        n_params = np.prod([len(self.hyper_params_grid[h]) for h in self.hyper_params_grid])
        n_params = self.tuning_params['n_iter'] if self.tuning_params['random_search'] else n_params
        tau = y.shape[1]-1 if y.ndim > 1 else 0
        
        # Status
        print('#### Tuning random forest weights model for look-ahead = '+str(tau)+'...')
        
        # Timer start
        start_time = dt.datetime.now().replace(microsecond=0)
        st_exec = time.time()
        st_cpu = time.process_time() 

        # Tuning params: set max features
        self.hyper_params_grid['max_features'] = [
            max_features 
            for max_features 
            in self.hyper_params_grid.get('max_features', [0]) 
            if max_features <= n_features
        ]

        # Base model
        self.weightsmodel.set_params(**self.model_params)  
        
        # Tuning approach
        if self.tuning_params['random_search']:

            # Random search CV
            cv_search = RandomizedSearchCV(estimator=self.weightsmodel,
                                           cv=cv_folds,
                                           param_distributions=self.hyper_params_grid,
                                           n_iter=self.tuning_params['n_iter'],
                                           scoring=self.tuning_params['scoring'],
                                           return_train_score=self.tuning_params['return_train_score'],
                                           refit=self.tuning_params['refit'],
                                           random_state=self.tuning_params['random_state'],
                                           n_jobs=self.tuning_params['n_jobs'],
                                           verbose=self.tuning_params['verbose'])

        else:

            # Grid search SV
            cv_search = GridSearchCV(estimator=self.weightsmodel,
                                     cv=cv_folds,
                                     scoring=self.tuning_params['scoring'],
                                     param_grid=self.hyper_params_grid,
                                     return_train_score=self.tuning_params['return_train_score'],
                                     n_jobs=self.tuning_params['n_jobs'],
                                     refit=self.tuning_params['refit'],
                                     verbose=self.tuning_params['verbose'])

        # Fit the cv search (y with marginal numeric adjustment to avoid fit method to identify as 'multiclass-multioutput')
        cv_search.fit(X, y + 10**(-9)) 

        # Status
        print('... took', dt.datetime.now().replace(microsecond=0) - start_time)      
        
        # Timer end
        exec_time_sec = time.time()-st_exec
        cpu_time_sec = time.process_time()-st_cpu
        
        # Grid search results
        cv_result = {
            'tau': tau,
            'n_samples': n_samples,
            'n_folds': len(cv_folds),
            'n_params:': n_params,
            'exec_time_sec': exec_time_sec,
            'cpu_time_sec': cpu_time_sec,
            'OOB score': abs(cv_search.best_estimator_.oob_score_),
            'Val MSE': abs(cv_search.cv_results_['mean_test_MSE']),
            'Train MSE': abs(cv_search.cv_results_['mean_train_MSE']), 
            'hyper_params': cv_search.best_params_
        
        }   
        
        
        # Store       
        self.cv_search = cv_search
        self.cv_result = cv_result
        
        # Update hyper params with best hyper params
        self.hyper_params.update(cv_search.best_params_)
                
        # Return
        return cv_result

    
    
    
    
    
    
    #### Function to fit weights model to training data
    def fit(self, X, y, **kwargs):
        
        """

        Fits a weights model on training data X and y, using specified hyper params.
        Stores training data X and y in order to apply model for weights generation 
        in predict function.
        
        If no hyper parameters (hyper_params) are provided, uses default hyper parameters
        or previously specified hyper parameters. The same applies to model parameters 
        (model_params).
        
        Inputs:
            
            - self: initialized weights model including default or provided parameters
            - X: np.array of feature training data
            - y: np.array of response training data
            - kwargs: further keyword arguments (hyper_params, model_params, ...)
        
        Outputs:
        
            None

        """
        
        
        # Update model params if provided
        self.model_params.update(kwargs.get('model_params', {}))
                        
        # Update hyper params if provided
        self.hyper_params.update(kwargs.get('hyper_params', {}))
        
        # Set model params and hyper params
        self.weightsmodel.set_params(**dict(self.model_params, **self.hyper_params))
                
        # Fit weights model (y with marginal numeric adjustment to avoid fit method to identify as 'multiclass-multioutput')
        self.weightsmodel.fit(X, y + 10**(-9))
   

        
        
        
    
    
    ## Function to generate sample weights from a fitted weights model given new test features X
    def apply(self, X, x, **kwargs):

        """
        
        Generate N sample weights (each one weight per training sample in X) 
        based on application of the fitted weights model to X and a matrix of new test 
        features x.
        
        Uses model parameters and hyper parameters of fitted weights model. Model parameters
        (model_params) can be updated by providing a new dict to the function.
        
        Inputs:
            
            - X: features of training samples as 2D-array of shape (n training samples, n features)
            - x: features of test samples as 2D-array of shape (m test samples, n features)
            - kwargs: further keyword arguments (model_params, ...)
            
        Attributes:
        
            - weights: 2D-array of shape (m test samples, n training samples)


        """
        
        def get_weights(leaf_nodes_train, leaf_nodes_test):
    
            # Weights vector (average weights over weights per tree)
            w = np.mean((leaf_nodes_train == leaf_nodes_test) / sum(leaf_nodes_train == leaf_nodes_test), axis=1)

            return w


        # Update model params if provided
        self.model_params.update(kwargs.get('model_params', {}))
        
        # Set model params
        self.weightsmodel.set_params(**self.model_params)
        
        # Leaf nodes to which training samples are assigned to
        leaf_nodes_train = self.weightsmodel.apply(X)

        # Leaf nodes to which new test samples are assigned to
        leaf_nodes_test = self.weightsmodel.apply(x)
    
        # For each row in x (i.e., also in leaf_nodes_test) get weights
        weights = np.apply_along_axis(lambda r: get_weights(leaf_nodes_train, r), 1, leaf_nodes_test)
        
        # Save
        self.weights = weights
        
        return weights
    
    
    
     
    
    
    
    
    
    
    
    #### Function to save hyper param tuning results
    def save_cv_result(self, path):
        
        _ = joblib.dump(self.cv_result, path)
        
        
#     #### Function to load hyper param tuning results
#     def load_cv_result(self, path):
        
#         # Load cv results
#         self.cv_result = joblib.load(path)
                
#         # Update hyper params
#         self.hyper_params.update(self.cv_result.get('hyper_params', {}))
        
#         # Set model hyper params
#         self.weightsmodel.set_params(**self.hyper_params)
        
    
    #### Function to load hyper param tuning results
    def load_cv_result(self, path, SKU=None):
        
        if SKU is None:
            
            # Load cv results
            self.cv_result = joblib.load(path)
        
        else:
        
            # Load cv results of SKU
            self.cv_result = joblib.load(path)[SKU]
            
        # Update hyper params
        self.hyper_params.update(self.cv_result.get('hyper_params', {}))

        # Set model hyper params
        self.weightsmodel.set_params(**self.hyper_params)
        
        
    
    
    

    #### Function to save fitted model
    def save_fit(self, path):
        
        _ = joblib.dump(self, path)

        
    #### Function to load fitted model
    def load_fit(self, path):
        
        fitted_model = joblib.load(path)   
        
        self.weightsmodel = fitted_model.weightsmodel
        self.hyper_params = fitted_model.hyper_params
        self.model_params = fitted_model.model_params
        
       
        
        
    #### Function to save weights
    def save_weights(self, path):
        
        _ = joblib.dump(self.weights, path)

        
    #### Function to load weights
    def load_weights(self, path):
        
        weights = joblib.load(path)   
        self.weights = weights

        return weights
        
        
        
        
        
        
    #### Functions to get params
    def get_model_params(self):
        
        return self.model_params
    
    def get_hyper_params(self):
        
        return self.hyper_params
    
    def get_tuning_params(self):
        
        return self.tuning_params
    
    def get_hyper_params_grid(self):
        
        return self.hyper_params_grid
    
    #### Functions to set params
    def set_model_params(self, **kwargs):
    
        self.model_params.update(kwargs)  
    
    def set_hyper_params(self, **kwargs):
    
        self.hyper_params.update(kwargs)  
        
    def set_tuning_params(self, **kwargs):
    
        self.tuning_params.update(kwargs)  
        
    def set_hyper_params_grid(self, **kwargs):
    
        self.hyper_params_grid.update(kwargs)  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    