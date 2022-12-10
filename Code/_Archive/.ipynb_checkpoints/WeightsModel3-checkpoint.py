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
        
        """
        
        Inputs:
        
            args: ignored
            kwargs: ignored
            
        Outputs:
        
            PreProcessing: initialized PreProcessing object
           
        """
        
        return
    
    
    #### Function to scale response and response dependent features
    def scale_variables(self, vars_to_scale, vars_to_scale_with, vars_to_scale_groups, 
                        vars_to_scale_with_groups, scaler=MinMaxScaler(), **kwargs):

        """

        Function scales a set of variables based on provided data and scaling method.

        Inputs:

            vars_to_scale: np.array of variables that should be scaled with shape (n_samples, n_variables)
            vars_to_scale_with: np.array of variables whose data should be used to fit the scaler with shape (n_samples, n_variables)
            vars_to_scale_groups: grouping of variables to scale
            vars_to_scale_with_groups: grouping of variables to scale with (unqiue groups need to match groups of variables to scale)
            scaler: sklearn scaling method, e.g., MinMaxScaler()
            kwargs: ignored

        Outputs:

            vars_scaled: dict of np.arrays of scaled variables for each group in vars_to_scale_groups
            scaler_fitted: dict of scalers fitted on data provided by vars_to_scale_with for each group in vars_to_scale_groups

        """
        
        scaler_fitted = {}
        vars_scaled = {}

        for group in set(vars_to_scale_groups):

            scaler_fitted[group] = copy.deepcopy(scaler.fit(vars_to_scale_with[vars_to_scale_with_groups==group]))

            vars_scaled[group] = scaler_fitted[group].transform(vars_to_scale[vars_to_scale_groups==group])

        return vars_scaled, scaler_fitted
    
    
    
    
    
    
    

    #### Function to split and reshape data frames into train and test sets
    def train_test_split(self, data, train, test=None, rolling_horizon=None, to_array=False, **kwargs):


        """

        ...

        Arguments:
            data: ...
            train: ...
            test: ...
            rolling_horizon: ...
            to_array: False ...
            kwargs: passed to reshape_data(), performed on train data if required kwargs are provided (timePeriods, maxTimePeriod, tau)

        Outputs:

            data_train, data_test: if test is provided, touple of train and test data, else only data_train

        """

        ## Training data

        # Select columns (samples) and rows (rolling look-ahead horizon periods)
        data_train = data.loc[train].iloc[:,rolling_horizon] if not rolling_horizon is None else data.loc[train]

        # Reshape training data to match (tau+1)-periods rolling horizon
        if 'timePeriods' in kwargs and 'maxTimePeriod' in kwargs and 'tau' in kwargs:

            data_train = self.reshape_data(data_train, **kwargs)

        # Tansfrom data to arrays
        if to_array:
            
            data_train = np.array(data_train)
            
            if np.array(data_train).ndim > 1:
                if np.array(data_train).shape[1] == 1:
                    data_train = data_train.flatten()

        ## Test data
        if test is None:

            return data_train

        else:

            # Select columns (samples) and rows (rolling look-ahead horizon periods)
            data_test = data.loc[test].iloc[:,rolling_horizon] if not rolling_horizon is None else data.loc[test]

            # Tansfrom data to arrays
            if to_array:
            
                data_test = np.array(data_test)
            
                if np.array(data_test).ndim > 1:
                    if np.array(data_test).shape[1] == 1:
                        data_test = data_test.flatten()

            return data_train, data_test



    
    
    #### Function to split data frames into train (and test sets)
    def train_test_split2(self, data, train, test=None, rolling_horizon=None, to_array=False, **kwargs):


        """

        ...

        Arguments:
            data: ...
            train: ...
            test: ...
            rolling_horizon: ...
            to_array: False ...
            kwargs: ignored

        Outputs:

            data_train, (data_test): if test is provided, touple of train and test data, else only data_train

        """

        ## Training data

        # Select columns (samples) and rows (rolling look-ahead horizon periods)
        data_train = data.loc[train].iloc[:,rolling_horizon] if not rolling_horizon is None else data.loc[train]

        # Tansfrom data to arrays
        if to_array:
            
            data_train = np.array(data_train)
            
            if np.array(data_train).ndim > 1:
                if np.array(data_train).shape[1] == 1:
                    data_train = data_train.flatten()

        ## Test data
        if test is None:

            return data_train

        else:

            # Select columns (samples) and rows (rolling look-ahead horizon periods)
            data_test = data.loc[test].iloc[:,rolling_horizon] if not rolling_horizon is None else data.loc[test]

            # Tansfrom data to arrays
            if to_array:
            
                data_test = np.array(data_test)
            
                if np.array(data_test).ndim > 1:
                    if np.array(data_test).shape[1] == 1:
                        data_test = data_test.flatten()

            return data_train, data_test



    
    
    
    

    #### Reshape data to match multi-period rolling horizon
    def reshape_data(self, data, timePeriods, maxTimePeriod, tau, **kwargs):

        """

        Function that creates data slices to match the periods of the rolling horizon. It reshapes the data as consecutive vectors
        whose length equals the length of the rolling horizon and such that consecutive vectors do not overlap.

        Arguments:

            data: data frame to reshape
            timePeriods: series providing the according timePeriod of each row in data
            maxTimePeriod: last timePeriod that is allowed to be included in slices
            tau: look-ahead
            kwargs: ignored

        Outputs: 

            sliced_data: data with n_samples rows whose periods match (tau+1)-periods rolling horizon

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


    


    #### Function to generate training and validation data splits
    def split_timeseries_cv(self, n_splits, timePeriods, **kwargs):

        """

        Function creates n_splits "rolling" cv folds of the timePeriods provided. The output is a list of n_splits folds,
        each of which is a series of True/False indicators for train/validation sets, respectively.
        
        Arguments:
        
            n_splits: number of training/validation splits (folds) to create
            timePeriods: series of the timePeriods to be split
            kwargs: ignored
            
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
    def __init__(self, model_params=None, hyper_params=None, **kwargs):
           
        """
    
        Initializes random forest weights model and sets parmeters if provided (model meta paramaters 'model_params' and
        model hyper parameters 'hyper_params' if provided.
        
        Arguments:
            
            model_params: dictionary of model meta parmaters accepted by sklearn RandomForestRegressor
            hyper_params: dictionary of model hyper parmaters accepted by sklearn RandomForestRegressor
            kwargs: ignored
            
        Outputs:
        
            RandomForestWeightsModel: initialized RandomForestWeightsModel object

        """
        
        # Initialize pre-processing module
        self.pp = PreProcessing()
        
        # Set default model params
        self.model_params = {

            'oob_score': True,
            'random_state': 12345,
            'n_jobs': 32,
            'verbose': 0

        }
        
        # Update with provided model params
        if not model_params is None:
            self.model_params.update(model_params)

        # Set default hyper params
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

        # Update with provided hyper params
        if not hyper_params is None:
            self.hyper_params.update(hyper_params)

    
        # Initialize base model 
        self.weightsmodel = RandomForestRegressor(**{**self.model_params, **self.hyper_params})  


    
    
    
    
    
    
    
    #### Function to tune a weights model
    def tune(self, X, y, cv_folds, hyper_params_grid=None, tuning_params=None, random_search=True, print_status=False, **kwargs):

        """

        Tunes a weights model on training data X and y, using cross-validation with
        folds provided by input 'cv_folds'
        
        If no hyper parameter search grid (hyper_params_grid) is provided, uses default 
        hyper paramater search grid or previously specified hyper params search grid. 
        The same applies to tuning parameters (tuning_params) and model parameters 
        (model_params).
        
        Arguments:
            
            X: np.array of feature training data
            y: np.array of response training data
            cv_folds: time series splitted np.arrays of train/val index sets for each fold
            random_search: True to perfrom sklearn's randomized CV search
            print_status: True to print status and time update
            hyper_params_grid: dict of lists of hyper parameters to try 
            tuning_params: dict of meta parameters that control tuning (e.g., number of iterations for random search)
            kwargs: further keyword arguments that are passed to RandomForest (model_params)
            
        Outputs:
        
            cv_result: cross-validation results including best hyper parameters


        """
        
        
        # Set default hyper params grid
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

        # Update with provided hyper params grid
        if not hyper_params_grid is None:
            self.hyper_params_grid.update(hyper_params_grid)
        
        
        # Set default tuning params
        self.tuning_params = {
            
            'n_iter': 100,
            'scoring': {'MSE': 'neg_mean_squared_error'},
            'return_train_score': True,
            'refit': 'MSE',
            'random_state': 12345,
            'n_jobs': 8,
            'verbose': 2

        }
        

            
        # Update with provided tuning params
        if not tuning_params is None:
            self.tuning_params.update(tuning_params)

        # Update model meta params if provided
        self.model_params.update(kwargs.get('model_params', {}))
   
        # Get meta info
        n_samples = X.shape[0]
        n_features = X.shape[1]
        n_params = np.prod([len(self.hyper_params_grid[h]) for h in self.hyper_params_grid])
        n_params = self.tuning_params['n_iter'] if random_search else n_params
        tau = y.shape[1]-1 if y.ndim > 1 else 0
        
        # Status
        if print_status==True: 
            print('#### Tuning random forest weights model for look-ahead = '+str(tau)+'...')
        
        # Timer
        start_time = dt.datetime.now().replace(microsecond=0)
        st_exec = time.time()
        st_cpu = time.process_time() 

        # Hyper params grid: ensure max features is not too large
        self.hyper_params_grid['max_features'] = [
            max_features 
            for max_features 
            in self.hyper_params_grid.get('max_features', [0]) 
            if max_features <= n_features
        ]

        # Base model with latest model meta params
        self.weightsmodel.set_params(**self.model_params)  
        
        # Tuning approach
        if random_search:

            # Random search CV
            cv_search = RandomizedSearchCV(estimator=self.weightsmodel,
                                           cv=cv_folds,
                                           param_distributions=self.hyper_params_grid,
                                           **self.tuning_params)

        else:

            # Grid search SV
            cv_search = GridSearchCV(estimator=self.weightsmodel,
                                     cv=cv_folds,
                                     param_grid=self.hyper_params_grid,
                                     **self.tuning_params)

        # Fit the cv search (y with marginal numeric adjustment to avoid fit method to identify as 'multiclass-multioutput')
        cv_search.fit(X, y + 10**(-9)) 

        # Status
        if print_status: 
            print('... took', dt.datetime.now().replace(microsecond=0) - start_time)      
        
        # Timer
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
        
        Arguments:
            
            X: np.array of feature training data
            y: np.array of response training data
            kwargs: further keyword arguments passed to RandomForestRegressor (model_params, hyper_params)
            
        Outputs:
        
            RandomForestWeightsModel: fitted RandomForestWeightsModel

        """
        
        
        # Update model params if provided
        self.model_params.update(kwargs.get('model_params', {}))
                        
        # Update hyper params if provided
        self.hyper_params.update(kwargs.get('hyper_params', {}))
        
        # Set model params and hyper params
        self.weightsmodel.set_params(**{**self.model_params, **self.hyper_params})
                
        # Fit weights model (y with marginal numeric adjustment to avoid fit method to identify as 'multiclass-multioutput')
        self.weightsmodel.fit(X, y + 10**(-9))
        
        return self
   


    def get_weights(self, leaf_nodes_train, leaf_nodes_test, **kwargs):
        
        """

        Calculates n weights based on a comparison of the leaf nodes of the n training samples and
        the leaf nodes of a test sample
        
        Arguments:
            
            leaf_nodes_train: np.array of leaf nodes over training samples for each estimator in RandomForestRegressor
            leaf_nodes_test: np.array of leaf nodes over test sample for each estimator in RandomForestRegressor
            kwargs: ignored
            
        Outputs:
        
            weights: vector of n weights (i.e., a weight for each training sample)

        """

        # Weights vector (average weights over weights per tree)
        weights = np.mean((leaf_nodes_train == leaf_nodes_test) / sum(leaf_nodes_train == leaf_nodes_test), axis=1)

        return weights
        
        
    ## Function to generate sample weights from a fitted weights model given new test features X
    def apply(self, X, x, **kwargs):

        """
        
        Generate N sample weights (each one weight per training sample in X) 
        based on application of the fitted weights model to X and a matrix of new test 
        features x.
        
        Uses model parameters and hyper parameters of fitted weights model. Model meta parameters
        (model_params) can be updated by providing a new dict to the function.
        
        Arguments:
            
            X: features of training samples as 2D-array of shape (n training samples, n features)
            x: features of test samples as 2D-array of shape (m test samples, n features)
            kwargs: further keyword arguments passed to RandomForestRegressor (model_params)
            
        Outputs:
        
            weights: 2D-array of shape (m test samples, n training samples)


        """


        # Update model params if provided
        self.model_params.update(kwargs.get('model_params', {}))
        
        # Set model params
        self.weightsmodel.set_params(**self.model_params)
        
        # Leaf nodes to which training samples are assigned to
        leaf_nodes_train = self.weightsmodel.apply(X)

        # Leaf nodes to which new test samples are assigned to
        leaf_nodes_test = self.weightsmodel.apply(x)
    
        # For each row in x (i.e., also in leaf_nodes_test) get weights
        weights = np.apply_along_axis(lambda r: self.get_weights(leaf_nodes_train, r), 1, leaf_nodes_test)
                
        return weights
    
    
    
    
    

    
    
    #### Function to save hyper param tuning results
    def save_cv_result(self, path):
        
        _ = joblib.dump(self.cv_result, path)

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
        
        return self
    

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
        
        
        
        
       
    
    ### Function to generate samples, fit weight functions, and generate weights
    def training_and_sampling(self, ID_Data, X_Data, Y_Data, tau, timePeriods, timePeriodsTestStart, **kwargs):

        """
        ...
        
        """           
        # Select and reshape training and test data
        args = {'train': (timePeriods < timePeriodsTestStart), 'test': (timePeriods == timePeriodsTestStart), 
                'timePeriods': timePeriods[(timePeriods < timePeriodsTestStart)], 'maxTimePeriod': timePeriodsTestStart-1, 'tau': tau}

        id_train, id_test = self.pp.train_test_split(ID_Data, **args)
        X_train, X_test = self.pp.train_test_split(X_Data, **args, to_array=True)
        y_train, y_test = self.pp.train_test_split(Y_Data, **args, rolling_horizon=[l for l in range(0,tau+1)], to_array=True)

        # Store samples of historical demands
        samples = {'y_train': y_train, 'y_test': y_test, 
                   'X_train': X_train, 'X_test': X_test, 
                   'id_train': id_train, 'id_test': id_test}

        # Fit weight function  
        st_exec, st_cpu = time.time(), time.process_time() 
        weightfunction = self.fit(X_train, y_train, **kwargs)
        weightfunction_times = {'exec_time_sec': time.time()-st_exec, 'cpu_time_sec': time.process_time()-st_cpu}

        # Generate weights  
        st_exec, st_cpu = time.time(), time.process_time() 
        weights = self.apply(X_train, X_test, **kwargs)
        weights_times = {'exec_time_sec': time.time()-st_exec, 'cpu_time_sec': time.process_time()-st_cpu}

        return samples, weightfunction, weightfunction_times, weights, weights_times
    
    
    
    
    ### Function to generate samples, fit weight functions, and generate weights
    def training_and_sampling2(self, ID_Data, X_Data, Y_Data, tau, timePeriods, timePeriodsTestStart, **kwargs):

        """
        ...
        
        """           
        # Select and reshape training and test data
        args = {'train': (timePeriods < timePeriodsTestStart - tau), 'test': (timePeriods == timePeriodsTestStart)}

        id_train, id_test = self.pp.train_test_split2(ID_Data, **args)
        X_train, X_test = self.pp.train_test_split2(X_Data, **args, to_array=True)
        y_train, y_test = self.pp.train_test_split2(Y_Data, **args, rolling_horizon=[l for l in range(0,tau+1)], to_array=True)

        # Store samples of historical demands
        samples = {'y_train': y_train, 'y_test': y_test, 
                   'X_train': X_train, 'X_test': X_test, 
                   'id_train': id_train, 'id_test': id_test}

        # Fit weight function  
        st_exec, st_cpu = time.time(), time.process_time() 
        weightfunction = self.fit(X_train, y_train, **kwargs)
        weightfunction_times = {'exec_time_sec': time.time()-st_exec, 'cpu_time_sec': time.process_time()-st_cpu}

        # Generate weights  
        st_exec, st_cpu = time.time(), time.process_time() 
        weights = self.apply(X_train, X_test, **kwargs)
        weights_times = {'exec_time_sec': time.time()-st_exec, 'cpu_time_sec': time.process_time()-st_cpu}

        return samples, weightfunction, weightfunction_times, weights, weights_times
    
    
    
    