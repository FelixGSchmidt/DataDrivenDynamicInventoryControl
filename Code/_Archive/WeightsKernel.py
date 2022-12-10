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
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


class PreProcessor:
    
    """
    
    Description ...
    
    """
        
        
    def __init__(self, *args, **kwargs):
        
        return
    
    

    #### Reshaping based on rolling horizon (iid)
    def reshape_data(self, ID_Data, Y_Data, X_Data, T_horizon_rolling, TEST_START, iid = True):

        """

        Function creates training data based on start of test horizon (indicated by the first sale_yearweek
        of the test horizon) and given the rolling horizon reshapes the data as consecutive demand vectors
        whose length equals the length of the rolling horizon. If iid == True then reshapes such that the
        rows of the data are iid in the sense that consecutive demand vectors do not overlap.

        Inputs:

            ID_Data[dataframe]: full data storing identifiers
            Y_Data[dataframe]: full demand data
            X_Data[dataframe]: full feature data
            T_horizon_rolling[int]: length of the rolling horizon
            TEST_START[int]: first sale_yearweek of the test horizon 
            iid[bool]: True if data should be reshaped to iid

        Outputs: 

            ID_train[dataframe]: data storing identifiers of reshaped training data
            Y_train[dataframe]: reshaped demand training data
            X_train[dataframe]: feature training data


        """


        # Combine IDs with Y data, sort by SKU and sale yearweek, and select training data
        Data = pd.concat(objs=[ID_Data,Y_Data], axis=1).sort_values(by=['SKU', 'sale_yearweek'])
        Data = Data.loc[Data.sale_yearweek < TEST_START]

        ## Create look-aheads given rolling horizon

        # For rolling horizon == 1, use current Y
        Y_shifted = pd.DataFrame({'Y'+str(1): Data.Y})

        # For rolling horizon > 1, shift Y upwards
        if T_horizon_rolling != 1:

            # Shift for each priod of the rolling horizon
            for T in range(1,T_horizon_rolling):
                Y_shifted = pd.concat([Y_shifted, pd.DataFrame({'Y'+str(T+1): Data.groupby('SKU').Y.shift(-T)})], axis=1)

        # Date frame of shifted Y
        Y_shifted = pd.concat([Data[['SKU', 'sale_year', 'sale_week', 'sale_yearweek']], Y_shifted], axis=1).dropna()

        ## Merge to ID Data and X_Data
        Data = pd.concat(
            objs=[ID_Data.merge(
                Y_shifted, how='left', on=['SKU', 'sale_year', 'sale_week', 'sale_yearweek']), 
                  X_Data], 
            axis=1)

        # Reduce to training data
        Data = Data.loc[Data.sale_yearweek < TEST_START].dropna()

        ## Reshape to iid
        if iid:

            ## Get slices of sale yearweeks to ensure iid data
            max_sale_yearweek = min(max(Y_shifted.sale_yearweek),TEST_START-T_horizon_rolling)

            slices = []
            factor=0
            step=0

            # Get every T_horizon_rolling'th sale_yearweek starting from the last sale_yearweek
            while max_sale_yearweek - step > 0:
                factor = factor + 1
                slices = slices + [max_sale_yearweek - step]
                step = T_horizon_rolling * factor        

            # Apply slices
            Data = Data.loc[Data.sale_yearweek.isin(slices)]

        # Create final training data
        ID_train = Data[['SKU', 'sale_year', 'sale_week', 'sale_yearweek']]
        Y_train = Data[['Y'+str(t+1) for t in range(T_horizon_rolling)]]
        X_train = Data[X_Data.columns]

        # Return 
        return ID_train, Y_train, X_train
    
    
    #### Function to scale response and response dependent features

    """

    ToDo: function to scale training and test data based on scaler fitted on training data

    """

    #### Function to generate training and validation data splits
    def split_timeseries_cv(self, n_splits, timePeriods):

        """

        Function creates n_splits "rolling" cv folds of the timePeriods provided. The output is a list of n_splits folds,
        each of which is a series of True/False indicators for train/validation sets, respectively.


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
    
    
    
    
    
    
    
    
    
    
    
    
    
class RandomForestWeightsKernel:
    
    """
    
    Description ...
    
    """
        
    #### Initialize
    def __init__(self, **kwargs):
           
        """
    
        Initializes random forest weights kernel and sets parmeters if provided (meta paramaters 
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
        self.weightskernel = RandomForestRegressor(**{**self.model_params, **self.hyper_params})  


    
    
    
    
    
    
    
    #### Function to tune a weights kernel
    def tune(self, X, y, cv_folds, **kwargs):

        """

        Tunes a weights kernel on training data X and y, using cross-validation with
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
        n_features = X.shape[1]
        tau = y.shape[1] if y.ndim > 1 else 1

        # Print status
        start_time = dt.datetime.now().replace(microsecond=0)
        print('## Tuning random forest weights kernel for rolling horizon =', str(tau))
        
        # Tuning params: set max features
        self.hyper_params_grid['max_features'] = [
            max_features 
            for max_features 
            in self.hyper_params_grid.get('max_features', [0]) 
            if max_features <= n_features
        ]

        # Base model
        self.weightskernel.set_params(**self.model_params)  
        
        # Tuning approach
        if self.tuning_params['random_search']:

            # Random search CV
            cv_search = RandomizedSearchCV(estimator=self.weightskernel,
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
            cv_search = GridSearchCV(estimator=self.weightskernel,
                                     cv=cv_folds,
                                     scoring=self.tuning_params['scoring'],
                                     param_grid=self.hyper_params_grid,
                                     return_train_score=self.tuning_params['return_train_score'],
                                     n_jobs=self.tuning_params['n_jobs'],
                                     refit=self.tuning_params['refit'],
                                     verbose=self.tuning_params['verbose'])

        # Fit the cv search (note the irrelevant small numeric adjustment to avoid "int error")
        cv_search.fit(X, y + 10**(-12)) 

        # Grid search results
        print('## CV took', dt.datetime.now().replace(microsecond=0) - start_time)      

        cv_result = {
            'T_horizon_rolling': tau,
            'n_features': n_features,
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

    
    
    
    
    
    
    #### Function to fit weights kernel to training data
    def fit(self, X, y, **kwargs):
        
        """

        Fits a weights kernel on training data X and y, using specified hyper params.
        Stores training data X and y in order to apply model for weights generation 
        in predict function.
        
        If no hyper parameters (hyper_params) are provided, uses default hyper parameters
        or previously specified hyper parameters. The same applies to model parameters 
        (model_params).
        
        Inputs:
            
            - self: initialized weights kernel including default or provided parameters
            - X: np.array of feature training data
            - y: np.array of response training data
            - kwargs: further keyword arguments (hyper_params, model_params, ...)
        
        Outputs:
        
            None

        """
        
        
        # Update model params if provided
        self.model_params.update(kwargs.get('model_params', {}))
                        
        # Update hyper params grid if provided
        self.hyper_params_grid.update(kwargs.get('hyper_params_grid', {}))
        
        # Set model params and hyper params
        self.weightskernel.set_params(**dict(self.model_params, **self.hyper_params))
                
        # Fit weights kernel (y with small numeric adjustment to avoid integers)
        self.weightskernel.fit(X, y + 10**(-12))
        
        # Store training data
        self.X = X
        self.y = y
        
        
        
        
    
#     ## Function to generate sample weights from a fitted RF regressor given a new test feature x
#     def predict(self, x, **kwargs):

#         """
        
#         Generate N sample weights (each one weight per training sample in X_train / Y_train) 
#         based on application of the fitted RF model to X_train and a new test observation x.
        
#         Uses model parameters and hyper parameters of fitted weights kernel. Model parameters
#         (model_params) can be updated by providing a new dict to the function.
        
#         Inputs:
            
#             - x: new test feature vector
#             - kwargs: further keyword arguments (model_params, ...)
            
#         Attributes:
        
#             - weights: 1-d array of weights (shape is equal to shape of Y_train)


#         """
#         # Update model params if provided
#         self.model_params.update(kwargs.get('model_params', {}))
        
#         # Set model params
#         self.weightskernel.set_params(**self.model_params)
        
#         # Leaf nodes to which training samples are assigned to
#         leaf_nodes_train = self.weightskernel.apply(self.X)

#         # Leaf nodes to which new test sample is assigned to
#         leaf_nodes_test = self.weightskernel.apply(x)

#         # Weights vector (average weights over weights per tree)
#         weights = np.mean((leaf_nodes_train == leaf_nodes_test) * 1 / sum(leaf_nodes_train == leaf_nodes_test), 1)
        
#         return weights
    
    
    
    
     ## Function to generate sample weights from a fitted weights kernel given new test features X
    def apply(self, X, **kwargs):

        """
        
        Generate N sample weights (each one weight per training sample in X_train / Y_train) 
        based on application of the fitted weights kernel to X_train and a matrix of new test 
        features X.
        
        Uses model parameters and hyper parameters of fitted weights kernel. Model parameters
        (model_params) can be updated by providing a new dict to the function.
        
        Inputs:
            
            - X: new test features as 2D-array of shape (n test samples, n features)
            - kwargs: further keyword arguments (model_params, ...)
            
        Attributes:
        
            - weights: 2D-array of shape (n test samples, n train samples)


        """
        
        def get_weights(leaf_nodes_train, leaf_nodes_test):
    
            # Weights vector (average weights over weights per tree)
            w = np.mean((leaf_nodes_train == leaf_nodes_test) / sum(leaf_nodes_train == leaf_nodes_test), axis=1)

            return w


        # Update model params if provided
        self.model_params.update(kwargs.get('model_params', {}))
        
        # Set model params
        self.weightskernel.set_params(**self.model_params)
        
        # Leaf nodes to which training samples are assigned to
        leaf_nodes_train = self.weightskernel.apply(self.X)

        # Leaf nodes to which new test sample is assigned to
        leaf_nodes_test = self.weightskernel.apply(X)
    
        # For each row in X (i.e., also in leaf_nodes_test) get weights
        weights = np.apply_along_axis(lambda r: get_weights(leaf_nodes_train, r), axis=1, leaf_nodes_test)
        
        # Save
        self.weights = weights
        
        return weights
    
    
    
     
    
    
    
    
    
    
    
    #### Function to save hyper param tuning results
    def save_cv_result(self, path):
        
        _ = joblib.dump(self.cv_result, path)
        
        
        
    #### Function to load hyper param tuning results
    def load_cv_result(self, path):
        
        # Load cv results
        self.cv_result = joblib.load(path)
                
        # Update hyper params
        self.hyper_params.update(self.cv_result.get('hyper_params', {}))
        
        # Set model hyper params
        self.weightskernel.set_params(**self.hyper_params)
        
        

    #### Function to save fitted model
    def save_fit(self, path):
        
        _ = joblib.dump(self, path)

        
    #### Function to load fitted model
    def load_fit(self, path):
        
        fitted_model = joblib.load(path)   
        
        self.weightskernel = fitted_model.weightskernel
        self.X = fitted_model.X
        self.y = fitted_model.y
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    