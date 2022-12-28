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



#### Feature scaler
class MaxQFeatureScaler:
    
    """
    ...
    
    """
    
    ## Initialize
    def __init__(self, q_outlier=0.999, lags=[1,2,3,4,5], **kwargs):
        
        """
        ...
        
        """
        
        # Quantile up to which data should be considered for fitting the scaler
        self.q_outlier = q_outlier
        
        # Lags to consider for features used for scaling
        self.lags = lags
        
        # Scaling values
        self.scaling_value = None
        
        return None
    
    ## Fit scaler to data
    def fit(self, X, features=None, features_to_scale=None, features_to_scale_with=None, **kwargs):
        
        """
        
        Fits feature scaler to (training data).
        
        Arguments:
        
            X: np.array feature matrix, shape (n_samples, n_features)
            features: np.array feature names, length n_features
            features_to_scale: np.array names of features that should be scaled, length <= n_features 
            features_to_scale_with: np.array names of features that should be used for sscaling, length <= n_features
            kwargs: can provide q_outlier and lags to overwrite defaults
            
        Returns:
        
            self: fitted scaler
        
        """
        
        # Update parameters if provided
        self.q_outlier = kwargs.get('q_outlier', self.q_outlier)
        self.lags = kwargs.get('lags', self.lags)
        
        # Store features and features_to_scale_with
        self.features = features
        self.features_to_scale_with = features_to_scale_with

        # Check
        if not len(features_to_scale) == len(features_to_scale_with):
            raise ValueError('features_to_scale and features_to_scale_with need to be of same length.')
        
        # Initialize scaling values
        scaling_values = []

        # If features, features_to_scale, and features_to_scale_with are provided
        if not ((features is None) or (features_to_scale is None) or (features_to_scale_with is None)):
            
            # Initialize scaling values of features used for scaling
            qMaxVals = {}
            qMaxVals_ = {}
        
            # For each feature used for scaling, get the scaling value (max of q-quantile per lag of the feature)
            for feature_to_scale_with in np.unique(features_to_scale_with):  
               
                # If lags are provided
                if not self.lags is None:
                    
                    # Initialize scaling value per lag
                    z = []
                    
                    # For each lag of the feature
                    for l in self.lags:
                        
                        # Select th feature to be used for scaling
                        if feature_to_scale_with+'_lag'+str(l) in features:
                            
                            # Select the lag feature to be used for scaling
                            sel = (features == feature_to_scale_with+'_lag'+str(l))
                            
                            # Find the q-Quantile of the lag feature
                            z_ = np.quantile(X[:,sel], self.q_outlier, method='closest_observation')
                            
                            # Use z_ if greater than zero
                            if z_ > 0:
                                
                                z += [z_]
                            
                            # Else, use max of the feature unless
                            else:
                                
                                z += [max(X[:,sel].flatten())]
                                
                    # Use max of q-Quantiles of lag features as scaling value (default to 1 of not greater than zero)
                    qMaxVals_[feature_to_scale_with] = max(z) if max(z) > 0 else 1
                
                # If no lags are provided
                else:
                    
                    # Select the feature to be used for scaling
                    sel = (features == feature_to_scale_with)
                    
                    # Find the q-Quantile of the feature
                    z_ = np.quantile(X[:,sel], self.q_outlier, method='closest_observation')
                    
                    # Use z_ if greater than zero
                    if z_ > 0:
                        
                        qMaxVals_[feature_to_scale_with] = z_
                        
                    # Else, use max of the feature unless also max is not greater zero, then default to 1
                    else:
                        
                        qMaxVals_[feature_to_scale_with] = max(X[:,sel].flatten()) if max(X[:,sel].flatten()) > 0 else 1
                        
            # Map scaling values to all features to be scaled (as some my be redundant, see "unique" call above)
            for feature_to_scale, feature_to_scale_with in zip(features_to_scale, features_to_scale_with):
                qMaxVals[feature_to_scale] = copy.deepcopy(qMaxVals_[feature_to_scale_with])

            # Map scaling values to all features
            for feature in features:
                
                # If feature is part of features to be scaled
                if feature in features_to_scale:
                    
                    scaling_values += [qMaxVals[features_to_scale[features_to_scale==feature].item()]]
                
                # Else fefault to 1
                else:
                    
                    scaling_values += [1.0]
              
                
        # Else, use all columns in X for scaling
        else:
            
            # For each column in X, get the scaling value (max of q-quantile)
            for col in range(X.shape[1]):
     
                # Find the q-Quantile of the feature
                z_ = np.quantile(X[:,col], self.q_outlier, method='closest_observation')
            
                # Use z_ if greater than zero
                if z_ > 0:
                    
                    scaling_values += [z_]
                    
                # Else, use max of the feature unless also max is not greater zero, then default to 1
                else:
                    
                    scaling_values += [max(X[:,col].flatten()) if max(X[:,col].flatten()) > 0 else 1]
                    
        # Store fitted scaling values
        self.scaling_values = np.asarray(scaling_values)       
   
        return self
    
    
    ## Transform data
    def transform(self, X, **kwargs):
        
        # Check if fitted
        if self.scaling_values is None:
            raise ValueError('Scaling_values are None. Maybe the scaler is not et fitted?')
            
        # Check if shapes match
        if not X.shape[1] == self.scaling_values.shape[0]:
            raise ValueError('Shapes of scaling_values and X do not match.')
        
        # Tansform
        X_transformed = X / self.scaling_values
        
        return X_transformed   
    
    
    ## Inverse transform data
    def inverse_transform(self, X, **kwargs):
        
        # Check if fitted
        if self.scaling_values is None:
            raise ValueError('Scaling_values are None. Maybe the scaler is not et fitted?')
            
        # Check if shapes match
        if not X.shape[1] == self.scaling_values.shape[0]:
            raise ValueError('Shapes of scaling_values and X do not match.')
        
        # Inverse transform
        X_inverse_transformed = X * self.scaling_values
        
        return X_inverse_transformed   
    
    
    
    
    
#### Demand scaler
class MaxQDemandScaler:
    
    """
    ...
    
    """
    
    ## Initialize
    def __init__(self, q_outlier=0.999, tau=0, **kwargs):
        
        """
        ...
        
        """
        
        # Quantile up to which data should be considered for fitting the scaler
        self.q_outlier = q_outlier
        
        # Look-ahead tau to be used
        self.tau = tau
        
        # Scaling values
        self.scaling_value = None
        
        return None
    
    ## Fit scaler to data
    def fit(self, y, **kwargs):
        
        """
        
        Fits feature scaler to (training data).
        
        Arguments:
        
            y: np.array demands, shape (n_samples, 1) or shape (n_samples, )
            kwargs: can provide q_outlier and tau to overwrite defaults
            
        Returns:
        
            self: fitted scaler
        
        """
        
        # Update parameters if provided
        self.q_outlier = kwargs.get('q_outlier', self.q_outlier)
        self.tau = kwargs.get('tau', self.tau)
        
        # Prep
        y = y.flatten()

        # Initialize scaling values
        scaling_values = []

        # Find the q-Quantile
        z_ = np.quantile(y, self.q_outlier, method='closest_observation')

        # Use z_ if greater than zero
        if not z_ > 0:
            
            z_ = max(y) if max(y) > 0 else 1

        # Map scaling values to all periods of the look-ahead
        for tau_ in range(self.tau+1):
            scaling_values += [copy.deepcopy(z_)]

        # Store fitted scaling values
        self.scaling_values = np.array(scaling_values)
   
        return self
    
    
    ## Transform data
    def transform(self, y, **kwargs):
        
        # Check if fitted
        if self.scaling_values is None:
            raise ValueError('Scaling_values are None. Maybe the scaler is not et fitted?')
            
        # Check if shapes match
        if not (y.shape[1] if y.ndim > 1 else 1) == self.scaling_values.shape[0]:
            raise ValueError('Shapes of scaling_values and y do not match.')
        
        # Tansform
        y_transformed = y / self.scaling_values
        
        return y_transformed   
    
    
    ## Inverse transform data
    def inverse_transform(self, y, **kwargs):
        
        # Check if fitted
        if self.scaling_values is None:
            raise ValueError('Scaling_values are None. Maybe the scaler is not et fitted?')
            
        # Check if shapes match
        if not (y.shape[1] if y.ndim > 1 else 1) == self.scaling_values.shape[0]:
            raise ValueError('Shapes of scaling_values and y do not match.')
        
        # Inverse transform
        y_inverse_transformed = y * self.scaling_values
        
        return y_inverse_transformed   
    
    
    
    
    
    
    
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
    
    
    
    
    
    #### Function to scale data
    def scale(self, arr, fitted_scaler, groups=None, **kwargs):

        """

        Function scales data by group (if provided). If groups are provided, fitted_scalers should
        be a dict with keys matching the provided (unique) groups.

        Inputs:

            arr: np.array of shape (n_samples, n_variables)
            fitted_scaler: (dict of) fitted scaler(s) that understand '.transform()' call
            groups: np.array or pd.Series with length n_samples over which to apply scaling
            kwargs: ignored

        Outputs:

            arr_scaled: scaled data

        """
        
        # Scale features
        arr_scaled = copy.deepcopy(arr)
        
        # By group
        if not groups is None:
            for group in np.unique(groups):
                arr_scaled[groups == group] = fitted_scaler[group].transform(arr[groups == group])
        else:
            arr_scaled = fitted_scaler.transform(arr)

        return arr_scaled
    
    
    
    #### Function to rescale data
    def rescale(self, arr, fitted_scaler, groups=None, **kwargs):

        """

        Function rescales data by group (if provided). If groups are provided, fitted_scalers should
        be a dict with keys matching the provided (unique) groups.

        Inputs:

            arr: np.array of shape (n_samples, n_variables)
            fitted_scaler: (dict of) fitted scaler(s) that understand '.inverse_transform()' call
            groups: np.array or pd.Series with length n_samples over which to apply scaling
            kwargs: ignored

        Outputs:

            arr_rescaled: rescaled data

        """
        
        # Scale features
        arr_rescaled = copy.deepcopy(arr)
        
        # By group
        if not groups is None:
            for group in np.unique(groups):
                arr_rescaled[groups == group] = fitted_scaler[group].inverse_transform(arr[groups == group])
        else:
            arr_rescaled = fitted_scaler.inverse_transform(arr)

        return arr_rescaled
    
    
    #### Reshape data to match multi-period rolling horizon
    def reshape_data(self, arr, timePeriods, tau, **kwargs):
        
        """

        Function that creates data slices to match the periods of the rolling horizon. It reshapes the data as consecutive vectors
        whose length equals the length of the rolling look-ahead horizon and such that consecutive vectors do not overlap.

        Arguments:

            arr: np.array to reshape, shape (n_samples, n_columns)
            timePeriods: np.array providing the according timePeriod of each row in data, shape (n_samples,)
            tau: look-ahead
            kwargs: ignored

        Outputs: 

            sliced_arr: np.array, shape (n_samples/(tau+1), n_columns)

        """
        
        # Get slices of timePeriods
        maxTimePeriod = max(timePeriods)

        slices = []
        factor = 0
        step = 0

        while maxTimePeriod - step > 0:
            factor = factor + 1
            slices = slices + [maxTimePeriod - step]
            step = (tau+1) * factor        

        # Apply slices
        sliced_arr = arr[pd.Series(timePeriods).isin(slices)]

        # Return 
        return sliced_arr    
    
      
    
    
    
    
    
    
    ### Function to preprocess global data
    def preprocess_weightsmodel_data(self, X, y, ids_products, ids_timePeriods, timePeriodsTestStart, tau, T=None, 
                                     features=None, features_to_scale=None, features_to_scale_with=None, 
                                     X_scaler=None, y_scaler=None, products=None, **kwargs):                  
                    
        
        """
        ...
        
        """
        
        def preprocess_(X, y, ids_products, ids_timePeriods, timePeriodsTestStart, tau, features=None, 
                        features_to_scale=None, features_to_scale_with=None, X_scaler=None, y_scaler=None, **kwargs):

            """
            
            Inner function...

            """
            
            # Check if all args for feature and demand scaling are provided   
            scale = (
                
                (not features is None) and 
                (not features_to_scale is None) and 
                (not features_to_scale_with is None) and 
                (not X_scaler is None) and (not y_scaler is None)      
            
            )
            
            # Feature and demand scaling
            if scale:
                
                # Select training horizons for feature scaling (we allow for <= as we start with lag 1 for all features)
                train_features = (ids_timePeriods <= timePeriodsTestStart)

                # Fit feature scalers per product (on training horizon)
                X_scalers = {}
                for product in np.unique(ids_products):
                    scaler = copy.deepcopy(X_scaler, **kwargs)
                    X_scalers[product] = scaler.fit(X[train_features & (ids_products == product)], features, features_to_scale, features_to_scale_with)

                # Select training horizons for demand scaling
                train_demands = (ids_timePeriods < timePeriodsTestStart)

                # Fit demand scalers per product (on training horizon)
                y_scalers = {}
                for product in np.unique(ids_products):
                    scaler = copy.deepcopy(y_scaler)
                    y_scalers[product] = scaler.fit(y[train_demands & (ids_products == product)], tau=tau, **kwargs)

            # Reshape demand time series to multi-period 
            data = pd.DataFrame({'product': ids_products, 'timePeriod': ids_timePeriods, 'y': y.flatten()})
            y_mp = {}
            for tau_ in range(tau+1):
                y_mp['y'+str(tau_)] = data.groupby(['product']).shift(-tau_)['y']
            y = np.array(pd.DataFrame(y_mp))

            # Select train / test horizons considering multi-period demands given tau
            train, test = ids_timePeriods < timePeriodsTestStart - tau, ids_timePeriods == timePeriodsTestStart

            # Select train / test data
            X_train, X_test = X[train], X[test]
            y_train, y_test = y[train], y[test]
            ids_products_train, ids_products_test = ids_products[train], ids_products[test]
            ids_timePeriods_train, ids_timePeriods_test = ids_timePeriods[train], ids_timePeriods[test]

            # Rehsape training data aligned to periods of the rolling look ahead horizon
            X_train = self.reshape_data(X_train, ids_timePeriods_train, tau)
            y_train = self.reshape_data(y_train, ids_timePeriods_train, tau)
            ids_products_train = self.reshape_data(ids_products_train, ids_timePeriods_train, tau)
            ids_timePeriods_train = self.reshape_data(ids_timePeriods_train, ids_timePeriods_train, tau)

            # Flatten 1-d arrays
            ids_products_train, ids_products_test = ids_products_train.flatten(), ids_products_test.flatten()
            ids_timePeriods_train, ids_timePeriods_test = ids_timePeriods_train.flatten(), ids_timePeriods_test.flatten()
            if tau == 0:
                y_train, y_test = y_train.flatten(), y_test.flatten()

            # Data to return
            data = {

                'ids_timePeriods_train': ids_timePeriods_train, 'ids_timePeriods_test': ids_timePeriods_test,
                'ids_products_train': ids_products_train, 'ids_products_test': ids_products_test,
                'X_train': X_train, 'X_test': X_test,
                'y_train': y_train, 'y_test': y_test
            }
                       
            if scale:
                data.update({'X_scalers': X_scalers, 'y_scalers': y_scalers})

            return data

        
        """
        
        ...
        
        
        """
        
        # If products are provided, apply by product
        if not products is None:
            
            # Initialize
            data = {}
            
            # For each product (SKU) k=1,...,M
            for product in products:
                
                # If a horizon is provided
                if not T is None:

                    # Initialize
                    data[product] = {}
                
                    # For each period t=1,...,T
                    for t in range(1,T+1):
                        
                        # Adjust look-ahead tau to account for end of horizon
                        tau_ = min(tau,T-t)

                        # Preprocess data
                        data[product][t] = preprocess_(X[ids_products==product], y[ids_products==product], 
                                                       ids_products[ids_products==product], ids_timePeriods[ids_products==product], 
                                                       timePeriodsTestStart+t-1, tau_)
                        
                # No horizon is provided
                else:
        
                    # Preprocess data
                    data[product] = preprocess_(X[ids_products==product], y[ids_products==product], 
                                                ids_products[ids_products==product], ids_timePeriods[ids_products==product], 
                                                timePeriodsTestStart, tau)
                
        # No products are provided
        else:
            
            # If a horizon is provided
            if not T is None:

                # Initialize
                data = {}
            
                # For each period t=1,...,T
                for t in range(1,T+1):

                    # Adjust look-ahead tau to account for end of horizon
                    tau_ = min(tau,T-t)
                    
                    # Preprocess data
                    data[t] = preprocess_(X, y, ids_products, ids_timePeriods, timePeriodsTestStart+t-1, tau_, features, 
                                          features_to_scale, features_to_scale_with, X_scaler, y_scaler)
                    
            # No horizon is provided
            else:

                # Preprocess data
                data = preprocess_(X, y, ids_products, ids_timePeriods, timePeriodsTestStart, tau, features, 
                                   features_to_scale, features_to_scale_with, X_scaler, y_scaler)
        
        
        # Return
        return data

    
    
    


    #### Function to generate training and validation data splits
    def split_timeseries_cv(self, n_splits, ids_timePeriods, **kwargs):

        """

        Function creates n_splits "rolling" cv folds of the ids_timePeriods provided. The output is a list of n_splits folds,
        each of which is a series of True/False indicators for train/validation sets, respectively.
        
        Arguments:
        
            n_splits: number of training/validation splits (folds) to create
            ids_timePeriods: series of the timePeriods to be split
            kwargs: ignored
            
        Outputs: 
        
            cv_folds: list of tuples of training and validation folds, each of which is a list of True/False 
                      indicators with length of ids_timePeriods

        """

        # Range of sale yearweeks
        ids_timePeriods_range = np.array(range(int(min(ids_timePeriods)),
                                               int(max(ids_timePeriods))))

        # Initailize rolling time series cross validation splits
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_folds = list()

        # For each fold
        for timePeriods_train_idx, timePeriods_val_idx in tscv.split(range(0,len(ids_timePeriods_range))):

            idx_train = ((ids_timePeriods >= min(ids_timePeriods_range[timePeriods_train_idx])) & 
                         (ids_timePeriods <= max(ids_timePeriods_range[timePeriods_train_idx])))

            idx_val = ((ids_timePeriods >= min(ids_timePeriods_range[timePeriods_val_idx])) & 
                      (ids_timePeriods <= max(ids_timePeriods_range[timePeriods_val_idx])))

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
    def load_cv_result(self, path, product=None):
        
        if product is None:
            
            # Load cv results
            self.cv_result = joblib.load(path)
        
        else:
        
            # Load cv results of product
            self.cv_result = joblib.load(path)[product]
            
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
        
        
        
        
       
    