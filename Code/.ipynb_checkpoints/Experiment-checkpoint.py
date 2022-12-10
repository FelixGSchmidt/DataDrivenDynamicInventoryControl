"""

...


"""

# Imports
import numpy as np
import pandas as pd
import copy
import time
import datetime as dt
import pickle
import json
import joblib
from joblib import dump, load

import itertools
import contextlib
from tqdm import tqdm



# Import Weights Model
from WeightsModel import PreProcessing
from WeightsModel import MaxQFeatureScaler
from WeightsModel import MaxQDemandScaler
from WeightsModel import RandomForestWeightsModel

# Import (Rolling Horizon) Weighted SAA models
from WeightedSAA import WeightedSAA
from WeightedSAA import RobustWeightedSAA
from WeightedSAA import RollingHorizonOptimization


#### ...
class Experiment:
    
    """
    ...
    
    """
    
    ## Initialize
    def __init__(self, weightsmodel_name=None, optimizationmodel_name=None, **kwargs):
        
        """
        ...
        
        """
        
        # Names of experiment modules
        self.weightsmodel_name = weightsmodel_name if not weightsmodel_name is None else 'weightsmodel'
        self.optimizationmodel_name = optimizationmodel_name if not optimizationmodel_name is None else 'optimizationmodel'
        
        # Default experiment setup
        self.experiment_setup = dict(

            # Set paths
            path_data = '~',
            path_weightsmodel = '~',
            path_results = '~',

            # Set identifier of start period of testing horizon
            timePeriodsTestStart = 114,

            # Set product identifiers
            products = range(1,460+1),   # Products (SKUs) k=1,...,M

            # Set problem params
            T = 13,             # Planning horizon T
            ts = range(1,13+1), # Periods t=1,...,T of the planning horizon
            taus = [0,1,2,3,4], # Look-aheads tau=0,...,4
            es = [1,3,6,9,12],  # Uncertainty set specifications e=1,...,12

            # Set cost params
            cost_params = [
                {'CR': 0.50, 'K': 100, 'u': 0.5, 'h': 1, 'b': 1},
                {'CR': 0.75, 'K': 100, 'u': 0.5, 'h': 1, 'b': 3},
                {'CR': 0.90, 'K': 100, 'u': 0.5, 'h': 1, 'b': 9}
            ]
        )
        
        # Update experiment setup if provided
        self.experiment_setup.update(**kwargs)
        
        return None

    
    
    #### ...
    def load_data(self, which_data=None, path=None, **kwargs):
        
         
        """
        
        ...
        
        Arguments:
        
            ...
            
        Returns:
        
            ...
        
        """
    
        # Update path for data if provided
        if path is None:
            path = self.experiment_setup['path_data']         
            
        # Read data
        ID_Data = pd.read_csv(path+'/'+kwargs.get('ID_Data_name', 'ID_Data.csv'))
        X_Data = pd.read_csv(path+'/'+kwargs.get('X_Data_name', 'X_Data.csv'))
        X_Data_Columns = pd.read_csv(path+'/'+kwargs.get('X_Data_Columns_name', 'X_Data_Columns.csv'))
        Y_Data = pd.read_csv(path+'/'+kwargs.get('Y_Data_name', 'Y_Data.csv'))

        # Feature names
        features = np.array(X_Data_Columns.Feature)
        features_global = np.array(X_Data_Columns.loc[X_Data_Columns.Global == 'YES'].Feature)
        features_global_to_scale = np.array(X_Data_Columns.loc[(X_Data_Columns.Global == 'YES') & (X_Data_Columns.Scale == 'YES')].Feature)
        features_global_to_scale_with = np.array(X_Data_Columns.loc[(X_Data_Columns.Global == 'YES') & (X_Data_Columns.Scale == 'YES')].ScaleWith)
        features_local = np.array(X_Data_Columns.loc[X_Data_Columns.Local == 'YES'].Feature)

        # Ensure data is sorted by SKU and sale_yearweek for further preprocessing
        data = pd.concat([ID_Data, X_Data, Y_Data], axis=1).sort_values(by=['SKU', 'sale_yearweek']).reset_index(drop=True)
        X = np.array(data[X_Data.columns])
        y = np.array(data[Y_Data.columns]).flatten()
        products = np.array(data['SKU']).astype(int)
        timePeriods = np.array(data['sale_yearweek']).astype(int)

        # Decide which data to return
        if which_data is None:

            data = X, y, products, timePeriods, features, features_global, features_global_to_scale, features_global_to_scale_with, features_local

        elif which_data == 'global':

            # Select global features
            X = X[:,[feat in features_global for feat in features]]

            data = X, y, products, timePeriods, features_global, features_global_to_scale, features_global_to_scale_with

        elif which_data == 'local':

            # Select local features
            X = X[:,[feat in features_local for feat in features]]

            data = X, y, products, timePeriods, features_local

        return data
    
    
    #### ...
    def preprocess_data(self, data, weights=None, e=None, **kwargs):
    
        """

        ...


        """

        # Update products and periods if provided
        products = kwargs.get('products', copy.deepcopy(self.experiment_setup['products']))
        ts = kwargs.get('ts', copy.deepcopy(self.experiment_setup['ts']))

        # Initialize preprocessing module of weights model
        pp = PreProcessing()

        # Local (if data is provided per product for each period)
        if len(data) == len(products):

            """ Local training and sampling data """

            # Samples
            samples_ = {}

            # For products k=1,...,M
            for product in products:

                # Initialize
                samples_[product] = {}

                # For periods t=1,...,T
                for t in ts:

                    # Get demand data of current product
                    y_train = data[product][t]['y_train']

                    # Store demand samples
                    samples_[product][t] = copy.deepcopy(y_train)

            # Actuals
            actuals_ = {}

            # For products k=1,...,M
            for product in products:

                # Initialize
                actuals_[product] = {}

                # For periods t=1,...,T
                for t in ts:

                    # Get demand actuals of current product
                    y_test = data[product][t]['y_test']

                    # Store demand actuals
                    actuals_[product][t] = y_test.flatten()

            # Weights   
            if not weights is None:

                weights_ = {}

                # For products k=1,...,M
                for product in products:

                    # Initialize
                    weights_[product] = {}

                    # For periods t=1,...,T
                    for t in ts:

                        # Store weights
                        weights_[product][t] = weights[product][t].flatten()

            # Epsilons  
            if not e is None:

                epsilons_ = {}

                # For products k=1,...,M
                for product in products:

                    # Initialize
                    epsilons_[product] = {}

                    # For periods t=1,...,T
                    for t in ts:

                        # Get demand data of current product
                        y_train = data[product][t]['y_train']

                        # Calculate epsilon as e * in-sample standard deviation of current product's demand
                        epsilons_[product][t] = e * np.std(y_train.flatten())




        # Global (if data is provided directly per period without separation by products)
        elif len(data) == len(ts):

            """ Global training and sampling data """

            # Samples
            samples_ = {}

            # For products k=1,...,M
            for product in products:

                # Initialize
                samples_[product] = {}

                # For periods t=1,...,T
                for t in ts:

                    # Get demand data, product identifiers, and fitted scalers
                    y_train = data[t]['y_train']
                    ids_products_train = data[t]['ids_products_train']
                    y_scalers = data[t]['y_scalers']

                    # Scale demand with fitted demand scaler per product
                    y_train_z = pp.scale(y_train, y_scalers, ids_products_train)  

                    # Rescale demand with fitted demand scaler of the current product
                    y_train_zz = pp.rescale(y_train_z, y_scalers[product]) 

                    # Store demand samples
                    samples_[product][t] = copy.deepcopy(y_train_zz)

            # Actuals
            actuals_ = {}

            # For products k=1,...,M
            for product in products:

                # Initialize
                actuals_[product] = {}

                # For periods t=1,...,T
                for t in ts:

                    # Get demand actuals and product identifiers
                    y_test = data[t]['y_test']
                    ids_products_test = data[t]['ids_products_test']

                    # Store demand actuals
                    actuals_[product][t] = y_test[ids_products_test==product].flatten()

            # Weights   
            if not weights is None:

                weights_ = {}

                # For products k=1,...,M
                for product in products:

                    # Initialize
                    weights_[product] = {}

                    # For periods t=1,...,T
                    for t in ts:

                        # Get product identifiers
                        ids_products_test = data[t]['ids_products_test']

                        # Store weights
                        weights_[product][t] = weights[t][ids_products_test==product].flatten()

            # Epsilons  
            if not e is None:

                epsilons_ = {}

                # For products k=1,...,M
                for product in products:

                    # Initialize
                    epsilons_[product] = {}

                    # For periods t=1,...,T
                    for t in ts:

                        # Get demand data, product identifiers, and fitted scalers
                        y_train = data[t]['y_train']
                        ids_products_train = data[t]['ids_products_train']

                        # Calculate epsilon as e * in-sample standard deviation of current product's demand
                        epsilons_[product][t] = e * np.std(y_train[ids_products_train==product].flatten())

        else:
            raise ValueError("Structure of data does not match any of the logics for local or global training and sampling.")



        # Decide which data to return
        if not weights is None:
            if not e is None:
                data_ = samples_, actuals_, weights_, epsilons_
            else:
                data_ = samples_, actuals_, weights_
        else:
            if not e is None:
                data_ = samples_, actuals_, epsilons_
            else:
                data_ = samples_, actuals_



        # Retrun
        return data_
    
    
    
    
    
#     #### Function to run an experiment over a list of given cost parameter settings and the specified model
#     def run(self, wsaamodel, cost_params, actuals, samples=None, weights=None, epsilons=None, 
#             print_progress=False, path_to_save=None, name_to_save=None, return_results=True, **kwargs):

#         """
        
#         ...
        
#         Arguments:
        
#             ...
            
#         Returns:
        
#             ...
        
#         """
        
        
#         # Update experiment setup if provided
#         self.experiment_setup.update(**kwargs)
        
#         locals().update(self.experiment_setup)

#         # Raise error if cost_params is not a list of dict(s)
#         if not type(cost_params)==list:
#             raise ValueError('Argument cost_params has to be a list of at least one dict with keys {K, u, h, b}')  

#         # Timer
#         st_exec, st_cpu = time.time(), time.process_time()

#         # Status
#         if print_progress and 'SKU' in kwargs: print('SKU:', kwargs['SKU'])

#         # Initialize
#         ropt, results = RollingHorizonOptimization(), pd.DataFrame()

#         # For each cost param setting
#         for cost_params_ in cost_params:

#             # Print progress
#             if print_progress: print('...cost param setting:', cost_params_)

#             # Check if samples provided
#             if not samples is None:

#                 # Apply (Weighted) SAA  model
#                 wsaamodel.set_params(**{**kwargs, **cost_params_})
#                 result = ropt.run(wsaamodel, samples, actuals, weights, epsilons)

#                 # Get T
#                 T = len(samples)

#             else:

#                 # Apply ex-post clairvoyant model
#                 wsaamodel.set_params(**{**kwargs, **cost_params_})
#                 result = ropt.run_expost(wsaamodel, actuals)

#                 # Get T
#                 T = actuals.shape[1]

#             # Store results
#             meta = pd.DataFrame({'CR': cost_params_['CR'], **kwargs}, index=list(range(T)))
#             results = pd.concat([results, pd.concat([meta, result], axis=1)], axis=0)

#         # Save result as csv file
#         if not path_to_save is None and not name_to_save is None:
#             results.to_csv(path_or_buf=(path_to_save+'/'+name_to_save+'_SKU'+str(kwargs.get('SKU', None))+
#                                         '_tau'+str(kwargs.get('tau', None))+'.csv'), sep=',', index=False)

#         # Timer
#         exec_time_sec, cpu_time_sec = time.time() - st_exec, time.process_time() - st_cpu

#         # Status
#         if print_progress: print('>>>> Done:', str(np.around(exec_time_sec/60,1)), 'minutes')

#         # Return  
#         return results if return_results else {'SKU': kwargs.get('SKU', None), 'exec_time_sec': exec_time_sec, 'cpu_time_sec': cpu_time_sec}
    
    
    
    
    
    
    
    
    
    #### Function to run an experiment over a list of given cost parameter settings and the specified model
    def run(self, wsaamodel, actuals, samples=None, weights=None, epsilons=None, print_progress=False, return_results=False, save_results=True, **kwargs):

        """
        
        ...
        
        Arguments:
        
            ...
            
        Returns:
        
            ...
        
        """
        
        
        # Update experiment setup if provided
        self.experiment_setup.update(**kwargs)
        
        # Get cost params
        cost_params = copy.deepcopy(self.experiment_setup['cost_params'])
        
        # Get Gurobi params
        gurobi_params = copy.deepcopy(self.experiment_setup['gurobi_params'])

        # Raise error if cost_params is not a list of dict(s)
        if not type(cost_params)==list:
            raise ValueError('Argument cost_params has to be a list of at least one dict with keys {K, u, h, b}')  

        # Timer
        st_exec, st_cpu = time.time(), time.process_time()

        # Status
        if print_progress and 'product' in kwargs: print('Product:', kwargs['product'])

        # Initialize
        ropt, results = RollingHorizonOptimization(), pd.DataFrame()

        # For each cost param setting
        for cost_params_ in cost_params:

            # Print progress
            if print_progress: print('...cost param setting:', cost_params_)

            # Check if samples provided
            if not samples is None:

                # Apply (Weighted) SAA  model
                wsaamodel.set_params(**{**gurobi_params, **cost_params_})
                result = ropt.run(wsaamodel, samples, actuals, weights, epsilons)

                # Get T
                T = len(samples)

            else:

                # Apply ex-post clairvoyant model
                wsaamodel.set_params(**{**gurobi_params, **cost_params_})
                result = ropt.run_expost(wsaamodel, actuals)

                # Get T
                T = actuals.shape[1]

            # Store results
            meta = pd.DataFrame({'CR': cost_params_['CR'], **gurobi_params}, index=list(range(T)))
            results = pd.concat([results, pd.concat([meta, result], axis=1)], axis=0)

        # Save result as csv file
        if save_results:
            name_to_save = self.optimizationmodel_name+'_e'+str(e) if 'e' in kwargs else copy.deepcopy(self.optimizationmodel_name)
            results.to_csv(path_or_buf=(path_results+'/'+self.optimizationmodel_name+'_Product'+str(kwargs.get('product', None))+
                                        '_tau'+str(kwargs.get('tau', None))+'.csv'), sep=',', index=False)

        # Timer
        exec_time_sec, cpu_time_sec = time.time() - st_exec, time.process_time() - st_cpu

        # Status
        if print_progress: print('>>>> Done:', str(np.around(exec_time_sec/60,1)), 'minutes')

        # Return  
        return results if return_results else {'Product': kwargs.get('product', None), 'exec_time_sec': exec_time_sec, 'cpu_time_sec': cpu_time_sec}

    
    
    
    ### Context manager (Credits: 'https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution')
    @contextlib.contextmanager
    def tqdm_joblib(self, tqdm_object):
        
        """
        
        ...
        
        Arguments:
        
            ...
            
        Returns:
        
            ...
        
        """
        
        
        """Context manager to patch joblib to report into tqdm progress bar given as argument"""
        class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
            def __call__(self, *args, **kwargs):
                tqdm_object.update(n=self.batch_size)
                return super().__call__(*args, **kwargs)

        old_batch_callback = joblib.parallel.BatchCompletionCallBack
        joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
        try:
            yield tqdm_object
        finally:
            joblib.parallel.BatchCompletionCallBack = old_batch_callback
            tqdm_object.close()

    
    
    
    
    
    
    
    
    
    

#### ...
class Evaluation:
    
    """
    ...
    
    """
    
    ## Initialize
    def __init__(self, **kwargs):
        
        """
        ...
        
        """
        
        # ...
        
        
        return None

    
       
    
    #### ...
    def xxx(self, **kwargs):
        
         
        """
        ...
        
        """
        
        # ...
        
        return None
        