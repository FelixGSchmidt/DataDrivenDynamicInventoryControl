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






#### ...
class Experiment:
    
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
    def load_experiment_data(self, which_data=None, path=None, **kwargs):
        
         
        """
        
        ...
        
        Arguments:
        
            ...
            
        Returns:
        
            ...
        
        """
    
        if path is None:
            path = ''

        # Read data
        ID_Data = pd.read_csv(path+'/ID_Data.csv')
        X_Data = pd.read_csv(path+'/X_Data.csv')
        X_Data_Columns = pd.read_csv(path+'/X_Data_Columns4.csv')
        Y_Data = pd.read_csv(path+'/Y_Data.csv')

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
    
    
    
    
    
    
    #### Function to run an experiment over a list of given cost parameter settings and the specified model
    def run_experiment(self, wsaamodel, cost_params, actuals, samples=None, weights=None, epsilons=None, print_progress=False,
                       path_to_save=None, name_to_save=None, return_results=True, **kwargs):

        """
        
        ...
        
        Arguments:
        
            ...
            
        Returns:
        
            ...
        
        """

        # Raise error if cost_params is not a list of dict(s)
        if not type(cost_params)==list:
            raise ValueError('Argument cost_params has to be a list of at least one dict with keys {K, u, h, b}')  

        # Timer
        st_exec, st_cpu = time.time(), time.process_time()

        # Status
        if print_progress and 'SKU' in kwargs: print('SKU:', kwargs['SKU'])

        # Initialize
        ropt, results = RollingHorizonOptimization(), pd.DataFrame()

        # For each cost param setting
        for cost_params_ in cost_params:

            # Print progress
            if print_progress: print('...cost param setting:', cost_params_)

            # Check if samples provided
            if not samples is None:

                # Apply (Weighted) SAA  model
                wsaamodel.set_params(**{**kwargs, **cost_params_})
                result = ropt.run(wsaamodel, samples, actuals, weights, epsilons)

                # Get T
                T = len(samples)

            else:

                # Apply ex-post clairvoyant model
                wsaamodel.set_params(**{**kwargs, **cost_params_})
                result = ropt.run_expost(wsaamodel, actuals)

                # Get T
                T = actuals.shape[1]

            # Store results
            meta = pd.DataFrame({'CR': cost_params_['CR'], **kwargs}, index=list(range(T)))
            results = pd.concat([results, pd.concat([meta, result], axis=1)], axis=0)

        # Save result as csv file
        if not path_to_save is None and not name_to_save is None:
            results.to_csv(path_or_buf=(path_to_save+'/'+name_to_save+'_SKU'+str(kwargs.get('SKU', None))+
                                        '_tau'+str(kwargs.get('tau', None))+'.csv'), sep=',', index=False)

        # Timer
        exec_time_sec, cpu_time_sec = time.time() - st_exec, time.process_time() - st_cpu

        # Status
        if print_progress: print('>>>> Done:', str(np.around(exec_time_sec/60,1)), 'minutes')

        # Return  
        return results if return_results else {'SKU': kwargs.get('SKU', None), 'exec_time_sec': exec_time_sec, 'cpu_time_sec': cpu_time_sec}
    
    
    
    
    
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
        