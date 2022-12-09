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




# Import Weights Model
from WeightsModel4 import PreProcessing
from WeightsModel4 import MaxQFeatureScaler
from WeightsModel4 import MaxQDemandScaler
from WeightsModel4 import RandomForestWeightsModel

# Import (Rolling Horizon) Weighted SAA models
from WeightedSAA6 import WeightedSAA
from WeightedSAA6 import RobustWeightedSAA
from WeightedSAA6 import RollingHorizonOptimization



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
        
        # Set folder names as global variables
        os.chdir('/home/fesc/DataDrivenDynamicInventoryControl/')
        global PATH_DATA, PATH_WEIGHTSMODEL, PATH_RESULTS

        PATH_DATA = '/home/fesc/DataDrivenDynamicInventoryControl/Data' 
        PATH_WEIGHTSMODEL = '/home/fesc/DataDrivenDynamicInventoryControl/Data/WeightsModel'
        PATH_RESULTS = '/home/fesc/DataDrivenDynamicInventoryControl/Data/Results'

        # Weights models
        global_weightsmodel = 'rfwm_global_r_z_h' 
        local_weightsmodel = 'rfwm_local_r_z_h' 

        # Optimization models
        GwSAA = 'GwSAA'
        GwSAAR = 'GwSAAR'
        wSAA = 'wSAA'
        wSAAR = 'wSAAR'
        SAA = 'SAA'
        ExPost = 'ExPost'
        
        
        # Problem params
        self.T = 13                      # Planning horizon T
        self.ts = range(1,self.T+1)      # Periods t=1,...,T of the planning horizon
        self.products = range(1,460+1)   # Products (SKUs) k=1,...,M
        self.taus = [0,1,2,3,4]          # Look-aheads tau=0,...,4
        self.es = [1,3,6,9,12]           # Uncertainty set specifications e=1,...,12

     

        # Train/test split (first timePeriods of testing horizon)
        self.timePeriodsTestStart = 114

        # Cost param settings
        self.cost_params = [

            {'CR': 0.50, 'K': 100, 'u': 0.5, 'h': 1, 'b': 1},
            {'CR': 0.75, 'K': 100, 'u': 0.5, 'h': 1, 'b': 3},
            {'CR': 0.90, 'K': 100, 'u': 0.5, 'h': 1, 'b': 9}

        ]
        
        
        # Data
        data_paths = {
        
            'ID_Data': PATH_DATA+'/ID_Data.csv',
            'X_Data': PATH_DATA+'/X_Data.csv',
            'X_Data_Columns': PATH_DATA+'/X_Data_Columns4.csv',
            'Y_Data': PATH_DATA+'/Y_Data.csv',
        
        }
        
        results_paths = '...'
        
        weightsmodels_paths = '...'
    
    


        
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
    
    
    #### ...
    def preprocess_experiment_data(self, data, weights=None, e=None, **kwargs):
    
        """

        ...


        """

        #self.products
        #self.es
        #self.ts
        #self.taus

        #self.cost_params
        #self.experiment_params
        # ...

        products = kwargs.get('products', copy.deepcopy(self.products))
        ts = kwargs.get('ts', copy.deepcopy(self.ts))

        # Initialize preprocessing
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
                    products_train = data[t]['products_train']
                    y_scalers = data[t]['y_scalers']

                    # Scale demand with fitted demand scaler per product
                    y_train_z = pp.scale(y_train, y_scalers, products_train)  

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
                    products_test = data[t]['products_test']

                    # Store demand actuals
                    actuals_[product][t] = y_test[products_test==product].flatten()

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
                        products_test = data[t]['products_test']

                        # Store weights
                        weights_[product][t] = weights[t][products_test==product].flatten()

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
                        products_train = data[t]['products_train']

                        # Calculate epsilon as e * in-sample standard deviation of current product's demand
                        epsilons_[product][t] = e * np.std(y_train[products_train==product].flatten())

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
    
    
    
    
    
    #### Function to run an experiment over a list of given cost parameter settings and the specified model
    def run(self, wsaamodel, cost_params, actuals, samples=None, weights=None, epsilons=None, print_progress=False,
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
        