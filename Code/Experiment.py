"""

...


"""


# Imports
import numpy as np
import pandas as pd
from scipy import stats
import copy
import time
import datetime as dt
import os
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
                        
                        if kwargs.get('maxnorm',False):
                            epsilons_[product][t] = e * max(abs(np.mean(y_train.flatten()) - y_train.flatten()))
                        
                        
                        
                        




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
                        y_train_product = y_train[ids_products_train==product].flatten()
                        epsilons_[product][t] = e * np.std(y_train_product)
                        
                        if kwargs.get('maxnorm',False):
                            epsilons_[product][t] = e * max(abs(np.mean(y_train_product) - y_train_product))
                        

        else:
            raise ValueError("Structure of data does not match any of the logics for local or global training and sampling.")



        # Decide which data to return
        if not weights is None:
            if not e is None:
                data_ = weights_, samples_, actuals_, epsilons_
            else:
                data_ = weights_, samples_, actuals_
        else:
            if not e is None:
                data_ = samples_, actuals_, epsilons_
            else:
                data_ = samples_, actuals_



        # Retrun
        return data_
    
    
    
    
    
    
    
    
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
        
        # Get path to save results
        path_to_save = self.experiment_setup['path_results']+'/'+self.optimizationmodel_name
        
        # Get name to save results
        name_to_save = self.optimizationmodel_name+'_e'+str(kwargs.get('e')).replace('.', '') if 'e' in kwargs else copy.deepcopy(self.optimizationmodel_name)
            
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
            meta = pd.DataFrame({'CR': cost_params_.get('CR'), 
                                 **{**{'product': kwargs.get('product'), 'tau': kwargs.get('tau'), 'e': kwargs.get('e')}, **gurobi_params}}, 
                                 index=list(range(T)))
            results = pd.concat([results, pd.concat([meta, result], axis=1)], axis=0)

        # Save result as csv file
        if save_results: 
            results.to_csv(path_or_buf=(path_to_save+'/'+name_to_save+'_product'+str(kwargs.get('product', None))+
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

    
       
    
    ### Function to concatenate all results
    def concatenate_results(self, path, name, products, taus=[None], es=[None]):
        
        """
        ...
        
        """
        
        results = pd.DataFrame()

        # For each product (SKU) k=1,...,M
        for product in products:

            # For each look-ahead tau=0,...,4
            for tau in taus:

                # For each uncertainty set specification  e=1,...,12
                for e in es:

                    if not e is None:
                        
                        file_name = path+'/'+name+'_e'+str(e).replace('.', '')+'_product'+str(product)+'_tau'+str(tau)+'.csv'

                        # Check if results exist   
                        if os.path.exists(file_name):
                            results = pd.concat([results, pd.read_csv(file_name)])

                    else:

                        file_name = path+'/'+name+'_product'+str(product)+'_tau'+str(tau)+'.csv'

                        # Check if results exist   
                        if os.path.exists(file_name):
                            results = pd.concat([results, pd.read_csv(file_name)])
    
        return results
    
    

    ### Function to aggregate results over periods
    def aggregate_results(self, results, groupby=['CR', 'tau', 'e', 'product']):
        
        # Number of periods
        results['n_periods'] = 1
        
        # Service level
        results['n_stockouts'] = results.I_q_y < 0
        
        # Aggregate results over periods t=1,...,T
        results_aggregated = results.groupby(groupby).agg({
            'MIPGap': lambda x: x.iloc[0],
            'NumericFocus': lambda x: x.iloc[0],
            'obj_improvement': lambda x: x.iloc[0],
            'obj_timeout_sec': lambda x: x.iloc[0],
            'obj_timeout_max_sec': lambda x: x.iloc[0],
            'K': lambda x: x.iloc[0],
            'u': lambda x: x.iloc[0],
            'h': lambda x: x.iloc[0],
            'b': lambda x: x.iloc[0],
            'I': np.mean,
            'q': np.mean,
            'I_q': np.mean,
            'y': np.mean,
            'I_q_y': np.mean,
            'n_stockouts': sum,
            'n_periods': sum,
            'c_o': sum,
            'c_s': sum,
            'cost': sum,
            'defaulted': sum,
            'solutions': lambda x: sum(x>0),
            'gap': np.mean,
            'exec_time_sec': sum,
            'cpu_time_sec': sum
        }).reset_index()
        
        return results_aggregated
    
    
    
    
    ### Function to find best tau per defined group (e.g., by CR, product)
    def best_tau(self, results, results_ExPost, groupby=['CR', 'product']):

        """

        If groupby == ['CR', 'product', 'e']: finds best tau per product and per uncertainty set sepcification for given CR
        If groupby == ['CR', 'product']: finds best tau per product for given CR
        If groupby == ['CR']: finds best tau across all products for given CR
        If groupby == []: not implemented


        """

        # Merge aggregated results with ex-post clairvoyant results
        results_best_tau = pd.merge(left=results,
                                    right=results_ExPost[['CR', 'product', 'cost']],
                                    on=['CR', 'product'],
                                    suffixes=('', '_expost'))

        # Calculate gap to ex-post clairvoyant results
        results_best_tau['gap'] = (
            (results_best_tau.cost == results_best_tau.cost_expost) * 1 
            + (results_best_tau.cost != results_best_tau.cost_expost) * (results_best_tau.cost / results_best_tau.cost_expost)
        )

        # Calculate (median) gap per tau and per group
        results_best_tau = results_best_tau.groupby(groupby+['tau']).agg({'gap': np.median}).reset_index()

        # Find tau that minimizes (median) gap
        results_best_tau = results_best_tau.groupby(groupby).apply(
            lambda df:  pd.Series({ 'best_tau': df.tau.iloc[np.argmin(df.gap)], 'best_gap_tau': np.min(df.gap)})).reset_index()

        # Merge best tau back to original results for ex-post selection
        results_best_tau = pd.merge(left=results,
                                    right=results_best_tau,
                                    on=groupby)

        # Select
        results_best_tau = results_best_tau.loc[results_best_tau.tau==results_best_tau.best_tau].reset_index(drop=True)

        return results_best_tau
    
    
    
    
    ### Function to find best e per defined group (e.g., by CR, product)
    def best_e(self, results, results_ExPost, groupby=['CR', 'product']):

        """

        If groupby == ['CR', 'product', 'tau']: finds best e per product and per tau for given CR
        If groupby == ['CR', 'product']: finds best e per product across all tau for given CR
        If groupby == ['CR']: finds best e across all products and across all tau for given CR
        If groupby == []: not implemented


        """

        # Merge aggregated results with ex-post clairvoyant results
        results_best_e = pd.merge(left=results,
                                  right=results_ExPost[['CR', 'product', 'cost']],
                                  on=['CR', 'product'],
                                  suffixes=('', '_expost'))

        # Calculate gap to ex-post clairvoyant results
        results_best_e['gap'] = (
            (results_best_e.cost == results_best_e.cost_expost) * 1 
            + (results_best_e.cost != results_best_e.cost_expost) * (results_best_e.cost / results_best_e.cost_expost)
        )

        # Calculate (median) gap per e and per group
        results_best_e = results_best_e.groupby(groupby+['e']).agg({'gap': np.median}).reset_index()

        # Find e that minimizes (median) gap
        results_best_e = results_best_e.groupby(groupby).apply(
            lambda df:  pd.Series({ 'best_e': df.e.iloc[np.argmin(df.gap)], 'best_gap_e': np.min(df.gap)})).reset_index()

        # Merge best e back to original results for ex-post selection
        results_best_e = pd.merge(left=results,
                                  right=results_best_e,
                                  on=groupby)

        # Select
        results_best_e = results_best_e.loc[results_best_e.e==results_best_e.best_e].reset_index(drop=True)

        return results_best_e
    
    
    
    
    ## Root Mean Scaled Squared Error
    def root_mean_scaled_squared_error(self, weights, samples, samples_saa, actuals, **kwargs):
        
        """
        
        ...
        
        
        """

        # Initialize
        se = np.array([])
        se_saa = np.array([])

        for t in actuals.keys():

            # Ingredients
            w = weights[t]
            d = samples[t]
            d_saa = samples_saa[t]
            a = actuals[t]

            # Reshape
            if len(d.shape) == 1:
                d = d.reshape(-1, 1)
            if len(d_saa.shape) == 1:
                d_saa = d_saa.reshape(-1, 1)

            # Squared errors Model
            se = np.append(se, (np.sum(w.reshape(-1,1) * d, axis = 0) - a.flatten())**2)

            # Squared errors SAA
            se_saa = np.append(se_saa, (np.sum(1/len(d_saa) * d_saa, axis = 0) - a.flatten())**2)

        # MSE Model
        mse = np.mean(se)

        # MSE SAA
        mse_saa = np.mean(se_saa)

        # RMSSE   
        with np.errstate(divide='ignore'):
            rmsse = (mse == mse_saa) * 1.0 + (mse != mse_saa) * (mse / mse_saa)**(1/2)

        return rmsse



    ## Scaled Pinball Loss
    def scaled_pinball_loss(self, weights, samples, samples_saa, actuals, u, **kwargs):
    
        """
        
        ...
        
        """
        
        # Initialize
        q_u = np.array([])
        q_u_saa = np.array([])
        y = np.array([])

        for t in actuals.keys():

            # Ingredients
            w = weights[t]
            d = samples[t]
            d_saa = samples_saa[t]
            a = actuals[t]

            # Reshape
            if len(d.shape) == 1:
                d = d.reshape(-1, 1)
            if len(d_saa.shape) == 1:
                d_saa = d_saa.reshape(-1, 1)

            # u-Percentile Model
            for s in range(d.shape[1]):

                # Create cumulative distribution 
                cdf = pd.DataFrame({'w': w[w>0], 'd': d[:,s].flatten()[w>0]})
                cdf = cdf.groupby('d').agg({'w': sum}).reset_index().sort_values('d')
                cdf['w'] = cdf['w'].cumsum()

                # Get percentile at u: q(u)
                q_u = np.append(q_u, min(cdf.d.loc[u <= cdf.w].values))

            # u-Percentile SAA
            for s in range(d_saa.shape[1]):

                # Get percentile at u: q(u)
                q_u_saa = np.append(q_u_saa, np.quantile(d_saa[:,s].flatten(), u, method='closest_observation'))

            # Actuals
            y = np.append(y, a.flatten())

        # Pinball Loss Model
        pl = np.mean((y - q_u) * u * (q_u <= y) + (q_u - y) * (1-u) * (q_u > a))

        # Pinball Loss SAA
        pl_saa = np.mean((y - q_u_saa) * u * (q_u_saa <= y) + (q_u_saa - y) * (1-u) * (q_u_saa > a))

        # Scaled Pinball Loss
        with np.errstate(divide='ignore'):
            spl = (pl == pl_saa) * 1.0 + (pl != pl_saa) * (pl / pl_saa)

        return spl

    
#     ## Scaled Mean Wasserstein Distance
#     def scaled_mean_wasserstein_distance(self, weights, samples, samples_saa, actuals, p, **kwargs):
        
#         """
        
#         ...
        
#         """
        

#         # Initialize
#         wd = []
#         wd_saa = []

#         for t in actuals.keys():

#             # Ingredients
#             w = weights[t]
#             d = samples[t]
#             d_saa = samples_saa[t]
#             a = actuals[t]

#             # Reshape
#             if len(d.shape) == 1:
#                 d = d.reshape(-1, 1)
#             if len(d_saa.shape) == 1:
#                 d_saa = d_saa.reshape(-1, 1)
#             if len(a.shape) == 1:
#                 a = a.reshape(1, -1)

#             # Wasserstein Distances Model
#             wd = np.append(wd, np.sum(w * np.sum(np.abs(d - a)**p, axis = 1))**(1/p))

#             # Wasserstein Distances SAA
#             wd_saa = np.append(wd_saa, np.sum(1/len(d_saa) * np.sum(np.abs(d_saa - a)**p, axis = 1))**(1/p))

#         # Wasserstein Distance Model
#         mwd = np.mean(wd)

#         # Wasserstein Distance SAA
#         mwd_saa = np.mean(wd_saa)

#         # Scaled Mean Wasserstein Distance
#         smwd = 1.0 if mwd==mwd_saa else mwd/mwd_saa

#         return smwd
    

    
    
#     ### Function to evaluate predictive performance
#     def predictive_performance(self, weights, samples, samples_saa, actuals, 
#                                u=[0.005, 0.025, 0.165, 0.250, 0.500, 0.750, 0.835, 0.975, 0.995], p=[1,2], **kwargs):

#         """

#         ...

#         """

#         # Root Mean Scaled Squared Error
#         rmsse = self.root_mean_scaled_squared_error(weights, samples, samples_saa, actuals)

#         # Scaled Pinball Loss
#         if type(u)==list:
#             spl = {}
#             for u_ in u:
#                 spl[u_] = self.scaled_pinball_loss(weights, samples, samples_saa, actuals, u_)
#         else:
#             spl = self.scaled_pinball_loss(weights, samples, samples_saa, actuals, u)

#         # Scaled Mean Wasserstein Distance
#         if type(p)==list:
#             smwd = {}
#             for p_ in p:
#                 smwd[p_] = self.scaled_mean_wasserstein_distance(weights, samples, samples_saa, actuals, p_)
                
#         else:
#             smwd = self.scaled_mean_wasserstein_distance(weights, samples, samples_saa, actuals, p)
            
#         return rmsse, spl, smwd
    
    
     
    ### Function to evaluate predictive performance
    def predictive_performance(self, weights, samples, samples_saa, actuals, 
                               u=[0.005, 0.025, 0.165, 0.250, 0.500, 0.750, 0.835, 0.900, 0.975, 0.995], **kwargs):

        """

        ...

        """

        # Root Mean Scaled Squared Error
        rmsse = self.root_mean_scaled_squared_error(weights, samples, samples_saa, actuals)

        # Scaled Pinball Loss
        if type(u)==list:
            spl = {}
            for u_ in u:
                spl[u_] = self.scaled_pinball_loss(weights, samples, samples_saa, actuals, u_)
        else:
            spl = self.scaled_pinball_loss(weights, samples, samples_saa, actuals, u)

        return rmsse, spl
    
    
    
    
    
    def prescriptive_performance(self, cost, cost_saa, ids_groups=None, groupby=[None], **kwargs):
    
        """

        ...


        """

        # For each element
        with np.errstate(divide='ignore'):
                
            pq = (cost == cost_saa) * 1 + (cost != cost_saa) * (cost / cost_saa)

        # By group(s)
        if not ids_groups is None and not groupby is None:
            pq_bygroup = pd.DataFrame({'cost': cost, 'cost_saa': cost_saa, **ids_groups}, columns = ['cost', 'cost_saa'] + groupby)
            pq_bygroup = pq_bygroup.groupby(groupby).agg({'cost': sum, 'cost_saa': sum}).reset_index()
            
            with np.errstate(divide='ignore'):
                
                pq_bygroup['pq'] = (
                    (pq_bygroup.cost == pq_bygroup.cost_saa) * 1.0 
                    + (pq_bygroup.cost != pq_bygroup.cost_saa) * (pq_bygroup.cost / pq_bygroup.cost_saa)
                )

            return pq, pq_bygroup

        else:
            return pq
        
        
        
    
    def service_level(self, results_data, **kwargs):
    
        """

        ...


        """

        result = results_data.groupby(['CR', 'model']).agg({
            
            'n_stockouts': sum,
            'n_stockouts_SAA': sum,
            'n_periods': sum

        }).reset_index()
        
        result['sl'] = 1 - result.n_stockouts / result.n_periods
        
        result['sl_SAA'] = 1 - result.n_stockouts_SAA / result.n_periods

        return result
        
        
        
        
        
    def differences(self, results_data, test='paired', **kwargs):
        
        """
        
        ...
        
        """
    
        # Groups
        models = list(results_data.model.unique())
        CRs = list(results_data.CR.unique())

        # Initialize
        result = []

        # For each CR
        for CR in CRs:

            # For each model
            for model in models:

                # Models to compare to (all others)
                benchmarks = [i for (i, v) in zip(models, [model != m for m in models]) if v]

                # For all models to compare to
                for benchmark in benchmarks:

                    # Cost
                    cost_model = np.array(results_data.loc[(results_data.CR == CR) & (results_data.model == model)]['cost'])
                    cost_benchmark = np.array(results_data.loc[(results_data.CR == CR) & (results_data.model == benchmark)]['cost'])
                    cost_ExPost = np.array(results_data.loc[(results_data.CR == CR) & (results_data.model == model)]['cost_ExPost'])

                    # Prescriptive performance
                    pq_model_ = np.array(results_data.loc[(results_data.CR == CR) & (results_data.model == model)]['pq'])
                    pq_benchmark_ = np.array(results_data.loc[(results_data.CR == CR) & (results_data.model == benchmark)]['pq'])    

                    # Differences
                    with np.errstate(divide='ignore'):

                        diffs_ = (cost_model == cost_benchmark) * 0 + (cost_model != cost_benchmark) * (pq_model_ - pq_benchmark_)

                    # Remove inf / nan
                    cost_ExPost = cost_ExPost[np.isfinite(diffs_)]
                    diffs = diffs_[np.isfinite(diffs_)]
                    pq_model = pq_model_[np.isfinite(pq_model_) & np.isfinite(pq_benchmark_)]
                    pq_benchmark = pq_benchmark_[np.isfinite(pq_model_) & np.isfinite(pq_benchmark_)]

                    ## Paired test of differences (Wilcoxon Signed Rank Sum Test)
                    if test == 'paired':

                        # Mean of differences
                        mean_of_differences = np.mean(diffs)

                        # Median of differences
                        median_of_differences = np.median(diffs)

                        # Share of cases where model is better than benchmark
                        share_model_is_better = sum(diffs < 0) / len(diffs)

                        # Share of associated ex-post optimal cost where model is better than benchmark
                        share_cost_model_is_better = sum(cost_ExPost[diffs < 0]) / sum(cost_ExPost)

                        # Statictical significance
                        statistic, pvalue = stats.wilcoxon(diffs, alternative='less', nan_policy='raise')

                        # Store
                        res = {

                            'CR': CR, 
                            'model': model, 
                            'benchmark': benchmark, 
                            'mean_of_differences': mean_of_differences,
                            'median_of_differences': median_of_differences,
                            'share_model_is_better': share_model_is_better,
                            'share_cost_model_is_better': share_cost_model_is_better,
                            'statistic': statistic,
                            'pvalue': pvalue
                        }

                        # Append
                        result += [res]


                    ## Unpaired test of differences (Mann-Whitney U Test)
                    elif test == 'unpaired':

                        # Difference of means
                        difference_of_means = np.mean(pq_model) - np.mean(pq_benchmark) 

                        # Difference of medians
                        difference_of_medians = np.median(pq_model) - np.median(pq_benchmark) 

                        # Share of cases where model is better than benchmark
                        share_model_is_better = sum(diffs < 0) / len(diffs)

                        # Share of associated ex-post optimal cost where model is better than benchmark
                        share_cost_model_is_better = sum(cost_ExPost[diffs < 0]) / sum(cost_ExPost)

                        # Statictical significance
                        statistic, pvalue = stats.mannwhitneyu(pq_model, pq_benchmark, alternative='less', nan_policy='raise')

                        # Store
                        res = {

                            'CR': CR, 
                            'model': model, 
                            'benchmark': benchmark, 
                            'difference_of_means': difference_of_means,
                            'difference_of_medians': difference_of_medians,
                            'share_model_is_better': share_model_is_better,
                            'share_cost_model_is_better': share_cost_model_is_better,
                            'statistic': statistic,
                            'pvalue': pvalue
                        }

                        # Append
                        result += [res]

        # Result         
        return pd.DataFrame(result)