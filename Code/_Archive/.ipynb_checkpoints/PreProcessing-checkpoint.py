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
    def reshape_data(self, ID_Data, Y_Data, X_Data, T_horizon_rolling, max_sale_yearweek, iid = True):

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
            max_sale_yearweek[int]: last sale_yearweek of the training horizon 
            iid[bool]: True if data should be reshaped to iid

        Outputs: 

            ID_train[dataframe]: data storing identifiers of reshaped training data
            Y_train[dataframe]: reshaped demand training data
            X_train[dataframe]: feature training data


        """


        # Combine IDs with Y data, sort by SKU and sale yearweek, and select training data
        Data = pd.concat(objs=[ID_Data,Y_Data], axis=1).sort_values(by=['SKU', 'sale_yearweek'])
        Data = Data.loc[Data.sale_yearweek <= max_sale_yearweek]

        ## Create look-aheads given rolling horizon

        # For rolling horizon == 1, use current Y
        Y_shifted = pd.DataFrame({'Y'+str(1): Data.Y})

        # For rolling horizon > 1, shift Y upwards
        if T_horizon_rolling != 1:

            # Shift for each priod of the rolling horizon
            for T in range(1,T_horizon_rolling):
                Y_shifted = pd.concat([Y_shifted, pd.DataFrame({'Y'+str(T+1): Data.groupby('SKU').Y.shift(-T)})], axis=1)

        # Data frame of shifted Y
        Y_shifted = pd.concat([Data[['SKU', 'sale_year', 'sale_week', 'sale_yearweek']], Y_shifted], axis=1).dropna()

        ## Merge to ID Data and X_Data
        Data = pd.concat(
            objs=[ID_Data.merge(
                Y_shifted, how='left', on=['SKU', 'sale_year', 'sale_week', 'sale_yearweek']), 
                  X_Data], 
            axis=1)

        # Reduce to training data
        Data = Data.loc[Data.sale_yearweek <= max_sale_yearweek].dropna()

        ## Reshape to iid
        if iid:

            ## Get slices of sale_yearweek to ensure iid data
            max_sale_yearweek = min(max(Y_shifted.sale_yearweek),max_sale_yearweek-T_horizon_rolling+1)

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
    def split_timeseries_cv(self, n_splits, ID_Data):

        """

        ToDo: describe ...


        """


        # Range of sale yearweeks
        sale_yearweek_range = np.array(range(int(min(ID_Data.sale_yearweek)),
                                             int(max(ID_Data.sale_yearweek))))

        # Initailize rolling time series cross validation splits
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_folds = list()

        # For each fold
        for sale_yearweek_train_idx, sale_yearweek_val_idx in tscv.split(range(0,len(sale_yearweek_range))):

            idx_train = ((ID_Data.sale_yearweek >= min(sale_yearweek_range[sale_yearweek_train_idx])) & 
                         (ID_Data.sale_yearweek <= max(sale_yearweek_range[sale_yearweek_train_idx])))

            idx_val = ((ID_Data.sale_yearweek >= min(sale_yearweek_range[sale_yearweek_val_idx])) & 
                      (ID_Data.sale_yearweek <= max(sale_yearweek_range[sale_yearweek_val_idx])))

            cv_folds.append((idx_train, idx_val))

        # Return list of (train, val) split indices
        return cv_folds
    
    