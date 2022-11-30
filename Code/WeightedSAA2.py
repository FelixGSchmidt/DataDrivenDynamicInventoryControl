"""

...


"""


# Import utils
import numpy as np
import pandas as pd
import math
import time
import json
import pyreadr
import pickle
from joblib import dump, load
import os


# Import Gurobi
import gurobipy as gp
from gurobipy import GRB




    



#### Weighted SAA
class WeightedSAA:
    
    """
    
    Description ...
    
    """
        
    ### Init
    def __init__(self, **kwargs):
        
        # Set (default) params
        self.params = {
        
            'LogToConsole': kwargs.get('LogToConsole', 0),
            'Threads': kwargs.get('Threads', 1),
            'NonConvex': kwargs.get('NonConvex', 2),
            'PSDTol': kwargs.get('PSDTol', 0),
            'MIPGap': kwargs.get('MIPGap', 1e-3),
            'NumericFocus': kwargs.get('NumericFocus', 0),
            'obj_improvement': kwargs.get('obj_improvement', 1e-3),
            'obj_timeout_sec': kwargs.get('obj_timeout_sec', 3*60),
            'obj_timeout_max_sec': kwargs.get('obj_timeout_max_sec', 10*60),

            'K': kwargs.get('K', 100),
            'u': kwargs.get('u', 0.5),
            'h': kwargs.get('h', 1),
            'b': kwargs.get('b', 9),
            
            'epsilon': kwargs.get('epsilon', 0)
        
        }
        
        
    ### Function to set params
    def set_params(self, **kwargs):
        
        for item in kwargs:
            
            self.params[item] = kwargs[item]
            
        
    ### Function to get params
    def get_params(self):
        
        return self.params
        
 
        
        
        
        
    ### Function to create and set up the model
    def create(self, I, d, w, **kwargs):

        """
        
        This function initializes and sets up a tau-periods look-ahead control
        problem in MIP formulation with weighted SAA optimization to find the next
        decision to take (ordering quantity q in t=1)
        
        Arguments:
        
            I: starting inventory 
            d: demand samples i=1...n_samples
            w: sample weights i=1...n_samples
        
        Optional arguments:
        
            K: fixed cost of ordering
            u: per unit cost of ordering
            h: per unit cost of inventory holding
            b: per unit cost of demand backlogging
        
            epsilon: uncertainty threshold
            
            q_ub: upper bound of ordering quantity
        

        """

        # Set params
        self.set_params(**kwargs)
     
        # Length of look-ahead horizon (tau+1)
        n_periods = d.shape[1] if len(d.shape)==2 else 1
        
        # Number of demand samples
        n_samples = d.shape[0]
        
        # Number of model constraints (per demand sample i and period t)
        n_constraints = 5
        
        # Cost params
        K = self.params['K']
        u = self.params['u']
        h = self.params['h']
        b = self.params['b']

        # Limits
        q_ub = kwargs['q_ub'] if 'q_ub' in kwargs else math.ceil(max(np.max(d, axis=0)) * n_periods)
         
        ## Constraint coefficients
        
        # LHS constraint coefficient matrix A[t,s,m] with dim = tau x tau x n_constraints where A[t,s,m]==0 for s > t
        A = np.array([np.array([(np.array([-1,0,1,h,-b])
                                 if s==t
                                 else np.array([0,0,0,h,-b]))
                                if s<=t
                                else np.array([0,0,0,0,0])
                                for s in range(n_periods)])
                      for t in range(n_periods)])

        # LHS constraint coefficients B[t,s,m] with dim = tau x tau x n_constraints where B[t,s,m]==0 for s > t
        B = np.array([np.array([np.array([0,0,0,-h,b])
                                if s<=t
                                else np.array([0,0,0,0,0])
                                for s in range(n_periods)])
                      for t in range(n_periods)])

        # LHS constraint coefficients C[t,s,m] with dim = tau x tau x n_constraints where C[t,s,m]==0 for s <> t
        C = np.array([np.array([np.array([0,0,0,-1,-1])
                                if s==t
                                else np.array([0,0,0,0,0])
                                for s in range(n_periods)])
                      for t in range(n_periods)])

        # LHS constraint coefficients D[t,s,m] with dim = tau x tau x n_constraints where D[t,s,m]==0 for s <> t
        D = np.array([np.array([np.array([0,0,-1,0,0])
                                if s==t
                                else np.array([0,0,0,0,0])
                                for s in range(n_periods)])
                      for t in range(n_periods)])
        
        # LHS constraint coefficients E[t,s,m] with dim = tau x tau x n_constraints where E[t,s,m]==0 for s <> t
        E = np.array([np.array([np.array([0,-1,0,0,0])
                                if s==t
                                else np.array([0,0,0,0,0])
                                for s in range(n_periods)])
                      for t in range(n_periods)])

        # RHS constraint coefficients f[t,m] with dim = tau x n_constraints 
        f = np.array([np.array([0,0,0,-h*I,b*I])
                      for t in range(n_periods)])
   
        ## Create model
        self.m = gp.Model()

        # Primary decision variable (ordering quantity for each t)
        q = self.m.addVars(n_periods, vtype='I', lb=0, ub=q_ub, name='q')
            
        # Audliary decision variable (for fixed cost of ordering for each t)
        z = self.m.addVars(n_periods, vtype='B', name='z') 

        # Audliary decision variable (for cost of inventory holding or demand backlogging for each t and sample i)
        y_i = self.m.addVars(n_samples, n_periods, vtype='C', name='y_i') 

        ## Constraints   

        """

        Constraints (for each t=1...tau, i=1...n_samples, m=1...n_constraints):
        
            A*q + B*d + C*y_i + z*D*q + E*z <= f 
        
        """ 
             
        CONS = self.m.addConstrs(

            # A * q
            gp.quicksum(A[t,s,m]*q[s] for s in range(n_periods)) +

            # B * d
            gp.quicksum(B[t,s,m]*d[i,s] for s in range(n_periods)) +

            # C * y_i
            gp.quicksum(C[t,s,m]*y_i[i,s] for s in range(n_periods)) +

            # z * D * q
            gp.quicksum(z[s]*D[t,s,m]*q[s] for s in range(n_periods)) +
            
            # E * z
            gp.quicksum(E[t,s,m]*z[s] for s in range(n_periods))
            
            <= f[t,m]

            for m in range(n_constraints)
            for t in range(n_periods)
            for i in range(n_samples) 
        )
        
        

        ## Objective 
        OBJ = self.m.setObjective(

            # Weighted sum
            gp.quicksum(

                # i'th weight
                w[i] * (                                         

                    # u * q
                    gp.quicksum(u*q[t] for t in range(n_periods)) + 

                    # K * z
                    gp.quicksum(K*z[t] for t in range(n_periods)) + 

                    # y_i
                    gp.quicksum(y_i[i,t] for t in range(n_periods)) 


                ) for i in range(n_samples)),        

            # min
            GRB.MINIMIZE
        )

        # Store n periods
        self.n_periods = n_periods
        
        
        

    
    #### Function dump model
    def dump(self):
        
        self.m = None

        
    #### Function to optimize model
    def optimize(self, **kwargs):
        
        """
        
        Optional arguments:
        
            obj_improvement
            obj_timeout_sec
            obj_timeout_max_sec
        
        
        """
        
        # Set params
        self.set_params(**kwargs)         
            
        self.m.setParam('LogToConsole', self.params['LogToConsole'])
        self.m.setParam('Threads', self.params['Threads'])
        self.m.setParam('NonConvex', self.params['NonConvex'])
        self.m.setParam('PSDTol', self.params['PSDTol'])
        self.m.setParam('MIPGap', self.params['MIPGap'])
        self.m.setParam('NumericFocus', self.params['NumericFocus'])  
            
        ## Callback on solver time and objective improvement
        def cb(model, where):
            
 
            # MIP node
            if where == GRB.Callback.MIPNODE:

                # Get current incumbent objective
                objbst = model.cbGet(GRB.Callback.MIPNODE_OBJBST)   
                
                # Get current soluction count
                solcnt = model.cbGet(GRB.Callback.MIPNODE_SOLCNT)
                
                # If objective improved sufficiently
                if abs(objbst - model._cur_obj) > abs(model._cur_obj * self.params['obj_improvement']):

                    # Update incumbent and time
                    model._cur_obj = objbst
                    model._time = time.time()
                 
                # Terminate if objective has not improved sufficiently in 'obj_timeout_sec' seconds ...
                if time.time() - model._time > self.params['obj_timeout_sec']:        
                    
                    # ... and at least one soluction has been found
                    if solcnt > 0:
                        model.terminate()
                        
                    # ... or max sec have passed
                    elif time.time() - model._time > self.params['obj_timeout_max_sec']:
                        model.terminate()
                                               

            
        # Last updated objective and time
        self.m._cur_obj = float('inf')
        self.m._time = time.time() 

        # Optimize
        self.m.optimize(callback=cb)     
        
        
        ## Solution
        if self.m.SolCount > 0:
        
            # Objective value
            v_opt = self.m.objVal

            # Ordering quantities
            q_hat = [var.xn for var in self.m.getVars() if 'q' in var.VarName]
        
        else:
            
            q_hat = [np.nan]
            
        
        # Solution meta data
        status = self.m.status
        solutions = self.m.SolCount
        gap = self.m.MIPGap
        
                    
        # return decisions
        return q_hat, status, solutions, gap
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
#### Robust Weighted SAA
class RobustWeightedSAA:
    
    """
    
    Description ...
    
    """
        
    ### Init
    def __init__(self, **kwargs):
        
        # Set (default) params
        self.params = {
        
            'LogToConsole': kwargs.get('LogToConsole', 0),
            'Threads': kwargs.get('Threads', 1),
            'NonConvex': kwargs.get('NonConvex', 2),
            'PSDTol': kwargs.get('PSDTol', 0),
            'MIPGap': kwargs.get('MIPGap', 1e-3),
            'NumericFocus': kwargs.get('NumericFocus', 0),
            'obj_improvement': kwargs.get('obj_improvement', 1e-3),
            'obj_timeout_sec': kwargs.get('obj_timeout_sec', 3*60),
            'obj_timeout_max_sec': kwargs.get('obj_timeout_max_sec', 10*60),

            'K': kwargs.get('K', 100),
            'u': kwargs.get('u', 0.5),
            'h': kwargs.get('h', 1),
            'b': kwargs.get('b', 9),
            
            'epsilon': kwargs.get('epsilon', 0)
        
        }
        
    ### Function to set params
    def set_params(self, **kwargs):
        
        for item in kwargs:
            
            self.params[item] = kwargs[item]
            
        
    ### Function to get params
    def get_params(self):
        
        return self.params
               
        
        
        
    ### Function to create and set up the model
    def create(self, I, d, w, **kwargs):

        """
        
        This function initializes and sets up a tau-periods look-ahead control
        problem in MIP formulation with weighted SAA optimization to find the next
        decision to take (ordering quantity q in t=1)
        
        Arguments:
        
            I: starting inventory 
            d: demand samples i=1...n_samples
            w: sample weights i=1...n_samples
        
        Optional arguments:
        
            K: fixed cost of ordering
            u: per unit cost of ordering
            h: per unit cost of inventory holding
            b: per unit cost of demand backlogging
        
            epsilon: uncertainty threshold
            
            q_ub: upper bound of ordering quantity
        

        """

        # Set params
        self.set_params(**kwargs)
     
        # Length of look-ahead horizon (tau+1)
        n_periods = d.shape[1] if len(d.shape)==2 else 1
        
        # Number of demand samples
        n_samples = d.shape[0]
        
        # Number of model constraints (per demand sample i and period t)
        n_constraints = 5
        
        # Cost params
        K = self.params['K']
        u = self.params['u']
        h = self.params['h']
        b = self.params['b']
        
        # epsilon
        epsilon = self.params['epsilon']
        
        # Limits
        q_ub = kwargs['q_ub'] if 'q_ub' in kwargs else math.ceil((max(np.max(d, axis=0)) + epsilon) * n_periods)
            
        
        ## Constraint coefficients
        
        # LHS constraint coefficient matrix A[t,s,m] with dim = tau x tau x n_constraints where A[t,s,m]==0 for s > t
        A = np.array([np.array([(np.array([-1,0,1,h,-b])
                                 if s==t
                                 else np.array([0,0,0,h,-b]))
                                if s<=t
                                else np.array([0,0,0,0,0])
                                for s in range(n_periods)])
                      for t in range(n_periods)])

        # LHS constraint coefficients B[t,s,m] with dim = tau x tau x n_constraints where B[t,s,m]==0 for s > t
        B = np.array([np.array([np.array([0,0,0,-h,b])
                                if s<=t
                                else np.array([0,0,0,0,0])
                                for s in range(n_periods)])
                      for t in range(n_periods)])

        # LHS constraint coefficients C[t,s,m] with dim = tau x tau x n_constraints where C[t,s,m]==0 for s <> t
        C = np.array([np.array([np.array([0,0,0,-1,-1])
                                if s==t
                                else np.array([0,0,0,0,0])
                                for s in range(n_periods)])
                      for t in range(n_periods)])

        # LHS constraint coefficients D[t,s,m] with dim = tau x tau x n_constraints where D[t,s,m]==0 for s <> t
        D = np.array([np.array([np.array([0,0,-1,0,0])
                                if s==t
                                else np.array([0,0,0,0,0])
                                for s in range(n_periods)])
                      for t in range(n_periods)])
        
        # LHS constraint coefficients E[t,s,m] with dim = tau x tau x n_constraints where E[t,s,m]==0 for s <> t
        E = np.array([np.array([np.array([0,-1,0,0,0])
                                if s==t
                                else np.array([0,0,0,0,0])
                                for s in range(n_periods)])
                      for t in range(n_periods)])

        # RHS constraint coefficients f[t,m] with dim = tau x n_constraints 
        f = np.array([np.array([0,0,0,-h*I,b*I])
                      for t in range(n_periods)])
   
        ## Create model
        self.m = gp.Model()
        
        # Set model meta params
        self.m.setParam('LogToConsole', self.params['LogToConsole'])
        self.m.setParam('Threads', self.params['Threads'])
        self.m.setParam('NonConvex', self.params['NonConvex'])
        self.m.setParam('PSDTol', self.params['PSDTol'])
        self.m.setParam('MIPGap', self.params['MIPGap'])  
        self.m.setParam('NumericFocus', self.params['NumericFocus'])  
    
    
        # Primary decision variable (ordering quantity for each t)
        q = self.m.addVars(n_periods, vtype='I', lb=0, ub=q_ub, name='q')
            
        # Audliary decision variable (for fixed cost of ordering for each t)
        z = self.m.addVars(n_periods, vtype='B', name='z') 

        # Audliary decision variable (for cost of inventory holding or demand backlogging for each t and sample i)
        y_i = self.m.addVars(n_samples, n_periods, vtype='C', name='y_i') 

        # Audliary decision variable (from reformulation of robust constraints)
        Lambda_i = self.m.addVars(n_samples, n_periods, n_periods, n_constraints, 
                                  vtype='C', lb=0, name='Lambda_i')
        
        # Helper for multiplication
        rhs = self.m.addVars(n_samples, n_periods, n_constraints, vtype='C', name='rhs')
    
       

        ## Constraints   

        """

        Constraints (for each t=1...tau, i=1...n_samples, m=1...n_constraints):
        
            A*q + B*d + C*y_i + z*D*q + E*z + Lambda_i*d + epsilon*||B + Lambda_i||_2 <= f
        
        Reforumulation (|| ... ||_2 is the Euclidean norm):
        
            epsilon*||B + Lambda_i||_2 <= f - A*q - B*d - C*y_i - z*D*q - E*z - Lambda_i*d
        
            epsilon*((B + Lambda_i)*(B + Lambda_i))^(1/2) <= f - A*q - B*d - C*y_i - z*D*q - E*z - Lambda_i*d
        
        This can be written as two constraints that have to be met at the same time as
        epsilon*||B + Lambda_i|| >= 0 by definition:
        
            (1) A*q + B*d + C*y_i + z*D*q + E*z + Lambda_i*d <= f
        
            (2) epsilon^2*((B + Lambda_i)*(B + Lambda_i)) <= (f - A*q - B*d - C*y_i - z*D*q - E*z -Lambda_i*d)^2

        Let us also introduce a helper costraint with an additional audliary decision variable called rhs
        
            (0) rhs = f - A*q - B*d - C*y_i - z*D*q - E*z - Lambda_i*d
        
        With this, we have 3 constraints
        
            (C0) rhs = f - A*q - B*d - C*y_i - z*D*q - E*z - Lambda_i*d
        
            (C1) A*q + B*d + C*y_i + z*D*q + E*z + Lambda_i*d <= f
        
            (C2) epsilon^2*((B + Lambda_i)*(B + Lambda_i)) <= (f - A*q - B*d - C*y_i - z*D*q - E*z - Lambda_i*d)^2
  
        
        """ 
    
    
        # C0
        C0 = self.m.addConstrs(

            
            rhs[i,t,m]
            
            == 
            
            # f
            f[t,m] -
            
            # A * q
            gp.quicksum(A[t,s,m]*q[s] for s in range(n_periods)) -

            # B * d
            gp.quicksum(B[t,s,m]*d[i,s] for s in range(n_periods)) -

            # C * y_i
            gp.quicksum(C[t,s,m]*y_i[i,s] for s in range(n_periods)) -

            # z * D * q
            gp.quicksum(z[s]*D[t,s,m]*q[s] for s in range(n_periods)) -
            
            # E * z
            gp.quicksum(E[t,s,m]*z[s] for s in range(n_periods)) -
            
            # Lambda_i * d
            gp.quicksum(Lambda_i[i,t,s,m]*d[i,s] for s in range(n_periods))

            for m in range(n_constraints)
            for t in range(n_periods)
            for i in range(n_samples) 
        )
        
        
        
        # C1 
        C1 = self.m.addConstrs(

            0 <= rhs[i,t,m]

            for m in range(n_constraints)
            for t in range(n_periods)
            for i in range(n_samples) 
        )
        
        
        # C2
        C2 = self.m.addConstrs(
            
            epsilon**2 * gp.quicksum((B[t,s,m] + Lambda_i[i,t,s,m])*(B[t,s,m] + Lambda_i[i,t,s,m])
                                     for s in range(n_periods))
            
            <= rhs[i,t,m] * rhs[i,t,m]
            
            for m in range(n_constraints)
            for t in range(n_periods)
            for i in range(n_samples) 
        )
        

        ## Objective 
        OBJ = self.m.setObjective(

            # Weighted sum
            gp.quicksum(

                # i'th weight
                w[i] * (                                         

                    # u * q
                    gp.quicksum(u*q[t] for t in range(n_periods)) + 

                    # K * z
                    gp.quicksum(K*z[t] for t in range(n_periods)) + 

                    # y_i
                    gp.quicksum(y_i[i,t] for t in range(n_periods)) 


                ) for i in range(n_samples)),        

            # min
            GRB.MINIMIZE
        )

        # Store n periods
        self.n_periods = n_periods
        
        
        

    
    #### Function to dump model
    def dump(self):
        
        self.m = None

        
    #### Function to optimize model
    def optimize(self, **kwargs):
        
        """
        
        Optional arguments:
        
            obj_improvement
            obj_timeout_sec
            obj_timeout_max_sec
        
        
        """
        
        
        # Set params
        self.set_params(**kwargs)           
            
        # Set MIP start
        for var in self.m.getVars():
            if 'q' in var.VarName:
                for period in range(n_periods):
                    var[period].Start = 0

                
                
        ## Callback on solver time and objective improvement
        def cb(model, where):
            
 
            # MIP node
            if where == GRB.Callback.MIPNODE:

                # Get current incumbent objective
                objbst = model.cbGet(GRB.Callback.MIPNODE_OBJBST)   
                
                # Get current soluction count
                solcnt = model.cbGet(GRB.Callback.MIPNODE_SOLCNT)
                
                # If objective improved sufficiently
                if abs(objbst - model._cur_obj) > abs(model._cur_obj * self.params['obj_improvement']):

                    # Update incumbent and time
                    model._cur_obj = objbst
                    model._time = time.time()
                 
                # Terminate if objective has not improved sufficiently in 'obj_timeout_sec' seconds ...
                if time.time() - model._time > self.params['obj_timeout_sec']:        
                    
                    # ... and at least one soluction has been found
                    if solcnt > 0:
                        model.terminate()
                        
                    # ... or max sec have passed
                    elif time.time() - model._time > self.params['obj_timeout_max_sec']:
                        model.terminate()
            
            
        # Last updated objective and time
        self.m._cur_obj = float('inf')
        self.m._time = time.time() 

        # Optimize
        self.m.optimize(callback=cb)      
        
        ## Solution
        if self.m.SolCount > 0:
        
            # Objective value
            v_opt = self.m.objVal

            # Ordering quantities
            q_hat = [var.xn for var in self.m.getVars() if 'q' in var.VarName]
        
        else:
            
            q_hat = [np.nan]
            
        
        # Solution meta data
        status = self.m.status
        solutions = self.m.SolCount
        gap = self.m.MIPGap
        
                    
        # return decisions
        return q_hat, status, solutions, gap
    
    
    
    
    

    
    
    
#### Rolling Horizon Optimization
class RollingHorizonOptimization:

    """
    
    This class provides the functionality to run a Rolling Horizon Optimization experiment over a planning horizon
    given weights, samples, actuals and the optimization approach incl. its parameters
    
    """
        
    ### Init
    def __init__(self, **kwargs):
        
        # Set (default) params
        self.params = {
        
            **kwargs
        
        }
        
    ### Function to set params
    def set_params(self, **kwargs):
        
        for item in kwargs:
            
            self.params[item] = kwargs[item]
            
        
    ### Function to get params
    def get_params(self):
        
        return self.params

    
    ### Function to run experiment over planing horizon t=1...T
    def run(self, y, ids_train, ids_test, tau, wsaamodel, **kwargs):

        """

        Description ...
        
        Models: If weights are given, performs Weighted SAA, else performs SAA. If epsilons are given, 
        performs robust extension.



        # Arguments

            y: demand data - np.array of shape (n_samples, n_periods)
            ids_train: list of selector series (True/False of length n_samples) - list with lengths of the test horizon
            ids_test: list of selector series (True/False of length n_samples) - list with lengths of the test horizon
            tau: n_periods rolling look-ahead horizon - int


        # Optional arguments 

            weights: list of weights (flat np.array of length ids_train of the t'th test horizon period) - list 
            with length of the test horizon
            epsilons: list of epsilons - list with length of the test horizon

            LogToConsole: Gurobi meta param (0: high level status, 1: Gurobi log)
            Threads: Gurobi meta param
            NonConvex: Gurobi meta param 
            PSDTol: Gurobi meta param 
            MIPGap: Gurobi meta param 
            obj_improvement: Gurobi meta param 
            obj_timeout_sec: Gurobi meta param 

            K: Fixed cost of ordering
            u: Per unit cost of ordering
            h: Per unit cost of inventory holding
            b: Per unit cost of demand backlogging
            
            I_current: starting inventory

        # Returns: DataFrame of length ids_test storing results for t=1...T

        """


        # Get planning horizon (T_horizon)
        T = len(ids_test)

        # Initialize starting inventory
        I_current = kwargs['I_current'] if 'I_current' in kwargs else 0

        # Initialize results lists
        I_t = []
        q_t = []
        Iq_t = []
        y_t = []
        Iqy_t = []
        
        defaulted_t = []
        status_t = []
        solutions_t = []
        gap_t = []

        exec_time_sec = []
        cpu_time_sec = []

        ## Iterate over planning horizon t=1...T
        for t in range(T):

            # Timer
            st_exec = time.time()
            st_cpu = time.process_time()      

            # Rolling look-ahead horizon tau (adjusted for remaining planning horizon in t)
            tau = min(tau,T-t)

            # Historical demands up to t
            d = y[ids_train[t],0:tau]

            # Weights for weighted SAA
            if 'weights' in kwargs:

                w = np.array(kwargs['weights'][t])    

            # Weights for SAA (all weights are 1/n_samples)
            else:

                w = np.array([np.repeat(1/d.shape[0],d.shape[0])]).flatten()

            # Summarize weights and samples
            d = d[list(w > 0)]
            w = w[list(w > 0)]

            df=pd.DataFrame(data=np.hstack(
                (w.reshape(w.shape[0],1), d)), 
                    columns=[-1] + list(range(0,tau))).groupby(
                list(range(0,tau))).agg(
                    w = (-1, np.sum)).reset_index()

            d = np.array(df[list(range(0,tau))])
            w = np.array([df.w]).flatten()


            # Robust
            epsilon = kwargs['epsilons'][t] if 'epsilons' in kwargs else 0


            ## Optimization

            # Parameters
            params = {

                **kwargs, 'epsilon': epsilon

            } 

            
            # If inventory is nan (no further optimization)
            if np.isnan(I_current):
                
                # Store results
                I_t = I_t + [np.nan] # inventory before ordering
                q_t = q_t + [np.nan] # ordering quantity
                Iq_t = Iq_t + [np.nan] # inventory after ordering
                y_t = y_t + [np.nan] # demand
                Iqy_t = Iqy_t + [np.nan] # inventory after demand

                defaulted_t = defaulted_t + [""]
                status_t = status_t + [""]
                solutions_t = solutions_t + [""]
                gap_t = gap_t + [""]
                
                # Update inventory
                I_current = I_current + np.nan

            # If inventory is not nan
            else:

                # Create model 
                wsaamodel.create(I=I_current, d=d, w=w, **params)

                # Optimize and get decisions
                q_hat, status, solutions, gap = wsaamodel.optimize()    
                
                # Flag that did not default to WeightedSAA
                defaulted = False   
                    
                current_model_params = wsaamodel.get_params()
                current_model_type = str(type(wsaamodel))
                
                # Delete model
                wsaamodel.dump()
                
                # If no solution is found and current model is a robust extension, default to WeightedSAA
                if not solutions > 0 and 'Robust' in current_model_type:
                                
                    # Default to WeightedSAA without robust extension
                    wsaamodel_default = WeightedSAA(**current_model_params)
                    
                    # Create model 
                    wsaamodel_default.create(I=I_current, d=d, w=w, **params)

                    # Optimize and get decisions
                    q_hat, status, solutions, gap = wsaamodel_default.optimize()    
                    
                    # Flag that model defaulted to WeightedSAA
                    defaulted = True                
                        
                    # Delete model
                    wsaamodel_default.dump()
                    

                # Store results
                I_t = I_t + [I_current] # inventory before ordering
                q_t = q_t + [np.around(q_hat[0],0).item()] # ordering quantity
                Iq_t = Iq_t + [I_current + np.around(q_hat[0],0).item()] # inventory after ordering
                y_t = y_t + [y[ids_test[t],0].item()] # demand
                Iqy_t = Iqy_t + [I_current + np.around(q_hat[0],0).item() - y[ids_test[t],0].item()] # inventory after demand

                defaulted_t = defaulted_t + [defaulted]
                status_t = status_t + [status]
                solutions_t = solutions_t + [solutions]
                gap_t = gap_t + [gap]
        
                # Update inventory
                I_current = I_current + np.around(q_hat[0],0).item() - y[ids_test[t],0].item()
                 

            # Timing
            exec_time_sec = exec_time_sec+[time.time()-st_exec]  
            cpu_time_sec = cpu_time_sec+[time.process_time()-st_cpu]  
            

        ## Evaluate results

        # Store results
        results = pd.DataFrame({
            'K': np.repeat(kwargs['K'] if 'K' in kwargs else None,T),
            'u': np.repeat(kwargs['u'] if 'K' in kwargs else None,T),
            'h': np.repeat(kwargs['h'] if 'K' in kwargs else None,T),
            'b': np.repeat(kwargs['b'] if 'K' in kwargs else None,T),
            't': np.array(range(1,T+1)),
            'I': I_t,
            'q': q_t,
            'I_q': Iq_t,
            'y': y_t,
            'I_q_y': Iqy_t,
            'c_o': 0,
            'c_s': 0,
            'cost': 0,
            'defaulted': defaulted_t,
            'status': status_t,
            'solutions': solutions_t,
            'gap': gap_t,
            'exec_time_sec': exec_time_sec,
            'cpu_time_sec': cpu_time_sec
        })

        # Calculate resulting costs
        results.c_o = (results.q>0) * results.K + results.q * results.u
        results.c_s = (results.I_q_y>0) * results.I_q_y * results.h + (results.I_q_y<0) * (-results.I_q_y) * results.b
        results.cost = results.c_o + results.c_s

        ## Return
        return results
    
    

    