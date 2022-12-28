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
import copy


# Import Gurobi
import gurobipy as gp
from gurobipy import GRB




    


    


#### Weighted SAA
class WeightedSAA:
    
    """
    
    Description ...
    
    """
        
    ### Init
    def __init__(self, K=100, u=0.5, h=1, b=9, LogToConsole=0, Threads=1, NonConvex=2, PSDTol=0, MIPGap=1e-3, 
                 NumericFocus=0, obj_improvement=1e-3, obj_timeout_sec=3*60, obj_timeout_max_sec=10*60, **kwargs):

        """
        
        ...
        
        Arguments:
        
            K (default=100): Fixed cost of ordering
            u (default=0.5): Per unit cost of ordering
            h (default=1): Per unit cost of inventory holding
            b (default=9): Per unit cost of demand backlogging

            LogToConsole (default=0): ...
            Threads (default=1): ...
            NonConvex (default=2): ...
            PSDTol (default=0): ...
            MIPGap (default=1e-3): ...
            NumericFocus (default=0): ...
            
            obj_improvement (default=1e-3): ...
            obj_timeout_sec (default=3*60): ...
            obj_timeout_max_sec (default=10*60): ...
            
        Further key word arguments (kwargs): ignored

            
        """

        # Set params
        self.params = {
            
            'K': K,
            'u': u,
            'h': h,
            'b': b,
            
            'LogToConsole': LogToConsole,
            'Threads': Threads,
            'NonConvex': NonConvex,
            'PSDTol': PSDTol,
            'MIPGap': MIPGap,
            'NumericFocus': NumericFocus,
            
            'obj_improvement': obj_improvement,
            'obj_timeout_sec': obj_timeout_sec,
            'obj_timeout_max_sec': obj_timeout_max_sec
        
        }
        

    ### Function to set params
    def set_params(self, **kwargs):
        
        # Update all items that match an existing key
        self.params.update((k, kwargs[k]) for k in set(kwargs).intersection(self.params))
            
        
    ### Function to get params
    def get_params(self):
        
        return self.params
        
    
        
    ### Function to create and set up the model
    def create(self, d, w, I=0, **kwargs):

        """
        
        This function initializes and sets up a (tau+1)-periods rolling look-ahead control
        problem in MIP formulation with Weighted SAA optimization to find the next
        decision to take (ordering quantity q of the current, first period)
        
        Arguments:
            
            d: demand samples i=1...n_samples
            w: sample weights i=1...n_samples
            I: starting inventory 

        Further key word arguments (kwargs): passed to set_params() to update (valid) paramaters, e.g., cost parameters K, u, h, b.


        """

        # Set params
        self.set_params(**kwargs)
     
        # Length of rolling look-ahead horizon
        n_periods = d.shape[1] if d.ndim>1 else 1
        
        # Number of demand samples
        n_samples = d.shape[0]
        
        # Number of model constraints (per demand sample i and period t)
        n_constraints = 2
        
        # Cost params
        K = self.params['K']
        u = self.params['u']
        h = self.params['h']
        b = self.params['b']

            
        ## Constraint coefficients

        
        # LHS constraint coefficient matrix A1[t,s,m] with dim = tau x tau x n_constraints where A1[t,s,m]==0 for s > t
        A1 = np.array([np.array([np.array([h,-b])
                                if s<=t
                                else np.array([0,0])
                                for s in range(n_periods)])
                      for t in range(n_periods)])
        

        # LHS constraint coefficients A2[t,s,m] with dim = tau x tau x n_constraints where A2[t,s,m]==0 for s > t
        A2 = np.array([np.array([np.array([-h,b])
                                if s<=t
                                else np.array([0,0])
                                for s in range(n_periods)])
                      for t in range(n_periods)])
        
         # LHS constraint coefficients A3[t,s,m] with dim = tau x tau x n_constraints where A3[t,s,m]==0 for s <> t
        A3 = np.array([np.array([np.array([-1,-1])
                                if s==t
                                else np.array([0,0])
                                for s in range(n_periods)])
                      for t in range(n_periods)])

        # RHS constraint coefficients b[t,m] with dim = tau x n_constraints 
        B = np.array([np.array([-h*I,b*I])
                      for t in range(n_periods)])
        
        ## Create model
        if self.params['LogToConsole']==0:
            
            env = gp.Env(empty=True)
            env.setParam('OutputFlag',0)
            env.start()
            self.m = gp.Model(env=env)
            
        else:
            
            self.m = gp.Model()

        # Primary decision variable (ordering quantity for each t)
        q = self.m.addVars(n_periods, vtype='I', lb=0, name='q')
        
        # Auxiary decision variable (for fixed cost of ordering for each t)
        z = self.m.addVars(n_periods, vtype='B', name='z')        
        
        # Auxiary decision variable (for cost of inventory holding or demand backlogging for each t and sample i)
        s_i = self.m.addVars(n_samples, n_periods, vtype='C', name='s_i') 

        ## Constraints   

        """

        Constraints (for each m=1...n_constraints, t=1...tau, i=1...n_samples). For the implementation using Gurobi, we do not
        use a 'big M' formulation to model fixed cost of ordering. Thus, the mathematical formulation is slightly different than
        provided in the paper.
        
            A1*q + A2*d + A3*s_i <= B 
        
        """       
        
        # Fixed cost
        z_cons1 = self.m.addConstrs((z[t] == 0) >> (q[t] == 0) for t in range(n_periods))
        z_cons2 = self.m.addConstrs((z[t] == 1) >> (q[t] >= 1) for t in range(n_periods))
         
        # Remaining constraints
        cons = self.m.addConstrs(

            # A1 * q
            gp.quicksum(A1[t,s,m]*q[s] for s in range(n_periods)) +

            # A2 * d
            gp.quicksum(A2[t,s,m]*d[i,s] for s in range(n_periods)) +
            
            # A3 * s_i
            gp.quicksum(A3[t,s,m]*s_i[i,s] for s in range(n_periods)) 
            
            <= B[t,m]

            for m in range(n_constraints)
            for t in range(n_periods)
            for i in range(n_samples) 
        )
        
        

        ## Objective 
        obj = self.m.setObjective(

            # Weighted sum
            gp.quicksum(

                # i'th weight
                w[i] * (                                         

                    # u * q
                    gp.quicksum(u*q[t] for t in range(n_periods)) + 

                    # K * z
                    gp.quicksum(K*z[t] for t in range(n_periods)) + 

                    # s_i
                    gp.quicksum(s_i[i,t] for t in range(n_periods)) 


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
        
        ...
        
        Arguments: None
            
        Further key word arguments (kwargs): passed to update Gurobi meta params (other kwargs are ignored)

        Returns:
        
            q_hat: ...
            status: ...
            solutions: ...
            gap: ...
            
        """

        
        # Update gurobi meta params if provided
        gurobi_meta_params = {
            
            'LogToConsole': kwargs.get('LogToConsole', self.params['LogToConsole']),
            'Threads': kwargs.get('Threads', self.params['Threads']),
            'NonConvex': kwargs.get('NonConvex', self.params['NonConvex']),
            'PSDTol': kwargs.get('PSDTol', self.params['PSDTol']),
            'MIPGap': kwargs.get('MIPGap', self.params['MIPGap']),
            'NumericFocus': kwargs.get('NumericFocus', self.params['NumericFocus']),
            'obj_improvement': kwargs.get('obj_improvement', self.params['obj_improvement']),
            'obj_timeout_sec': kwargs.get('obj_timeout_sec', self.params['obj_timeout_sec']),
            'obj_timeout_max_sec': kwargs.get('obj_timeout_max_sec', self.params['obj_timeout_max_sec'])
        }
        
        self.set_params(**gurobi_meta_params)           
            
        # Set Gurobi meta params
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
    def __init__(self, K=100, u=0.5, h=1, b=9, epsilon=0, LogToConsole=0, Threads=1, NonConvex=2, PSDTol=0, MIPGap=1e-3, 
                 NumericFocus=0, obj_improvement=1e-3, obj_timeout_sec=3*60, obj_timeout_max_sec=10*60, **kwargs):

        """
        
        ...
        
        Arguments:
        
            K (default=100): Fixed cost of ordering
            u (default=0.5): Per unit cost of ordering
            h (default=1): Per unit cost of inventory holding
            b (default=9): Per unit cost of demand backlogging
            
            epsilon (default=0): Uncertainty set parameter

            LogToConsole (default=0): ...
            Threads (default=1): ...
            NonConvex (default=2): ...
            PSDTol (default=0): ...
            MIPGap (default=1e-3): ...
            NumericFocus (default=0): ...
            
            obj_improvement (default=1e-3): ...
            obj_timeout_sec (default=3*60): ...
            obj_timeout_max_sec (default=10*60): ...
            
        Further key word arguments (kwargs): ignored

            
        """

        # Set params
        self.params = {
            
            'K': K,
            'u': u,
            'h': h,
            'b': b,
            
            'epsilon': epsilon,
            
            'LogToConsole': LogToConsole,
            'Threads': Threads,
            'NonConvex': NonConvex,
            'PSDTol': PSDTol,
            'MIPGap': MIPGap,
            'NumericFocus': NumericFocus,
            
            'obj_improvement': obj_improvement,
            'obj_timeout_sec': obj_timeout_sec,
            'obj_timeout_max_sec': obj_timeout_max_sec
        
        }
        

    ### Function to set params
    def set_params(self, **kwargs):
        
        # Update all items that match an existing key
        self.params.update((k, kwargs[k]) for k in set(kwargs).intersection(self.params))
            
        
    ### Function to get params
    def get_params(self):
        
        return self.params
        

    
    ### Function to create and set up the model
    def create(self, d, w, I=0, **kwargs):

        """
        
        This function initializes and sets up a (tau+1)-periods rolling look-ahead control
        problem in MIP formulation with Robust Weighted SAA optimization to find the next
        decision to take (ordering quantity q of the current, first period)
        
        Arguments:
        
            d: demand samples i=1...n_samples
            w: sample weights i=1...n_samples
            I: starting inventory          
            
        Further key word arguments (kwargs): passed to set_params() to update (valid) paramaters, e.g., cost parameters K, u, h, b, 
        uncertainty set param epsilon


        """

        # Set params
        self.set_params(**kwargs)
     
        # Length of rolling look-ahead horizon
        n_periods = d.shape[1] if d.ndim>1 else 1
        
        # Number of demand samples
        n_samples = d.shape[0]
        
        # Number of model constraints (per demand sample i and period t)
        n_constraints = 2
        
        # Cost params
        K = self.params['K']
        u = self.params['u']
        h = self.params['h']
        b = self.params['b']
        
        # Uncertainty set parameter epsilon
        epsilon = self.params['epsilon']

        
        ## Constraint coefficients
        
        # LHS constraint coefficient matrix A1[t,s,m] with dim = tau x tau x n_constraints where A1[t,s,m]==0 for s > t
        A1 = np.array([np.array([np.array([h,-b])
                                if s<=t
                                else np.array([0,0])
                                for s in range(n_periods)])
                      for t in range(n_periods)])

        # LHS constraint coefficients A2[t,s,m] with dim = tau x tau x n_constraints where A2[t,s,m]==0 for s > t
        A2 = np.array([np.array([np.array([-h,b])
                                if s<=t
                                else np.array([0,0])
                                for s in range(n_periods)])
                      for t in range(n_periods)])

        # LHS constraint coefficients A3[t,s,m] with dim = tau x tau x n_constraints where A3[t,s,m]==0 for s != t
        A3 = np.array([np.array([np.array([-1,-1])
                                if s==t
                                else np.array([0,0])
                                for s in range(n_periods)])
                      for t in range(n_periods)])

        # RHS constraint coefficients f[t,m] with dim = tau x n_constraints 
        B = np.array([np.array([-h*I,b*I])
                      for t in range(n_periods)])
   


        ## Create model
        if self.params['LogToConsole']==0:
            
            env = gp.Env(empty=True)
            env.setParam('OutputFlag',0)
            env.start()
            self.m = gp.Model(env=env)
            
        else:
            
            self.m = gp.Model()
        
        # Primary decision variable (ordering quantity for each t)
        q = self.m.addVars(n_periods, vtype='I', lb=0, name='q')

        # Auxiary decision variable (for fixed cost of ordering for each t)
        z = self.m.addVars(n_periods, vtype='B', name='z') 

        # Auxiary decision variable (for cost of inventory holding or demand backlogging for each t and sample i)
        s_i = self.m.addVars(n_samples, n_periods, vtype='C', name='s_i') 

        # Auxiary decision variable (from reformulation of robust constraints)
        Lambda_i = self.m.addVars(n_samples, n_periods, n_periods, n_constraints, 
                                  vtype='C', lb=0, name='Lambda_i')
        
        # Auxiary decision variable for norms
        norm_inner = self.m.addVars(n_samples, n_periods, n_periods, n_constraints, vtype='C', name='norm_inner')
        norm_inner_abs = self.m.addVars(n_samples, n_periods, n_periods, n_constraints, vtype='C', name='norm_inner')

        # Auxiary decision variable for norms
        norm = self.m.addVars(n_samples, n_periods, n_constraints, vtype='C', lb=0, name='norm')



        ## Constraints   

        """
        
        Constraints (for each m=1...n_constraints, t=1...tau, i=1...n_samples). For the implementation using Gurobi, we do not
        use a 'big M' formulation to model fixed cost of ordering. Thus, the mathematical formulation is slightly different than
        provided in the paper.
  
            A1*q + A2*d + A3*s_i + Lambda_i*d + epsilon*||A2 + Lambda_i||_1 <= B
                
        To calculate the l1 norm ||.|| we use further helper constraints.
  
        
        """        
        
        # Fixed cost
        z_cons1 = self.m.addConstrs((z[t] == 0) >> (q[t] == 0) for t in range(n_periods))
        z_cons2 = self.m.addConstrs((z[t] == 1) >> (q[t] >= 1) for t in range(n_periods))
        
        # Constraint to set inner part of the norm
        norm_inner_cons = self.m.addConstrs(

            norm_inner[i,t,s,m] == A2[t,s,m] + Lambda_i[i,t,s,m] 
            
            for s in range(n_periods)
            for m in range(n_constraints)
            for t in range(n_periods)
            for i in range(n_samples) 
        )

         # Constraint to set inner part of the norm as absolute values
        norm_inner_abs_cons = self.m.addConstrs(

            norm_inner_abs[i,t,s,m] == gp.abs_(norm_inner[i,t,s,m])
            
            for s in range(n_periods)
            for m in range(n_constraints)
            for t in range(n_periods)
            for i in range(n_samples) 
        )

        
        # Constraint to get actual value of the norm
        norm_cons =  self.m.addConstrs(

            norm[i,t,m] == gp.quicksum(norm_inner_abs[i,t,s,m] for s in range(n_periods))

            for m in range(n_constraints)
            for t in range(n_periods)
            for i in range(n_samples) 
        )

        # Actual constraints
        cons = self.m.addConstrs(

            # A1 * q
            gp.quicksum(A1[t,s,m]*q[s] for s in range(n_periods)) +

            # A2 * d
            gp.quicksum(A2[t,s,m]*d[i,s] for s in range(n_periods)) +

            # A3 * s_i
            gp.quicksum(A3[t,s,m]*s_i[i,s] for s in range(n_periods)) +

            # Lambda_i * d
            gp.quicksum(Lambda_i[i,t,s,m]*d[i,s] for s in range(n_periods)) +

            # epsilon*||A2 + Lambda_i||_1
            epsilon * norm[i,t,m]

            <= B[t,m]

            for m in range(n_constraints)
            for t in range(n_periods)
            for i in range(n_samples) 
        )

        

        ## Objective 
        obj = self.m.setObjective(

            # Weighted sum
            gp.quicksum(

                # i'th weight
                w[i] * (                                         

                    # u * q
                    gp.quicksum(u*q[t] for t in range(n_periods)) + 

                    # K * z
                    gp.quicksum(K*z[t] for t in range(n_periods)) + 

                    # s_i
                    gp.quicksum(s_i[i,t] for t in range(n_periods)) 


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
        
        ...
        
        Arguments: None
            
        Further key word arguments (kwargs): passed to update Gurobi meta params (other kwargs are ignored)

        Returns:
        
            q_hat: ...
            status: ...
            solutions: ...
            gap: ...
            
        """

        
        # Update gurobi meta params if provided
        gurobi_meta_params = {
            
            'LogToConsole': kwargs.get('LogToConsole', self.params['LogToConsole']),
            'Threads': kwargs.get('Threads', self.params['Threads']),
            'NonConvex': kwargs.get('NonConvex', self.params['NonConvex']),
            'PSDTol': kwargs.get('PSDTol', self.params['PSDTol']),
            'MIPGap': kwargs.get('MIPGap', self.params['MIPGap']),
            'NumericFocus': kwargs.get('NumericFocus', self.params['NumericFocus']),
            'obj_improvement': kwargs.get('obj_improvement', self.params['obj_improvement']),
            'obj_timeout_sec': kwargs.get('obj_timeout_sec', self.params['obj_timeout_sec']),
            'obj_timeout_max_sec': kwargs.get('obj_timeout_max_sec', self.params['obj_timeout_max_sec'])
        }
        
        self.set_params(**gurobi_meta_params)           
            
        # Set Gurobi meta params
        self.m.setParam('LogToConsole', self.params['LogToConsole'])
        self.m.setParam('Threads', self.params['Threads'])
        self.m.setParam('NonConvex', self.params['NonConvex'])
        self.m.setParam('PSDTol', self.params['PSDTol'])
        self.m.setParam('MIPGap', self.params['MIPGap'])
        self.m.setParam('NumericFocus', self.params['NumericFocus'])  
 
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
    
    

#### Robust Weighted SAA with Linear Decision Rules
class RobustWeightedSAALDR:
    
    """
    
    Description ...
    
    """
        
    ### Init
    def __init__(self, K=100, u=0.5, h=1, b=9, epsilon=0, LogToConsole=0, Threads=1, NonConvex=2, PSDTol=0, MIPGap=1e-3, 
                 NumericFocus=0, obj_improvement=1e-3, obj_timeout_sec=3*60, obj_timeout_max_sec=10*60, **kwargs):

        """
        
        ...
        
        Arguments:
        
            K (default=100): Fixed cost of ordering
            u (default=0.5): Per unit cost of ordering
            h (default=1): Per unit cost of inventory holding
            b (default=9): Per unit cost of demand backlogging
            
            epsilon (default=0): Uncertainty set parameter

            LogToConsole (default=0): ...
            Threads (default=1): ...
            NonConvex (default=2): ...
            PSDTol (default=0): ...
            MIPGap (default=1e-3): ...
            NumericFocus (default=0): ...
            
            obj_improvement (default=1e-3): ...
            obj_timeout_sec (default=3*60): ...
            obj_timeout_max_sec (default=10*60): ...
            
        Further key word arguments (kwargs): ignored

            
        """

        # Set params
        self.params = {
            
            'K': K,
            'u': u,
            'h': h,
            'b': b,
            
            'epsilon': epsilon,
            
            'LogToConsole': LogToConsole,
            'Threads': Threads,
            'NonConvex': NonConvex,
            'PSDTol': PSDTol,
            'MIPGap': MIPGap,
            'NumericFocus': NumericFocus,
            
            'obj_improvement': obj_improvement,
            'obj_timeout_sec': obj_timeout_sec,
            'obj_timeout_max_sec': obj_timeout_max_sec
        
        }
        

    ### Function to set params
    def set_params(self, **kwargs):
        
        # Update all items that match an existing key
        self.params.update((k, kwargs[k]) for k in set(kwargs).intersection(self.params))
            
        
    ### Function to get params
    def get_params(self):
        
        return self.params
        

    
    ### Function to create and set up the model
    def create(self, d, w, I=0, **kwargs):

        """
        
        This function initializes and sets up a (tau+1)-periods rolling look-ahead control
        problem in MIP formulation with Robust Weighted SAA optimization to find the next
        decision to take (ordering quantity q of the current, first period)
        
        Arguments:
        
            d: demand samples i=1...n_samples
            w: sample weights i=1...n_samples
            I: starting inventory          
            
        Further key word arguments (kwargs): passed to set_params() to update (valid) paramaters, e.g., cost parameters K, u, h, b, 
        uncertainty set param epsilon


        """

        # Set params
        self.set_params(**kwargs)
     
        # Length of rolling look-ahead horizon
        n_periods = d.shape[1] if d.ndim>1 else 1
        
        # Number of demand samples
        n_samples = d.shape[0]
        
        # Number of model constraints (per demand sample i and period t)
        n_constraints = 2
        
        # Cost params
        K = self.params['K']
        u = self.params['u']
        h = self.params['h']
        b = self.params['b']
        
        # Uncertainty set parameter epsilon
        epsilon = self.params['epsilon']

        
        ## Constraint coefficients
        
        # LHS constraint coefficient matrix A1[t,s,m] with dim = tau x tau x n_constraints where A1[t,s,m]==0 for s > t
        A1 = np.array([np.array([np.array([h,-b])
                                if s<=t
                                else np.array([0,0])
                                for s in range(n_periods)])
                      for t in range(n_periods)])

        # LHS constraint coefficients A2[t,s,m] with dim = tau x tau x n_constraints where A2[t,s,m]==0 for s > t
        A2 = np.array([np.array([np.array([-h,b])
                                if s<=t
                                else np.array([0,0])
                                for s in range(n_periods)])
                      for t in range(n_periods)])

        # LHS constraint coefficients A3[t,s,m] with dim = tau x tau x n_constraints where A3[t,s,m]==0 for s != t
        A3 = np.array([np.array([np.array([-1,-1])
                                if s==t
                                else np.array([0,0])
                                for s in range(n_periods)])
                      for t in range(n_periods)])

        # RHS constraint coefficients f[t,m] with dim = tau x n_constraints 
        B = np.array([np.array([-h*I,b*I])
                      for t in range(n_periods)])
   


        ## Create model
        if self.params['LogToConsole']==0:
            
            env = gp.Env(empty=True)
            env.setParam('OutputFlag',0)
            env.start()
            self.m = gp.Model(env=env)
            
        else:
            
            self.m = gp.Model()
        
        # Primary decision variable (ordering quantity for each t)
        q0 = self.m.addVars(n_periods, vtype='C', name='q0')
        Q = self.m.addVars(n_periods, n_periods, vtype='C', name='Q')
    
        # Auxiary decision variable (for fixed cost of ordering for each sample i and t)
        z = self.m.addVars(n_samples, n_periods, vtype='B', name='z') 

        # Auxiary decision variable (for cost of inventory holding or demand backlogging for each t and sample i)
        s0_i = self.m.addVars(n_samples, n_periods, vtype='C', name='s0_i') 
        S_i = self.m.addVars(n_samples, n_periods, n_periods, vtype='C', name='S_i') 

        # Auxiary decision variable (from reformulation of robust constraints)
        l_i = self.m.addVars(n_samples, n_periods, vtype='C', lb=0, name='Lambda_i')
        
        # Auxiary decision variable (from reformulation of robust constraints)
        Lambda_i = self.m.addVars(n_samples, n_periods, n_periods, n_constraints, 
                                  vtype='C', lb=0, name='Lambda_i')
        
        # Auxiary decision variables for norm in objective
        norm_in_obj_inner = self.m.addVars(n_samples, n_periods, vtype='C', name='norm_in_obj_inner')
        norm_in_obj_inner_abs = self.m.addVars(n_samples, n_periods, vtype='C', name='norm_in_obj_inner_abs')
        norm_in_obj = self.m.addVars(n_samples, vtype='C', lb=0, name='norm_in_obj')
        
        # Auxiary decision variables for norms in constraints
        norm_in_cons_inner = self.m.addVars(n_samples, n_periods, n_periods, n_constraints, vtype='C', name='norm_in_cons_inner')
        norm_in_cons_inner_abs = self.m.addVars(n_samples, n_periods, n_periods, n_constraints, vtype='C', name='norm_in_cons_inner_abs')
        norm_in_cons = self.m.addVars(n_samples, n_periods, n_constraints, vtype='C', lb=0, name='norm_in_cons')

        
        

        ## Constraints   

        """
        
        Constraints (for each m=1...n_constraints, t=1...tau, i=1...n_samples). For the implementation using Gurobi, we do not
        use a 'big M' formulation to model fixed cost of ordering. Thus, the mathematical formulation is slightly different than
        provided in the paper.
  
        To calculate the l1 norm ||.|| we use further helper constraints.
  
        
        """        
        
        # Non-anticipativity of linear decision rules (set Q entries to zero where s >= t)
        Q_cons = self.m.addConstrs((Q[t,s] == 0)
                                   for s in range(n_periods)
                                   for t in range(n_periods)
                                   if (s >= t))
        
        # Non-anticipativity of linear decision rules (set Y_i entries to zero where s > t)
        S_i_cons = self.m.addConstrs((S_i[i,t,s] == 0)
                                     for s in range(n_periods) 
                                     for t in range(n_periods) 
                                     for i in range(n_samples)
                                     if (s > t))
        
        # Non-negative decision space for primary decision variable
        q_cons = self.m.addConstrs(q0[t]+gp.quicksum(Q[t,s]*d[i,s] for s in range(n_periods)) >= 0
                                   for t in range(n_periods)
                                   for i in range(n_samples))
                
        
        # Fixed cost
        z_cons1 = self.m.addConstrs((z[i,t] == 0) >> (q0[t]+gp.quicksum(Q[t,s]*d[i,s] for s in range(n_periods)) <= 0.5) 
                                    for t in range(n_periods) 
                                    for i in range(n_samples))
        z_cons2 = self.m.addConstrs((z[i,t] == 1) >> (q0[t]+gp.quicksum(Q[t,s]*d[i,s] for s in range(n_periods)) >= 0.5 + 10**-9) 
                                    for t in range(n_periods) 
                                    for i in range(n_samples))

        
        # Constraint to set inner part of the norm in the constraints
        norm_in_cons1 = self.m.addConstrs(

            norm_in_cons_inner[i,t,s,m] == A1[t,s,m] * Q[s,t] + A2[t,s,m] + A3[t,s,m] * S_i[i,s,t] +  Lambda_i[i,t,s,m]
            
            for s in range(n_periods)
            for m in range(n_constraints)
            for t in range(n_periods)
            for i in range(n_samples) 
        )

         # Constraint to set inner part of the norm in the constraints as absolute values
        norm_in_cons2 = self.m.addConstrs(

            norm_in_cons_inner_abs[i,t,s,m] == gp.abs_(norm_in_cons_inner[i,t,s,m])
            
            for s in range(n_periods)
            for m in range(n_constraints)
            for t in range(n_periods)
            for i in range(n_samples) 
        )

        
        # Constraint to get actual value of the norm in the constraints
        norm_in_cons3 =  self.m.addConstrs(

            norm_in_cons[i,t,m] == gp.quicksum(norm_in_cons_inner_abs[i,t,s,m] for s in range(n_periods))

            for m in range(n_constraints)
            for t in range(n_periods)
            for i in range(n_samples) 
        )
        
        
        # Constraint to set inner part of the norm in the objective
        norm_in_obj1 = self.m.addConstrs(

            norm_in_obj_inner[i,t] == (gp.quicksum(Q[s,t]*u[s] for s in range(n_periods)) + 
                                       gp.quicksum(S_i[i,s,t]*1 for s in range(n_periods))  + 
                                       l_i[i,t])
            
            for t in range(n_periods)
            for i in range(n_samples) 
        )

         # Constraint to set inner part of the norm in the objective as absolute values
        norm_in_obj2 = self.m.addConstrs(

            norm_in_obj_inner_abs[i,t] == gp.abs_(norm_in_obj_inner[i,t])
            
            for t in range(n_periods)
            for i in range(n_samples) 
        )

        
        # Constraint to get actual value of the norm in the objective
        norm_in_obj3 =  self.m.addConstrs(

            norm_in_obj[i] == gp.quicksum(norm_in_obj_inner_abs[i,t] for t in range(n_periods))

            for i in range(n_samples) 
        )
        


        # Actual constraints
        cons = self.m.addConstrs(

            
            # A1 * (q0 + Q d)
            gp.quicksum(A1[t,s,m]*(q0[s]+gp.quicksum(Q[s,ts]*d[i,ts] for ts in range(n_periods))) for s in range(n_periods)) +

            # A2 * d
            gp.quicksum(A2[t,s,m]*d[i,s] for s in range(n_periods)) +

            # A3 * (s0_i + S_i d)
            gp.quicksum(A3[t,s,m]*(s0_i[i,s]+gp.quicksum(S_i[i,s,ts]*d[i,ts] for ts in range(n_periods))) for s in range(n_periods)) +

            # Lambda_i * d
            gp.quicksum(Lambda_i[i,t,s,m]*d[i,s] for s in range(n_periods)) +

            # epsilon*||A1*Q + A2 + A3*S_i + Lambda_i||_1
            epsilon * norm_in_cons[i,t,m]

            <= B[t,m]

            for m in range(n_constraints)
            for t in range(n_periods)
            for i in range(n_samples) 
        )

        

        ## Objective 
        obj = self.m.setObjective(

            # Weighted sum
            gp.quicksum(

                # i'th weight
                w[i] * (                                         

                    # u * (q0 + Q d)
                    gp.quicksum(u*(q0[t]+gp.quicksum(Q[t,s]*d[i,s] for s in range(n_periods))) for t in range(n_periods)) + 

                    # K * z
                    gp.quicksum(K*z[i,t] for t in range(n_periods)) +                     
                    
                    # s0_i + S_i d
                    gp.quicksum(s0_i[i,t]+gp.quicksum(S_i[i,t,s]*d[i,s] for s in range(n_periods)) for t in range(n_periods)) +
                    
                    # l_i d
                    gp.quicksum(l_i[i,t] * d[i,t] for t in range(n_periods)) +
                    
                    # epsilon * || Q^u + S_i^1 + l_i ||
                    epsilon * norm_in_obj[i]                   

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
        
        ...
        
        Arguments: None
            
        Further key word arguments (kwargs): passed to update Gurobi meta params (other kwargs are ignored)

        Returns:
        
            q_hat: ...
            status: ...
            solutions: ...
            gap: ...
            
        """

        
        # Update gurobi meta params if provided
        gurobi_meta_params = {
            
            'LogToConsole': kwargs.get('LogToConsole', self.params['LogToConsole']),
            'Threads': kwargs.get('Threads', self.params['Threads']),
            'NonConvex': kwargs.get('NonConvex', self.params['NonConvex']),
            'PSDTol': kwargs.get('PSDTol', self.params['PSDTol']),
            'MIPGap': kwargs.get('MIPGap', self.params['MIPGap']),
            'NumericFocus': kwargs.get('NumericFocus', self.params['NumericFocus']),
            'obj_improvement': kwargs.get('obj_improvement', self.params['obj_improvement']),
            'obj_timeout_sec': kwargs.get('obj_timeout_sec', self.params['obj_timeout_sec']),
            'obj_timeout_max_sec': kwargs.get('obj_timeout_max_sec', self.params['obj_timeout_max_sec'])
        }
        
        self.set_params(**gurobi_meta_params)           
            
        # Set Gurobi meta params
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
            #q_hat = [var.xn for var in self.m.getVars() if 'q' in var.VarName]
            
            

            q0_hat = [var.x
                      for var in self.m.getVars()
                      for t in range(self.n_periods)
                      if (var.VarName == 'q0['+str(t)+']')]

            Q_hat = np.array([np.array([var.x 
                                        for var in self.m.getVars()
                                        for s in range(self.n_periods)
                                        if (var.VarName == 'Q['+str(t)+','+str(s)+']')])
                              for t in range(self.n_periods)])

            # If some d (actuals) are provided
            if 'd' in kwargs:
                q_hat = [q0_hat[t]+sum(Q_hat[t,s]*kwargs['d'][s]
                                       for s in range(self.n_periods))
                         for t in range(self.n_periods)]
            else:
                q_hat = copy.deepcopy(q0_hat)
        
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
    def run(self, wsaamodel, samples, actuals, weights=None, epsilons=None, I=0, **kwargs):

        """

        Description ...
        
        Models: If weights are given, performs Weighted SAA, else performs SAA. If wsaamodel is 'RobustWeightedSAA', then epsilons 
        should be passed as arguments and model performs the robust extension.


        Arguments
            
            wsaamodel: the model that should be used, which is either WeightedSAA() or RobustWeightedSAA()
            samples: dict for periods t=1,...,T of historical demand samples (like 'y_train') that are array-like with shape (n_samples, n_periods)
            actuals: dict for periods t=1,...,T of actual demands (like 'y_test') that are array-like with shape (n_periods,)
            weights (optional): dict for periods t=1,...,T of weights (flat array with same lengths as y_train demand 
                                samples for this period)
            epsilons (optional): dict for periods t=1,...,T of epsilons specifying uncertanty sets for robust extension
            I (optional): starting inventory
            q_ub (optional): uppper bound for q

        Further key word arguments (kwargs): passed as params to create() where vaild params will be used, rest will be ignored

        Returns: DataFrame storing results for t=1...T

        """


        # Get length of planning horizon T
        T = len(samples)

        # Initialize results lists
        I_t = []
        q_t = []
        Iq_t = []
        d_t = []
        Iqd_t = []
        
        defaulted_t = []
        status_t = []
        solutions_t = []
        gap_t = []

        exec_time_sec = []
        cpu_time_sec = []

        ## Iterate over planning horizon t=1...T
        for t in range(1,T+1):

            # Timer
            st_exec = time.time()
            st_cpu = time.process_time()      

            # Historical demand samples available in period t
            d = samples[t]

            # Weights for Weighted SAA
            if not weights is None:

                w = weights[t] 

            # Weights for SAA (all weights are 1/n_samples)
            else:

                w = np.array([np.repeat(1/d.shape[0],d.shape[0])]).flatten()


            ## Summarize weights and samples
            d = d[list(w > 0)]
            w = w[list(w > 0)]

            df=pd.DataFrame(data=np.hstack(
                (w.reshape(w.shape[0],1), d.reshape(d.shape[0],1) if d.ndim == 1 else d)), 
                    columns=[-1] + list(range(0,d.shape[1] if d.ndim > 1 else 1))).groupby(
                list(range(0,d.shape[1] if d.ndim > 1 else 1))).agg(
                    w = (-1, np.sum)).reset_index()

            d = np.array(df[list(range(0,d.shape[1] if d.ndim > 1 else 1))])
            w = np.array([df.w]).flatten()

            # Robust
            if not epsilons is None:

                epsilon = epsilons[t]   
                
            else:
                
                epsilon = None
            
            
            ## Optimization

            # Parameters
            params = {

                **kwargs, 'epsilon': epsilon

            } 

            
            # If inventory is nan (no further optimization)
            if np.isnan(I):
                
                # Store results
                I_t = I_t + [np.nan]     # inventory before ordering and demand
                q_t = q_t + [np.nan]     # ordering quantity 
                Iq_t = Iq_t + [np.nan]   # inventory after ordering 
                d_t = d_t + [np.nan]     # realization demand
                Iqd_t = Iqd_t + [np.nan] # inventory after ordering and demand

                defaulted_t = defaulted_t + [""]
                status_t = status_t + [""]
                solutions_t = solutions_t + [""]
                gap_t = gap_t + [""]
                
                # Update inventory
                I = I + np.nan

            # If inventory is not nan
            else:

                # Create model 
                wsaamodel.create(d, w, I, **params)
                
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
                    wsaamodel_default.create(d, w, I, **params)

                    # Optimize and get decisions
                    q_hat, status, solutions, gap = wsaamodel_default.optimize()    
                    
                    # Flag that model defaulted to WeightedSAA
                    defaulted = True                
                        
                    # Delete model
                    wsaamodel_default.dump()
                    

                # Store results
                I_t = I_t + [I] 
                q_t = q_t + [np.around(q_hat[0],0).item()]
                Iq_t = Iq_t + [I + np.around(q_hat[0],0).item()] 
                d_t = d_t + [actuals[t].flatten()[0].item()] 
                Iqd_t = Iqd_t + [I + np.around(q_hat[0],0).item() - actuals[t].flatten()[0].item()]

                defaulted_t = defaulted_t + [defaulted]
                status_t = status_t + [status]
                solutions_t = solutions_t + [solutions]
                gap_t = gap_t + [gap]
        
                # Update inventory
                I = I + np.around(q_hat[0],0).item() - actuals[t].flatten()[0].item()
                 

            # Timing
            exec_time_sec = exec_time_sec+[time.time()-st_exec]  
            cpu_time_sec = cpu_time_sec+[time.process_time()-st_cpu]  
            

        ## Evaluate results

        # Store results
        results = pd.DataFrame({
            'K': wsaamodel.get_params()['K'],
            'u': wsaamodel.get_params()['u'],
            'h': wsaamodel.get_params()['h'],
            'b': wsaamodel.get_params()['b'],
            't': np.array(range(1,T+1)),
            'I': I_t,
            'q': q_t,
            'I_q': Iq_t,
            'y': d_t,
            'I_q_y': Iqd_t,
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
    
    
    
    
    
    
    ### Function to run ex-post (clairvoyant) experiment over planing horizon t=1...T
    def run_expost(self, wsaamodel, actuals, I=0, **kwargs):

        """
        ...

        """


        # Get length of planning horizon T
        T = actuals.shape[1]

        # Timer
        st_exec = time.time()
        st_cpu = time.process_time()      

        ## Optimization

        # Demands
        d = copy.deepcopy(actuals)
        actuals = actuals.flatten()
        
        # Fake weights
        w = np.array([1])
        
        # Create model 
        wsaamodel.create(d, w, I, **kwargs)

        # Optimize and get decisions
        q_hat, status, solutions, gap = wsaamodel.optimize()    

        # Flag that did not default to WeightedSAA
        defaulted = False   

        # Delete model
        wsaamodel.dump()

        # Initialize results lists
        I_t = []
        q_t = []
        Iq_t = []
        d_t = []
        Iqd_t = []
        
        defaulted_t = []
        status_t = []
        solutions_t = []
        gap_t = []

        exec_time_sec = []
        cpu_time_sec = []       
        
        # Iterate over planning horizon t=1...T to calculate results
        for t in range(1,T+1):
            
            # Store results
            I_t = I_t + [I] 
            q_t = q_t + [np.around(q_hat[t-1],0).item()]
            Iq_t = Iq_t + [I + np.around(q_hat[t-1],0).item()] 
            d_t = d_t + [actuals[t-1].item()] 
            Iqd_t = Iqd_t + [I + np.around(q_hat[t-1],0).item() - actuals[t-1].item()]

            defaulted_t = defaulted_t + [defaulted]
            status_t = status_t + [status]
            solutions_t = solutions_t + [solutions]
            gap_t = gap_t + [gap]

            # Update inventory
            I = I + np.around(q_hat[t-1],0).item() - actuals[t-1].item()
                
            # Timing
            exec_time_sec = exec_time_sec+[time.time()-st_exec]  
            cpu_time_sec = cpu_time_sec+[time.process_time()-st_cpu]  

        ## Evaluate results

        # Store results
        results = pd.DataFrame({
            'K': wsaamodel.get_params()['K'],
            'u': wsaamodel.get_params()['u'],
            'h': wsaamodel.get_params()['h'],
            'b': wsaamodel.get_params()['b'],
            't': np.array(range(1,T+1)),
            'I': I_t,
            'q': q_t,
            'I_q': Iq_t,
            'y': d_t,
            'I_q_y': Iqd_t,
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
    
    