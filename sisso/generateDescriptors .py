# -*- coding: utf-8 -*-
# Source: https://github.com/NREL/SISSORegressor_MATLAB/blob/main/generateDescriptors.m

import numpy as np

#### Operation Block operates on descriptors, generates the new name (representing the mathematical operations), and records the indices of the parent descriptors

# Single Variable Operation

#### OperationBlock(X,X_names,X_ref)

#   Parameters
#        param X: (np.array) set of input desciptors
#        param X_names: (List[string]) List of descriptors names
#        param X_ref: (int) index that corresponds to the first descriptor in X
#   returns:
#        Xout: (np.array) new desciptors
#        Xoutvars: (List[string]) Names of new descriptors
#        parents: List[parentid1,-1]: List of indices of parents (2nd place is -1 to denote only one parent)

# 2-Variable Operation

#### OperationBlock(X,Y,X_names,Y_names,X_ref,Y_ref)

#   Parameters
#        param X: (np.array) set of input desciptors
#        param Y: (np.array) set of input desciptors
#        param X_names: (List[string]) List of descriptors names
#        param Y_names: (List[string]) List of descriptors names
#        param X_ref: (int) index that corresponds to the first descriptor in X
#        param Y_ref: (int) index that corresponds to the first descriptor in Y

#   returns:
#        Xout: (np.array) new desciptors
#        Xoutvars: (List[string]) Names of new descriptors
#        parents: List[parentid1,parentid2]: List of indices of parents (2nd place is -1 to denote only one parent)

# Each Block 

def Sqrt(X,Xvars,x_ref):
    Xout = np.copy(X)
    Xoutvars = Xvars[:]
    parents = []
    for i in range(X.shape[1]):
        Xout[:,i] = X[:,i]**0.5
        Xoutvars[i] = '(' + Xvars[i] + '^0.5'+')'
        parents.append([i+x_ref,-1])
    return [Xout,Xoutvars,parents]
    
def Square(X,Xvars,x_ref):
    Xout = np.copy(X)
    Xoutvars = Xvars[:]
    parents = []
    for i in range(X.shape[1]):
        Xout[:,i] = X[:,i]**2.0
        Xoutvars[i] = '(' + Xvars[i] + '^2'+')'
        parents.append([i+x_ref,-1])
    return [Xout,Xoutvars,parents]

def Cube(X,Xvars,x_ref):
    Xout = np.copy(X)
    Xoutvars = Xvars[:]
    parents = []
    for i in range(X.shape[1]):
        Xout[:,i] = X[:,i]**3.0
        Xoutvars[i] = '(' + Xvars[i] + '^3'+')'
        parents.append([i+x_ref,-1])
    return [Xout,Xoutvars,parents]
    
def CubeRoot(X,Xvars,x_ref):
    Xout = np.copy(X)
    Xoutvars = Xvars[:]
    parents = []
    for i in range(X.shape[1]):
        Xout[:,i] = X[:,i]**(1.0/3.0)
        Xoutvars[i] = '(' + Xvars[i] + '^(1/3)'+')'
        parents.append([i+x_ref,-1])
    return [Xout,Xoutvars,parents]
    
def FourthRoot(X,Xvars,x_ref):
    Xout = np.copy(X)
    Xoutvars = Xvars[:]
    parents = []
    for i in range(X.shape[1]):
        Xout[:,i] = X[:,i]**(1.0/4.0)
        Xoutvars[i] = '(' + Xvars[i] + '^(1/4)'+')'
        parents.append([i+x_ref,-1])
    return [Xout,Xoutvars,parents]
    
def FourthPower(X,Xvars,x_ref):
    Xout = np.copy(X)
    Xoutvars = Xvars[:]
    parents = []
    for i in range(X.shape[1]):
        Xout[:,i] = X[:,i]**4.0
        Xoutvars[i] = '(' + Xvars[i] + '^4'+')'
        parents.append([i+x_ref,-1])
    return [Xout,Xoutvars,parents]
        
def Reciprocal(X,Xvars,x_ref):
    Xout = np.copy(X)
    Xoutvars = Xvars[:]
    parents = []
    for i in range(X.shape[1]):
        Xout[:,i] = 1.0/X[:,i]
        Xoutvars[i] = '(1/' + Xvars[i] +')'
        parents.append([i+x_ref,-1])
    return [Xout,Xoutvars,parents]

def EXP(X,Xvars,x_ref):
    Xout = np.copy(X)
    Xoutvars = Xvars[:]
    parents = []
    for i in range(X.shape[1]):
        Xout[:,i] = np.exp(X[:,i])
        Xoutvars[i] = 'exp(' + Xvars[i] +')'
        parents.append([i+x_ref,-1])
    return [Xout,Xoutvars,parents]

def Log(X,Xvars,x_ref):
    Xout = np.copy(X)
    Xoutvars = Xvars[:]
    parents = []
    for i in range(X.shape[1]):
        Xout[:,i] = np.log(X[:,i])
        Xoutvars[i] = 'ln(' + Xvars[i] +')'
        parents.append([i+x_ref,-1])
    return [Xout,Xoutvars,parents]

def Sin(X,Xvars,x_ref):
    Xout = np.copy(X)
    Xoutvars = Xvars[:]
    parents = []
    for i in range(X.shape[1]):
        Xout[:,i] = np.sin(X[:,i])
        Xoutvars[i] = 'sin(' + Xvars[i] +')'
        parents.append([i+x_ref,-1])
    return [Xout,Xoutvars,parents]

def Cos(X,Xvars,x_ref):
    Xout = np.copy(X)
    Xoutvars = Xvars[:]
    parents = []
    for i in range(X.shape[1]):
        Xout[:,i] = np.sin(X[:,i])
        Xoutvars[i] = 'cos(' + Xvars[i] +')'
        parents.append([i+x_ref,-1])
    return [Xout,Xoutvars,parents]


def Abs(X,Xvars,x_ref):
    Xout = np.copy(X)
    Xoutvars = Xvars[:]
    parents = []
    for i in range(X.shape[1]):
        Xout[:,i] = np.sin(X[:,i])
        Xoutvars[i] = '|' + Xvars[i] +'|'
        parents.append([i+x_ref,-1])
    return [Xout,Xoutvars,parents]

    
#####
    
def Multiply(X1, X2, X1vars, X2vars, x1_ref, x2_ref):
    
    numdatapoints = X1.shape[0]
    numfeaturevars = X1.shape[1]*X2.shape[1]
    Xout = np.zeros((numdatapoints,numfeaturevars))
    Xoutvars = []
    parents = []
    idxXout = 0
    for i in range(X1.shape[1]):
        for j in range(X2.shape[1]):
            Xout[:,idxXout] = X1[:,i]*X2[:,j]
            new_str = '(' + X1vars[i] + '*' + X2vars[j] + ')'
            Xoutvars.append(new_str)
            idxXout += 1
            parents.append([i+x1_ref,j+x2_ref])
    return [Xout,Xoutvars,parents]


def Divide(X1, X2, X1vars, X2vars, x1_ref, x2_ref):
    
    numdatapoints = X1.shape[0]
    numfeaturevars = X1.shape[1]*X2.shape[1]
    Xout = np.zeros((numdatapoints,numfeaturevars))
    Xoutvars = []
    parents = []
    idxXout = 0
    for i in range(X1.shape[1]):
        for j in range(X2.shape[1]):
            Xout[:,idxXout] = X1[:,i]/X2[:,j]
            new_str = '(' + X1vars[i] + '/' + X2vars[j] + ')'
            Xoutvars.append(new_str)
            idxXout += 1
            parents.append([i+x1_ref,j+x2_ref])
    return [Xout,Xoutvars,parents]

def Subtract(X1, X2, X1vars, X2vars, x1_ref, x2_ref):
    
    numdatapoints = X1.shape[0]
    numfeaturevars = X1.shape[1]*X2.shape[1]
    Xout = np.zeros((numdatapoints,numfeaturevars))
    Xoutvars = []
    parents = []
    idxXout = 0
    for i in range(X1.shape[1]):
        for j in range(X2.shape[1]):
            Xout[:,idxXout] = X1[:,i]-X2[:,j]
            new_str = '(' + X1vars[i] + '-' + X2vars[j] + ')'
            Xoutvars.append(new_str)
            idxXout += 1
            parents.append([i+x1_ref,j+x2_ref])
    return [Xout,Xoutvars,parents]

def Add(X1, X2, X1vars, X2vars, x1_ref, x2_ref):
    
    numdatapoints = X1.shape[0]
    numfeaturevars = X1.shape[1]*X2.shape[1]
    Xout = np.zeros((numdatapoints,numfeaturevars))
    Xoutvars = []
    parents = []
    idxXout = 0
    for i in range(X1.shape[1]):
        for j in range(X2.shape[1]):
            Xout[:,idxXout] = X1[:,i]+X2[:,j]
            new_str = '(' + X1vars[i] + '+' + X2vars[j] + ')'
            Xoutvars.append(new_str)
            idxXout += 1
            parents.append([i+x1_ref,j+x2_ref])
    return [Xout,Xoutvars,parents]
        

def generateDescriptors(x,xVars,ops=2):
    # Parameters
    #       x: List[np.array]: List of sets of input features 
    #       xVars: List[List[string]]: List of names of input features associated with each set
    #       ops: int - number of rounds of desciptor construction to perform
    # Returns 
    #       x: np.array: Candidate descriptors
    #       XVars: List[string] - string representation of mathematical operations
    #       parents: List[parentid1,parentid2] - List of indexes of parent features, [id,-1] denotes one parent [-1,-1] denotes no parents
    #
    # Generates Candidate Descriptors
    # Features are input as a list of np.arrays, where each array can correspond to descriptors having particular units
    # This would allow modification of this function to constrain operations such that only ones that makes sense in terms of units can be performed
    # In this work our input features are unitless, so we do not constrain mathematical operations
    
    parents = []
    x_ref = []
    curr_ref_val = 0
    
    ### Initialize parents of starting features to [-1,-1] meaning no parents
    ### Keep track of first index of each set of features
    for i in range(len(x)):
        curr_x = x[i]
        parents.append(-1.0*np.ones((x[i].shape[1],2)))
        x_ref.append(curr_ref_val)
        curr_ref_val += x[i].shape[1]
    
    for op in range(ops):
        
        # New features to be added at end of operation round
        new_parents = []
        new_x_refs = []
        new_xs = []
        new_vars = []
    
        # Perform Single Variable operations for each set of features
        for idxGroup in range(len(x)):
            
            [Sqrt_X,Sqrt_Xvars,Sqrt_pars] = Sqrt(x[idxGroup],xVars[idxGroup],x_ref[idxGroup])
            new_xs.append(Sqrt_X)
            new_vars.append(Sqrt_Xvars)
            new_parents.append(Sqrt_pars)
            
            [Square_X,Square_Xvars,Square_pars] = Square(x[idxGroup],xVars[idxGroup],x_ref[idxGroup])
            new_xs.append(Square_X)
            new_vars.append(Square_Xvars)
            new_parents.append(Square_pars)
            
            [CubeRoot_X,CubeRoot_Xvars,CubeRoot_pars] = CubeRoot(x[idxGroup],xVars[idxGroup],x_ref[idxGroup])
            new_xs.append(CubeRoot_X)
            new_vars.append(CubeRoot_Xvars)
            new_parents.append(CubeRoot_pars)
            
            [Cube_X,Cube_Xvars,Cube_pars] = Cube(x[idxGroup],xVars[idxGroup],x_ref[idxGroup])
            new_xs.append(Cube_X)
            new_vars.append(Cube_Xvars)
            new_parents.append(Cube_pars)
            
            [FourthRoot_X,FourthRoot_Xvars,FourthRoot_pars] = FourthRoot(x[idxGroup],xVars[idxGroup],x_ref[idxGroup])
            new_xs.append(FourthRoot_X)
            new_vars.append(FourthRoot_Xvars)
            new_parents.append(FourthRoot_pars)
            
            [FourthPower_X,FourthPower_Xvars,FourthPower_pars] = FourthPower(x[idxGroup],xVars[idxGroup],x_ref[idxGroup])
            new_xs.append(FourthPower_X)
            new_vars.append(FourthPower_Xvars)
            new_parents.append(FourthPower_pars)
            
            [Reciprocal_X,Reciprocal_Xvars,Reciprocal_pars] = Reciprocal(x[idxGroup],xVars[idxGroup],x_ref[idxGroup])
            new_xs.append(Reciprocal_X)
            new_vars.append(Reciprocal_Xvars)
            new_parents.append(Reciprocal_pars)
            
            [EXP_X,EXP_Xvars,EXP_pars] = EXP(x[idxGroup],xVars[idxGroup],x_ref[idxGroup])
            new_xs.append(EXP_X)
            new_vars.append(EXP_Xvars)
            new_parents.append(EXP_pars)

            [Log_X,Log_Xvars,Log_pars] = Log(x[idxGroup],xVars[idxGroup],x_ref[idxGroup])
            new_xs.append(Log_X)
            new_vars.append(Log_Xvars)
            new_parents.append(Log_pars)
            
            [Sin_X,Sin_Xvars,Sin_pars] = Sin(x[idxGroup],xVars[idxGroup],x_ref[idxGroup])
            new_xs.append(Sin_X)
            new_vars.append(Sin_Xvars)
            new_parents.append(Sin_pars)
            
            [Cos_X,Cos_Xvars,Cos_pars] = Cos(x[idxGroup],xVars[idxGroup],x_ref[idxGroup])
            new_xs.append(Cos_X)
            new_vars.append(Cos_Xvars)
            new_parents.append(Cos_pars)
            
            [Abs_X,Abs_Xvars,Abs_pars] = Abs(x[idxGroup],xVars[idxGroup],x_ref[idxGroup])
            new_xs.append(Abs_X)
            new_vars.append(Abs_Xvars)
            new_parents.append(Abs_pars)
            

        
        fixed_x = len(x)

        # Perform 2-Variable operations for each pair of features
        for i in range(fixed_x):
            for j in range(i,fixed_x):
                
                [Multiply_X,Multiply_Xvars,Multiply_pars] = Multiply(x[i],x[j],xVars[i],xVars[j],x_ref[i],x_ref[j])
                new_xs.append(Multiply_X)
                new_vars.append(Multiply_Xvars)
                new_parents.append(Multiply_pars)
                
                [Divide_X,Divide_Xvars,Divide_pars] = Divide(x[i],x[j],xVars[i],xVars[j],x_ref[i],x_ref[j])
                new_xs.append(Divide_X)
                new_vars.append(Divide_Xvars)
                new_parents.append(Divide_pars)
                
                [Add_X,Add_Xvars,Add_pars] = Add(x[i],x[j],xVars[i],xVars[j],x_ref[i],x_ref[j])
                new_xs.append(Add_X)
                new_vars.append(Add_Xvars)
                new_parents.append(Add_pars)
                
                [Subtract_X,Subtract_Xvars,Subtract_pars] = Subtract(x[i],x[j],xVars[i],xVars[j],x_ref[i],x_ref[j])
                new_xs.append(Subtract_X)
                new_vars.append(Subtract_Xvars)
                new_parents.append(Subtract_pars)
                
        # Add new features

        x += new_xs
        xVars += new_vars
        parents += new_parents
        
        ### Remove bad columns
        for i in range(len(x)):
            curr_x = x[i]
            curr_x_names = xVars[i]
            curr_x_parents = parents[i]

            ##### Remove Features with Nan / inf values

            valid = np.isfinite(curr_x).all(axis=0)

            curr_x = curr_x[:,valid]
            curr_x_names = [curr_x_names[j] for j in range(len(valid)) if valid[j]]
            curr_x_parents = [curr_x_parents[j] for j in range(len(valid)) if valid[j]]

            ###### Remove Redundant Features where all values are equal
            
            out_comp = curr_x == curr_x[0,:]
            info = ~np.all(out_comp, axis = 0)
            curr_x = curr_x[:,info]
            curr_x_names = [curr_x_names[j] for j in range(len(info)) if info[j]]
            curr_x_parents = [curr_x_parents[j] for j in range(len(info)) if info[j]]
            
            x[i] = curr_x
            xVars[i] = curr_x_names
            parents[i] = curr_x_parents

            ##### Remove Features with very small standard deviation - these were found to be unstable in stages of SISSO
    
            valid_std = (x_finite.std(axis=0)>0.000001)

            x_out = x_finite[:,valid_std]
            xVars_out = [xVars_finite[i] for i in range(len(valid_std)) if valid_std[i]]
            parents_out = [parents_finite[i] for i in range(len(valid_std)) if valid_std[i]]
            
        x_ref = []
        curr_ref_val = 0
            
        ### Keep track of first index of each set of features
        for i in range(len(x)):
            x_ref.append(curr_ref_val)
            curr_ref_val += x[i].shape[1]
        
    ### Flatten Descriptors

    x = np.concatenate(x,axis=1)
    xVars = [item for sublist in xVars for item in sublist]
    parents = [item for sublist in parents for item in sublist]
     
    return [x,xVars,parents]


