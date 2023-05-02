# -*- coding: utf-8 -*-
# Source: https://github.com/NREL/SISSORegressor_MATLAB/blob/main/sisso.py
"""
ORIGINAL INTRODUCTION from PAUL GASPER
Docstring:     
A simple implementation of the SISSO algorithm (R. Ouyang, S. Curtarolo, 
E. Ahmetcik et al., Phys. Rev. Mater.2, 083802 (2018)) for regression for
workshop tutorials. SISSO is an iterative approach where at each iteration
first SIS (Sure Independence Sreening) and SO (Sparsifying Operator, here l0-regularization) 
is applied.
Note that it becomes the orthogonal matching pursuit for n_features_per_sis_iter=1.
This code was copied from:
https://analytics-toolkit.nomad-coe.eu/hub/user-redirect/notebooks/tutorials/compressed_sensing.ipynb
A more efficient fortran implementation can be found on https://github.com/rouyang2017.


THIS CLASSIFIER IS A MODIFIED VERSION OF THE CODE BY PAUL GASPER REFERENCED ABOVE
"""

import functools
import operator as op
import numpy as np
import pandas as pd
from itertools import combinations, product
from sklearn import tree
from sklearn import svm
import copy
from sklearn.model_selection import train_test_split
from scipy.spatial import ConvexHull,HalfspaceIntersection
from scipy.spatial.qhull import QhullError

#https://www.geeksforgeeks.org/python-convert-list-into-list-of-lists/
def extractDigits(lst):
    ### Converts list into list of lists
    return [[el] for el in lst]


class SissoClassifier(object):
    """
    Implementation of a SISSO classifier based on work of Ouyang et al. https://journals.aps.org/prmaterials/abstract/10.1103/PhysRevMaterials.2.083802
    1D domain overlap is used as the SIS stage. Both Domain overlap and decision trees of depth 2 are options for the SO step.
    See Bartel el al. https://www.science.org/doi/10.1126/sciadv.aav0693 for an example of a decision tree in the SO step
    For higher dimensional descriptors, SIS stage is based on residuals of the previous model
    By default new descriptors are selected keeping previous descriptors fixed.

    USAGE

    classifier.fit(Features,Labels)
    classifier.sis_selected_indices : List[List[ints]] - Features selected by SIS at each dimension
    classifier.l0_selected_indices: List[List[ints]] - Features selected by SO at each dimension
    classifier.stored_error: List[float] - List of Errors on Training set for each dimension
    """
    
    def __init__(self, n_nonzero_coefs=1, n_features_per_sis_iter=1, all_l0_combinations=False, weighted=False, SO_method="decision_tree"):
        """
        Parameters:
            n_nonzer_coefs: (int) - Number of descriptors to select
            n_features_per_sis_iter: (int) - Number of candidates to select in SIS stage of SISSO
            all_l0_combinations: (Bool) - Whether to consider all possible combinations of features from previous SIS stages during SO (by default previous descriptors are fixed)
            weighted: (Bool) - Whether inbalanced class weights are considered in classification strategy
            SO_method: (string) - Strategy for SO - If not decision_tree, domain overlap is used
        """

        self.n_nonzero_coefs = n_nonzero_coefs
        self.n_features_per_sis_iter = n_features_per_sis_iter
        self.all_l0_combinations = all_l0_combinations
        self.weighted = weighted
        self.SO_method = SO_method
    
    def fit(self, D, P):
        """
        Fit the classifier to data
        Params:
            D: np.array input features
            P: class labels
        """
        self._initialize_variables()
        self.sis_not_selected_indices = np.arange(D.shape[1])

        # standardize D
        self._set_standardizer(D)
        D = self._get_standardized(D)
        
        for i_iter in range(self.n_nonzero_coefs):
            ### SIS STEP
            # set residual, in first iteration the target P is used
            if i_iter == 0:
                indices_n_closest = self._sis(D, P)
            else:
                if self.SO_method == "decision_tree":
                    dt_error, Residual = self._get_decision_tree(D[:,self.l0_selected_indices[-1]],P)
                else:
                    dm_ov,vol_ov,Residual = self._get_domain_overlap(D[:,self.l0_selected_indices[-1]],P)

                ### select based on residuals
                indices_n_closest = self._sis(D[Residual==1], P[Residual==1])
            self.sis_selected_indices.append( indices_n_closest )

            ### SO STEP
            self.curr_selected_indices,best_err = self._l0_regularization(D, P, self.sis_selected_indices)
            self.stored_error.append(best_err)
            self.l0_selected_indices.append(self.curr_selected_indices)
                          
     
    def _initialize_variables(self):
        """
        Initialize Stored Data for SISSO model
        """
        # variabels for standardizer
        self.scales = 1.
        self.means = 0.

        # indices selected SIS 
        self.sis_selected_indices = []
        self.sis_not_selected_indices = None
        
        # indices selected by L0 (after each SIS step)
        self.l0_selected_indices = []
        self.stored_error = []
        self.curr_selected_indices = None
        self.store_best = None
        

    def _set_standardizer(self, D):
        """
        Record Mean / std of input data
        Params:
            D - np.array - Data to standardize
        """
        self.means  = D.mean(axis=0)
        self.scales = D.std(axis=0)
    
    def _get_standardized(self, D):
        """
        Standardize input data
        Params:
             D - np.array - Data to standardize
        Returns:
             D_norm - np.array - Standardized input data
        """
        return (D - self.means) / self.scales

    def _l0_regularization(self, D, P, list_of_sis_indices):
        """
        Select best low-dimensional descriptor based on 
        Params:
            D - np.array - Input Data
            P - np.array - Labels
            list_of_sis_indices - List[List[int]] - Record of descriptors selected by SIS at each stage
        returns:
            Selected_indices - List[int] - Indices of best descriptors
            Error_min: float - Best Training Error
        """

        Error_min = 9999999999.
        
        ### Consider all possible combinations of descriptors selected by SIS
        if self.all_l0_combinations:
            combinations_generator = combinations(np.concatenate(list_of_sis_indices), len(list_of_sis_indices))
        ### Fix Previous Descriptors
        else:
            if len(self.l0_selected_indices) > 0:
                combinations_generator = product(*extractDigits(self.l0_selected_indices[-1]),list_of_sis_indices[-1])
            else: 
                combinations_generator = extractDigits(list_of_sis_indices[-1].tolist())

        ### Test all possibilities
        for indices_combi in combinations_generator:
            D_ls = D[:, indices_combi]
            if self.SO_method == "decision_tree":
                dt_error, error_labels = self._get_decision_tree(D_ls,P)
                curr_error = dt_error
            else:
                curr_domain_overlap, curr_volume_overlap, error_labels = self._get_domain_overlap(D_ls,P)
                curr_error = curr_domain_overlap
            if curr_error  < Error_min: 
                Error_min = curr_error
                indices_combi_min = indices_combi
            
        return list(indices_combi_min),Error_min
    
            
    def _get_decision_tree(self,D,P):
        """ 
        Implement decision tree classifier for SO

        Params:
            D - np.array - Input Features
            P - np.array - Labels

        Returns:
            Error - int - Number of Misclassified points
            Residual - np.array - Record of whether points were correctly (0) or incorrectly (1) classified
        """
        if self.weighted:
            clf = tree.DecisionTreeClassifier(max_depth=2,class_weight="balanced")
        else:
            clf = tree.DecisionTreeClassifier(max_depth=2)
        clf.fit(D,P)
        predictions = clf.predict(D)
        Residual = np.abs(P - predictions)
        return np.sum(Residual),Residual
            
    def _get_domain_overlap(self, D_model, P):
        """
        Compute Domain Overlap from Ouyang et al.

        Params:
            D_model: - np.array - Input Features
            P: - np.array - Labels

        Returns:
            Domain_Overlap - float - Number of points in lying in intersection of convex hulls
            Volume_Overlap - float - Volume of Overlapping region of convex hulls
            Error_Labels - List[int] - Record of whether points were correctly (0) or incorrectly (1) classified (Incorrect points lie in both hulls)
        """

        init_features = D_model.shape[1]
        ### Features with label 0
        D_zero = D_model[P==0].reshape(-1,init_features)
        ### Features with label 1
        D_one = D_model[P==1].reshape(-1,init_features)
        ### Whether to apply weights to Domain Overlap
        if self.weighted:
            weight_zero = (2.0*D_zero.shape[0])/(D_model.shape[0])
            weight_one = (2.0*D_one.shape[0])/(D_model.shape[0])
        else:
            weight_zero = 1.0
            weight_one = 1.0
        
        ### One dimensional Problem
        if D_zero.shape[1]==1:

            ### If one set is empty - Presume convex hull is full range
            if len(D_zero)==0:
                zero_max = 999999
                zero_min = -999999
            else:
                zero_max = D_zero.max()
                zero_min = D_zero.min()
            
            if len(D_one)==0:
                one_max = 999999
                one_min = -999999
            else:
                one_max = D_one.max()
                one_min = D_one.min()
            
            error_labels = []
            domain_overlap = 0
            feasible = None
            for i in range(D_model.shape[0]):
                feat = D_model[i,0]
                ### Overlap Case
                if (feat>=zero_min and feat<=zero_max) and (feat<=one_max and feat>=one_min):
                    if P[i]==0:
                        domain_overlap+= 1/weight_zero
                    else:
                        domain_overlap+= 1/weight_one
                    error_labels.append(1)
                    feasible = D_model[i,:]
                else:
                    error_labels.append(0)
                    
            if type(feasible) != np.ndarray:
                volume_overlap = 0.0
            else:
                volume_overlap = min(zero_max,one_max)-max(zero_min,one_min)
            
        ### Multi_dimensional_Case
        else:
            
            try:
                ### Create Convex Hulls
                convex_hull_zero = ConvexHull(D_zero)
                convex_hull_one = ConvexHull(D_one)       

                error_labels = []
                domain_overlap = 0
                inside_hull_zero = self.in_hull(D_model, convex_hull_zero)
                inside_hull_one = self.in_hull(D_model, convex_hull_one)
                feasible = "disjoint"

                for i in range(D_model.shape[0]):
                    ### Overlap Case
                    if inside_hull_zero[i] and inside_hull_one[i]:

                        if P[i]==0:
                            domain_overlap+= 1/weight_zero
                        else:
                            domain_overlap+= 1/weight_one
                        error_labels.append(1)
                        
                        if type(feasible) != np.ndarray:
                            feasible = copy.deepcopy(D_model[i,:])
                        else:
                            feasible = feasible + (D_model[i,:]-feasible)/2.0
                    else:
                        error_labels.append(0)

                equations = np.concatenate((convex_hull_zero.equations,convex_hull_one.equations),axis=0)
                hs = HalfspaceIntersection(equations,feasible) 
                hull_three = ConvexHull(hs.intersections)
                volume_overlap = hull_three.volume/ min(convex_hull_zero.volume,convex_hull_one.volume)
            #### Catch any Errors 
            except QhullError:
                error_labels = []
                domain_overlap = D_model.shape[0]
                volume_overlap = 999999999
                    
        error_labels = np.asarray(error_labels)
                
        return domain_overlap, volume_overlap, error_labels

    def _get_domain_overlap_scores_one_d(self, D, P):
        """ 
        Perform 1D domain overlap over a feature set
        Params:
            D- np.array - Candidate Features
            P- np.array - Labels

        Returns:
            domain_overlap_scores: List[float] - Number of points in overlapping region
            volume_overlap_scores: List[float] - Volume (Length) of overlapping region
        """
        domain_overlap_scores = []
        volume_overlap_scores = []
        
        for i in range(D.shape[1]):
            curr_domain_overlap, curr_volume_overlap,new_labels = self._get_domain_overlap(D[:,i].reshape(-1,1),P)
            domain_overlap_scores.append(curr_domain_overlap)
            volume_overlap_scores.append(curr_volume_overlap)

        domain_overlap_scores = np.asarray(domain_overlap_scores)
        volume_overlap_scores = np.asarray(volume_overlap_scores)
        
        return domain_overlap_scores,volume_overlap_scores
    
    def _sis(self, D, P):
        """
        Perform SIS stage of SISSO

        Params:
            D - np.array - input features
            P - np.array - labels

        Return:
            Selected Indices - List[int] - Indices of Selected Indices
        """

        ### If considering all combinations at SO - only do SIS on new features
        if all_l0_combinations:
            domain_overlap_scores, volume_overlap_scores = self._get_domain_overlap_scores_one_d(D[:,sis_not_selected_indices],P)
        else:
            domain_overlap_scores, volume_overlap_scores = self._get_domain_overlap_scores_one_d(D,P)

        ### Sort by Domain Overlap - break ties with volume overlap
        indices_sorted = np.lexsort((volume_overlap_scores,domain_overlap_scores))
        indices_n_closest = indices_sorted[: self.n_features_per_sis_iter]
        
        if all_l0_combinations:
            indices_n_closest_out = self.sis_not_selected_indices[indices_n_closest]
            self.sis_not_selected_indices = np.delete(self.sis_not_selected_indices, indices_n_closest)
            return indices_n_closest_out
        
        return indices_n_closest
    
    ### Taken from https://stackoverflow.com/questions/16750618/whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl
    def in_hull(self, p, hull):
        """
        Test if points in `p` are in `hull`

        `p` should be a `NxK` coordinates of `N` points in `K` dimensions
        `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
        coordinates of `M` points in `K`dimensions for which Delaunay triangulation
        will be computed
        """
        from scipy.spatial import Delaunay
        if len(hull.vertices) < 4:
            hull_Del = Delaunay(hull.points)
        else:
            hull_Del = Delaunay(hull.points[hull.vertices])

        return hull_Del.find_simplex(p)>=0

