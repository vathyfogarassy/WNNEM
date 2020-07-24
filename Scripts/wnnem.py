__author__ = "Szabolcs Szekér <szeker@dcs.uni-pannon.hu>"
__copyright__ = "Copyright (C) 2019 Szabolcs Szekér"
__license__ = "Public Domain"
__version__ = "0.5"

"""
Weighted Nearest Neighbours Control Group Selection with Error Minimization.

The WNNEM method is a multivariate weighted nearest neighbours-based control group selection method. 
WNNEM method pairs the elements of the case and control groups in the original vector space of the covariates 
and the dissimilarities of the individuals are calculated as the weighted distances of the subjects. 
The weight factors are derived from a logistic regression model fitted on the status of treatment assignment.

The WNNEM method will be published in the following article:

Szekér S, Vathy-Fogarassy Á (2020) Weighted nearest neighbours-based control group selection method for observational studies. PLOS ONE 15(7): e0236531. 
https://doi.org/10.1371/journal.pone.0236531

In case of using the datasets or applying the WNNEM method, please cite the article above.


"""
# %%

import numpy as np
import scipy.spatial.distance as sc

# %%

def _normalize(sample, population, attributes, w, limits=None):
    if limits is None:
        limits = np.concatenate((sample[attributes], population[attributes]))
    
    sample_norm = sample[attributes].copy()
    sample_norm = sample_norm * w
    population_norm = population[attributes].copy()
    population_norm = population_norm * w
    
    sample_norm = (sample_norm - limits.min(axis=0)) / (limits.max(axis=0) - limits.min(axis=0))
    population_norm = (population_norm - limits.min(axis=0)) / (limits.max(axis=0) - limits.min(axis=0))    
    
    return sample_norm.values, population_norm.values


# %%
    
def _calc_error_NNCSE(row, result_indices, distance_matrix):
    "Calculates the error for a given pair"    
    current = distance_matrix[row,:].argsort()[:result_indices[row]+1][-2]
    alter = distance_matrix[row,:].argsort()[:result_indices[row]+1][-1]
    return (distance_matrix[row, alter] - distance_matrix[row, current])# / distance_matrix[row, current]


# %%
    
def _error_func_NNCSE(rows, result_indices, distance_matrix):
   "Calculates the error and returns the optimal index"      
   return np.argmax([_calc_error_NNCSE(row, result_indices, distance_matrix) for row in rows])


# %%
   
def  _update_result_NNCSE(result_vector, row, result_indices, distance_matrix):
    result_vector[row] = distance_matrix[row,:].argsort()[:result_indices[row]+1][-2]


# %%
    
def _update_result_vector_NNCSE(result_vector, rows, result_indices, distance_matrix):
    for row in rows:
        _update_result_NNCSE(result_vector, row, result_indices, distance_matrix)


# %%
            
def _unique_dist(a, b):
    w = len(a)
    abs_diff = np.absolute(a - b)
    d = np.sum(abs_diff) / w
    
    return d


# %%
    
def _NNCSE(distance_matrix):
    result_vector = np.argsort(distance_matrix, axis=1)[:,0]
    full_count = distance_matrix.shape[0]
    result_indices = np.ones(full_count, dtype=np.int32)
    res_count = np.unique(result_vector).shape[0]
    
    iteration = 1
    while(res_count != full_count):
        vals, inverse, count = np.unique(result_vector, return_inverse=True, return_counts=True)
        idx_vals_repeated = np.where(count > 1)[0]
        vals_repeated = vals[idx_vals_repeated]
        
        
        
        for val in vals_repeated:
            conflicting_indices = np.where(result_vector == val)
            conflicting_indices = [item for sublist in conflicting_indices for item in sublist]
            conflicting_indices.pop(_error_func_NNCSE(conflicting_indices, result_indices, distance_matrix))
            result_indices[conflicting_indices] = result_indices[conflicting_indices] + 1
            _update_result_vector_NNCSE(result_vector, conflicting_indices, result_indices, distance_matrix)
            
        res_count = np.unique(result_vector).shape[0]     
        iteration += 1    
    
    return result_vector


# %%
    
def match(_to, _from, ovar, w, **kwargs):
    """
    match(_to, _from, ovar, w, **kwargs)

    Pairs individuals of _to and _from.
    
    Parameters
    ----------
    _to : Pandas.DataFrame
        Case group.
    _from : Pandas.DataFrame
        Population.    
    ovar : List
        Observed variables.
    w : List
        Weights.
    **kwargs: {'pair_name'}
        index     : string (default 'index')
        Name of the index column.
        pair_name : string (default 'pair')
        Name of the pair column.
           
    Returns
    -------
    Pandas.DataFrame
        Control group.
    """
    
    pair_name = kwargs['pair_name'] if 'pair_name' in kwargs else 'pair'
    _id = kwargs['index'] if 'index' in kwargs else 'index'
    
    norm_to, norm_from = _normalize(_to, _from, ovar, w)
    d = sc.cdist(norm_to, norm_from, _unique_dist)
    
    res = _NNCSE(d)
    
    res = res[~np.isnan(res)]
    
    otpt = _from.reset_index().rename(columns={_from.index.name:_id}).loc[res]
    control = otpt.loc[res]
    
    _to['tmp'] = res
    
    for index, row in _to.iterrows():
        try:
            _to.loc[index, pair_name] = control.loc[_to.loc[index, 'tmp'], _id]
        except KeyError:
            _to.loc[index, pair_name] = None
    
    del _to['tmp']
    
    return control.set_index(_id)

