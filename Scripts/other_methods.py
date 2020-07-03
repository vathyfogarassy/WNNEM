import numpy as np
import scipy.spatial.distance as sc
import random

# %%

def convert(row):
    return ''.join([str(item) for item in row])


# %%

def SS(case, population, ovar, **kwargs):
    """
    SS(case, population, ovar, **kwargs)

    Pairs individuals of case and population.
    
    Parameters
    ----------
    case : Pandas.DataFrame
        Case group.
    population : Pandas.DataFrame
        Population.    
    ovar : List
        Observed variables.
    **kwargs: {'pair_name'}
        pair_name : string (default 'pair')
        Name of the pair column.
           
    Returns
    -------
    Pandas.DataFrame
        Control group.
    """
       
    case_dict = {}
    
    for index, row in case[ovar].iterrows():
        key = convert(row)
        if key not in case_dict: case_dict[key] = []
        case_dict[key].append(index)
    
    pop_dict = {}

    for index, row in population[ovar].iterrows():
        key = convert(row)
        if key not in pop_dict: pop_dict[key] = []
        pop_dict[key].append(index)
    
    pair_name = kwargs['pair_name'] if 'pair_name' in kwargs else 'pair'
    case[pair_name] = np.nan
        
    result_idx = []
    for key in case_dict:
        for case_id in case_dict[key]:
            if key in pop_dict and pop_dict[key]:
                pair = pop_dict[key].pop(random.randrange(len(pop_dict[key])))
                result_idx.append(pair)  
                
                case.loc[case_id, pair_name] = pair
    
    return population.loc[result_idx]


# %%
    
def NN(case, population, ovar, **kwargs):
    """
    NN(case, population, ovar, **kwargs)

    Pairs individuals of case and populatio.
    
    Parameters
    ----------
    case : Pandas.DataFrame
        Case group.
    population : Pandas.DataFrame
        Population.    
    ovar : List
        Observed variables.
    **kwargs: {'pair_name'}
        pair_name : string (default 'pair')
        Name of the pair column.
           
    Returns
    -------
    Pandas.DataFrame
        Control group.
    """
    
    pair_name = kwargs['pair_name'] if 'pair_name' in kwargs else 'pair'
    dist = kwargs['dist'] if 'dist' in kwargs else 'euclidean'
    _id = kwargs['index'] if 'index' in kwargs else 'index'
    
    d = sc.cdist(case[ovar], population[ovar], dist)
    
    result_idx = np.empty(case.shape[0])
    result_idx[:] = np.nan
    result_idx = result_idx.reshape(-1, 1)
    
    d_idx_sorted = np.argsort(d, axis=1)
    
    matching_order = list(range(case.shape[0]))
    random.shuffle(matching_order)

    for treated in matching_order:
        actual = 0
        while np.any(result_idx == d_idx_sorted[treated][actual]):
            actual = actual + 1
        untreated = d_idx_sorted[treated][actual]
        result_idx[treated] = untreated  
    
    case[pair_name] = result_idx
    case[pair_name][case[pair_name].duplicated()] = np.nan
    
    control = population.reset_index().loc[case[pair_name].dropna()]
    
    case['tmp'] = case[pair_name]
    
    for index, row in case.iterrows():
        try:
            case.loc[index, pair_name] = control.loc[case.loc[index, 'tmp'], _id]
        except KeyError:
            case.loc[index, pair_name] = None
    
    del case['tmp']
    
    return control.set_index(_id)
