__author__ = "Szabolcs Szekér <szeker@dcs.uni-pannon.hu>"
__version__ = "1.0"

# %%

import sys
sys.path.insert(1, 'Scripts')

import warnings
import numpy as np
import pandas as pd
import wnnem
import psm
import other_methods as om
import dissim
import time
import datetime

# %%

# display deprecation warnings
warnings.simplefilter('always', DeprecationWarning)

# %%
    
def create_output_file(case, population, ovar, col_list, write_to_file = True):
    
    tmp = population.copy()
    
    for col in col_list:
        tmp[col] = np.NaN
        
        for i, row in tmp.iterrows():
            try:
               tmp.loc[i, col] = case.loc[case[col] == i].index.astype(int)[0]
            except IndexError:
                pass
    
    otpt = pd.concat([case, tmp])
    
    if 'lp_t' in otpt.columns:
        del otpt['lp_t']
    if 'p_t' in otpt.columns:
        del otpt['p_t']
    otpt.index.name = '_id'
    
    if write_to_file:
        ts = time.time()
        timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')
        otpt.sort_index(axis=0).to_csv('results_{}.csv'.format(timestamp), sep=';', decimal=',')
    
    return otpt


# %%
  
M = 1000
N = 10

index = '_id'

X = pd.read_csv('Datasets/example_dataset_scenarioI.csv', sep=';', decimal=',', encoding='utf-8')
X.set_index(index, inplace=True)

ovar = [x for x in list(X) if 'x' in x]
X, r_weights = psm.calculate_ps(X, 'treated', ovar)
treated, untreated = separate_groups(X, 'treated')

print('Case group: {} (No. treated)'.format(len(treated)))
print('Population: {} (No. untreated)'.format(len(untreated)))

psm_control, caliper = psm.match(treated, untreated, 'ps', pair_name='psm_pair', index='_id')
print('Control group by PSM: {} ({:.2f}%)'.format(len(psm_control), len(psm_control) / len(treated) * 100))

wnnem_control = wnnem.match(treated, untreated, ovar, r_weights, pair_name='wnnem_pair', index='_id')
print('Control group by WNNEM: {} ({:.2f}%)'.format(len(wnnem_control), len(wnnem_control) / len(treated) * 100))

ss_control = om.SS(treated, untreated, ovar, pair_name='ss_pair')
print('Control group by SS: {} ({:.2f}%)'.format(len(ss_control), len(ss_control) / len(treated) * 100))

nn_control = om.NN(treated, untreated, ovar, pair_name='nn_pair', dist='mahalanobis', index='_id')
print('Control group by NN: {} ({:.2f}%)'.format(len(nn_control), len(nn_control) / len(treated) * 100))

categorical = ovar
groupby = 'treated'
nonnormal = None

print('\nCalculating individual balances')
balance = dissim.calculate_individual_balance(treated, wnnem_control, ovar, categorical, groupby, nonnormal)
print(balance)

print('\nEvaluating WNNEM')

ranges = [None for i in range(N)]
bins = [2 for i in range(N)]

ddi = dissim.DDI(treated, wnnem_control, ovar, ranges, bins)
print('DDI: {:.3f} (lower is better)'.format(ddi))

attribute_types = ['b' for i in range(N)]

nni = dissim.NNI(treated, untreated, ovar, attribute_types, pair_name='wnnem_pair')
print('NNI: {:.3f} (lower is better)'.format(nni))

gdi = dissim.GDI(treated, untreated, ovar, attribute_types, pair_name='wnnem_pair', weights=r_weights)
print('GDI: {:.3f} (lower is better)'.format(gdi))

print('\nEvaluating PSM')

ddi = dissim.DDI(treated, psm_control, ovar, ranges, bins)
print('DDI: {:.3f} (lower is better)'.format(ddi))

nni = dissim.NNI(treated, untreated, ovar, attribute_types, pair_name='psm_pair')
print('NNI: {:.3f} (lower is better)'.format(nni))

gdi = dissim.GDI(treated, untreated, ovar, attribute_types, pair_name='psm_pair', weights=r_weights)
print('GDI: {:.3f} (lower is better)'.format(gdi))

print('\nEvaluating SS')

ddi = dissim.DDI(treated, ss_control, ovar, ranges, bins)
print('DDI: {:.3f} (lower is better)'.format(ddi))

print('\nEvaluating NN')

ddi = dissim.DDI(treated, nn_control, ovar, ranges, bins)
print('DDI: {:.3f} (lower is better)'.format(ddi))

nni = dissim.NNI(treated, untreated, ovar, attribute_types, pair_name='nn_pair')
print('NNI: {:.3f} (lower is better)'.format(nni))

gdi = dissim.GDI(treated, untreated, ovar, attribute_types, pair_name='nn_pair', weights=r_weights)
print('GDI: {:.3f} (lower is better)'.format(gdi))

print('\nCalculating SMD\n')

test = pd.DataFrame()
test = test.rename(columns={test.index.name:'var'})

test['bef_SMD'] = dissim.calculate_SMD(X, 'treated', ovar)

tmp = pd.concat([treated, psm_control])
test['psm_SMD'] = dissim.calculate_SMD(tmp, 'treated', ovar)

tmp = pd.concat([treated, wnnem_control])
test['wnnem_SMD'] = dissim.calculate_SMD(tmp, 'treated', ovar)

tmp = pd.concat([treated, ss_control])
test['ss_SMD'] = dissim.calculate_SMD(tmp, 'treated', ovar)

tmp = pd.concat([treated, nn_control])
test['nn_SMD'] = dissim.calculate_SMD(tmp, 'treated', ovar)

test['psm_diff'] = abs(test['psm_SMD'] - test['bef_SMD'])
test['wnnem_diff'] = abs(test['wnnem_SMD'] - test['bef_SMD'])

#test['method_diff'] = test['psm_diff'] - test['wnnem_diff']

print(test)

needed = ['wnnem_pair', 'psm_pair', 'ss_pair', 'nn_pair']
otpt = create_output_file(treated, untreated, ovar, needed, False)