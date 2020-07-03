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

inv_logit = lambda p: np.exp(p) / (1 + np.exp(p))

def generate_binary_dataset(size, w_t, w_o, **kwargs):
    at = kwargs['at'] if 'at' in kwargs else 0
    ao = kwargs['ao'] if 'ao' in kwargs else 0

    M = size[0]
    N = size[1]

    labels = ['x{}'.format(i + 1) for i in range(N)]
    df = pd.DataFrame(np.random.binomial(1, 0.5, (M, N)), columns=labels)

    # treatment status
    df['lp_t'] = np.sum(df[labels].values * w_t, axis=1) + at
    df['p_t'] = inv_logit(df['lp_t'])
    df['treated'] = np.random.binomial(1, df['p_t'])

    # outcome
    if 'rr' in kwargs:
        df['lp_o'] = np.sum(df[labels].values * w_o, axis=1) + ao + kwargs['rr'] * df['t']

        if 'latent' in kwargs:
            df['latent'] = np.random.binomial(1, 0.5, (M, 1))
            df['lp_o'] = df['lp_o'] + kwargs['latent'] * df['latent']

        df['p_o'] = inv_logit(df['lp_o'])
        df['outcome'] = np.random.binomial(1, df['p_o'])

    return df

def separate_groups(df, col):
    return df[df[col] == 1], df[df[col] == 0]


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
    del otpt['lp_t']
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

aL = np.log10(1.1)
aM = np.log10(1.25)
aH = np.log10(1.5)
aVH = np.log10(2.0)

#latent = np.log10(1.125)
#corr_for_binary = 1.25

weights_o = [aL, aL, aL, aM, aM, aM, aH, aH, aH, aVH]
weights_t = [aL, aL, aL, aM, aM, aM, aH, aH, aH, aVH]

at = -1.34490
ao = -1.098537

X = generate_binary_dataset((M, N), weights_t, weights_o, at=at, ao=ao)

#X = pd.read_csv('sample_dataset.csv', sep=';', decimal=',', encoding='utf-8')
#X.set_index(index, inplace=True)

ovar = [x for x in list(X) if 'x' in x]
X, r_weights = psm.calculate_ps(X, 'treated', ovar)
treated, untreated = separate_groups(X, 'treated')

print('Case group: {} (No. treated)'.format(len(treated)))
print('Population: {} (No. untreated)'.format(len(untreated)))

psm_control, caliper = psm.match(treated, untreated, 'ps', pair_name='psm_pair')
print('Control group by PSM: {} ({:.2f}%)'.format(len(psm_control), len(psm_control) / len(treated) * 100))

wnnem_control = wnnem.match(treated, untreated, ovar, r_weights, pair_name='wnnem_pair')
print('Control group by WNNEM: {} ({:.2f}%)'.format(len(wnnem_control), len(wnnem_control) / len(treated) * 100))

ss_control = om.SS(treated, untreated, ovar, pair_name='ss_pair')
print('Control group by SS: {} ({:.2f}%)'.format(len(ss_control), len(ss_control) / len(treated) * 100))

nn_control = om.NN(treated, untreated, ovar, pair_name='nn_pair', dist='mahalanobis')
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