#!/usr/bin/env python3
"""Module with all data manipulations
   P.S. Can be converted to .ipynb"""

# %%
# All the imports
import warnings
import multiprocessing
from functools import partial
from sklearn.metrics import precision_score, recall_score, confusion_matrix,\
    accuracy_score, roc_auc_score, f1_score,\
    roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from bayes_opt import BayesianOptimization
import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Hyperopt modules
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_RUNNING


from src.lib import helpers as hlp


# %%
# Load datasets
datasets = [
    f'{hlp.DATASETS_ORIGINAL_PATH}train.csv',
    f'{hlp.DATASETS_ORIGINAL_PATH}test.csv',
    f'{hlp.DATASETS_ORIGINAL_PATH}structures.csv',
    f'{hlp.DATASETS_PRED_PATH}sample_submission.csv']

with multiprocessing.Pool() as pool:
    train, \
    test, \
    structures, \
    sub = pool.map(hlp.my_csv_read, datasets)

# %%
# Merging test and train
target = 'scalar_coupling_constant'
test[target] = 'test'
df = pd.concat([train, test], axis=0, sort=False)
df = df.reset_index()
# %%
# combine train, test with structures
# source  https://www.kaggle.com/seriousran/just-speed-up-calculate-distance-from-benchmark

def map_atom_info(df_train, atom_idx):
    """Merge train and structures"""
    df_train = pd.merge(df_train, structures, how='left',
                        left_on=['molecule_name', f'atom_index_{atom_idx}'],
                        right_on=['molecule_name', 'atom_index'])

    df_train = df_train.drop('atom_index', axis=1)
    df_train = df_train.rename(columns={'atom': f'atom_{atom_idx}',
                                        'x': f'x_{atom_idx}',
                                        'y': f'y_{atom_idx}',
                                        'z': f'z_{atom_idx}'})
    return df_train


df = map_atom_info(df, 0)
df = map_atom_info(df, 1)

# %%
# delete heavy parts
del train, test, structures

# calculate distances between atoms
df_p_0 = df[['x_0', 'y_0', 'z_0']].values
df_p_1 = df[['x_1', 'y_1', 'z_1']].values

df['dist'] = np.linalg.norm(df_p_0 - df_p_1, axis=1)
df['dist_x'] = (df['x_0'] - df['x_1']) ** 2
df['dist_y'] = (df['y_0'] - df['y_1']) ** 2
df['dist_z'] = (df['z_0'] - df['z_1']) ** 2


# %%
# new features
# source - https://www.kaggle.com/artgor/molecular-properties-eda-and-models
df['type_0'] = df['type'].apply(lambda x: x[0])
df['type_1'] = df['type'].apply(lambda x: x[1:])

df['dist_to_type_mean'] = df['dist'] / df.groupby('type')['dist'].transform('mean')
df['dist_to_type_0_mean'] = df['dist'] / df.groupby('type_0')['dist'].transform('mean')
df['dist_to_type_1_mean'] = df['dist'] / df.groupby('type_1')['dist'].transform('mean')
df[f'molecule_type_dist_mean'] = df.groupby(['molecule_name', 'type'])['dist'].transform('mean')

# %%
# categorical to num
for f in ['atom_0', 'atom_1', 'type_0', 'type_1', 'type']:
    lbl = LabelEncoder()
    lbl.fit(list(df[f].values))
    df[f] = lbl.transform(list(df[f].values))

# %%
# separate train and test
train, test = df[df[target] != 'test'], df[df[target]
                                           == 'test'].drop(target, axis=1)

X = train.drop(['id', 'molecule_name', 'scalar_coupling_constant'], axis=1)
Y = train['scalar_coupling_constant']
X_test = test.drop(['id', 'molecule_name'], axis=1)

del df



# %%
# check
# print(train.shape, test.shape)
# print(X.shape, Y.shape, X_test.shape)
# print(X.head(5))
# print(Y.head(5))





# %%
# train params
n_fold = 5
folds = KFold(n_splits=n_fold, shuffle=True)
# folds = TimeSeriesSplit(n_splits=n_fold)
# folds = StratifiedKFold(n_splits=n_fold, shuffle=True)

# %%
# train lgb
params = {'num_leaves': 128,
          'min_child_samples': 79,
          'objective': 'regression',
          'max_depth': 13,
          'learning_rate': 0.2,
          "boosting_type": "gbdt",
          "subsample_freq": 1,
          "subsample": 0.9,
          "bagging_seed": 11,
          "metric": 'mae',
          "verbosity": -1,
          'reg_alpha': 0.1,
          'reg_lambda': 0.3,
          'colsample_bytree': 1.0
         }

result_dict_lgb = hlp.train_model_regression(X=X, X_test=X_test, y=Y,
                                             params=params, folds=folds,
                                             model_type='lgb',
                                             eval_metric='group_mae',
                                             plot_feature_importance=True,
                                             verbose=1000,
                                             early_stopping_rounds=200,
                                             n_jobs=6,
                                             n_estimators=10000)

# %%
# save results
sub['scalar_coupling_constant'] = result_dict_lgb['prediction']
sub.to_csv(f'{hlp.DATASETS_PRED_PATH}submission.csv', index=False)
print(sub.head())
hlp.send_message_to_telegram(str(result_dict_lgb))

