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
# reduce sample size for test
# train = train[:10000]
# test = train[:10000]
# print(train.shape, test.shape)


# %%
# print datasets info

# print(f"Train dataset has {df[df[target]!='test'].shape[0]} rows and \
# {df[df[target]!='test'].shape[1]} columns.")
# print(f"Test dataset has {df[df[target]=='test'].shape[0]} rows and \
# {df[df[target]=='test'].shape[1]} columns.")

# print(df.head(5))
# # for column in df.columns:
# #     print(column)
# print(df.info())
# print(df.describe())

# # count Nans for every column
# for column in df.columns:
#     print(df[column].isnull().sum(), column)




# %%
# display(df['addr2'].value_counts().head(50))
# # print(df[['addr2']][df['addr2'] < 101])

# train['id_11'].value_counts(dropna=False, normalize=True).head()
# sns.distplot(train['id_07'].dropna())
# plt.title('Distribution of id_07 variable')

# %%
# compare train and test distributions

# sns.distplot(train['TransactionDT'], label='train')
# sns.distplot(test['TransactionDT'], label='test')
# plt.legend()
# plt.title('Distribution of transactiond dates')

# %%
# check distos for specific set of columns
# d_cols = [c for c in train_transaction if c[0] == 'D']
# train[d_cols].head()
# sns.pairplot(train,
#              hue=target,
#             vars=d_cols)
# plt.show()

# %%
# drop columns



# %%


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
                                             n_estimators=10000)

# %%
# save results
sub['scalar_coupling_constant'] = result_dict_lgb['prediction']
sub.to_csv(f'{hlp.DATASETS_PRED_PATH}submission.csv', index=False)
print(sub.head())
hlp.send_message_to_telegram(str(result_dict_lgb))

# %%
# blending
# blend_base = pd.read_csv(f'{hlp.DATASETS_PRED_PATH}submission_TST_09393_CVM_09297.csv')
# print(blend_base.head())
# print(blend_base.describe())

# sub[target] = (sub[target] + blend_base[target])/2
# sub.to_csv(f'{hlp.DATASETS_PRED_PATH}submission.csv', index=False)


# %%
# get most important features
# print(result_dict_lgb['feature_importance'].groupby('feature').mean(
# ).sort_values('importance', ascending=False).head(50)['importance'].index)


# %%
# optimization part
# def objective(model_params):
#     """Function to optimize"""
#     params = {
#         'max_depth': int(model_params['max_depth']),
#         # 'gamma': "{:.3f}".format(model_params['gamma']),
#         'subsample': int(model_params['max_depth']),
#         'subsample_freq': int(model_params['subsample_freq']),
#         'reg_alpha': float(model_params['reg_alpha']),
#         'reg_lambda': float(model_params['reg_lambda']),
#         'learning_rate': float(model_params['learning_rate']),
#         'num_leaves': int(model_params['num_leaves']),
#         'colsample_bytree': int(model_params['colsample_bytree']),
#         'min_child_samples': int(model_params['min_child_samples']),
#         'feature_fraction': float(model_params['feature_fraction']),
#         'bagging_fraction': float(model_params['bagging_fraction']),
#         'boosting_type': 'gbdt',
#         'verbosity': -1,
#         'metric': 'auc',
#         'objective': 'binary',
#         'device_type': 'gpu'
#     }

#     # print("\n############## New Run ################")
#     # print(f"params = {params}")
#     results_dict_lgb = hlp.train_model_classification(X=X,
#                                                       X_test=X_test, y=Y,
#                                                       params=params, folds=folds,
#                                                       model_type='lgb',
#                                                       eval_metric='auc',
#                                                       plot_feature_importance=False,
#                                                       verbose=500,
#                                                       early_stopping_rounds=100,
#                                                       n_estimators=5000,
#                                                       averaging='usual',
#                                                       n_jobs=6)

#     print(results_dict_lgb['scores'])
#     return -np.mean(results_dict_lgb['scores'])


# space = {
#     # The maximum depth of a tree, same as GBM.
#     # Used to control over-fitting as higher depth will allow model
#     # to learn relations very specific to a particular sample.
#     # Should be tuned using CV.
#     # Typical values: 3-10
#     'max_depth': hp.quniform('max_depth', 7, 23, 1),

#     # reg_alpha: L1 regularization term. L1 regularization encourages sparsity
#     # (meaning pulling weights to 0). It can be more useful when the objective
#     # is logistic regression since you might need help with feature selection.
#     'reg_alpha':  hp.uniform('reg_alpha', 0.1, 1.9),

#     # reg_lambda: L2 regularization term. L2 encourages smaller weights, this
#     # approach can be more useful in tree-models where zeroing
#     # features might not make much sense.
#     'reg_lambda': hp.uniform('reg_lambda', 0.1, 1.),

#     # eta: Analogous to learning rate in GBM
#     # Makes the model more robust by shrinking the weights on each step
#     # Typical final values to be used: 0.01-0.2
#     'learning_rate': hp.uniform('learning_rate', 0.003, 0.2),

#     # colsample_bytree: Similar to max_features in GBM. Denotes the
#     # fraction of columns to be randomly samples for each tree.
#     # Typical values: 0.5-1
#     'colsample_bytree': hp.uniform('colsample_bytree', 0.1, .9),

#     # A node is split only when the resulting split gives a positive
#     # reduction in the loss function. Gamma specifies the
#     # minimum loss reduction required to make a split.
#     # Makes the algorithm conservative. The values can vary
#     # depending on the loss function and should be tuned.
#     # 'gamma': hp.uniform('gamma', 0.01, .7),

#     # more increases accuracy, but may lead to overfitting.
#     # num_leaves: the number of leaf nodes to use. Having a large number
#     # of leaves will improve accuracy, but will also lead to overfitting.
#     'num_leaves': hp.choice('num_leaves', list(range(20, 500, 10))),

#     # specifies the minimum samples per leaf node.
#     # the minimum number of samples (data) to group into a leaf.
#     # The parameter can greatly assist with overfitting: larger sample
#     # sizes per leaf will reduce overfitting (but may lead to under-fitting).
#     'min_child_samples': hp.choice('min_child_samples', list(range(100, 500, 10))),

#     # subsample: represents a fraction of the rows (observations) to be
#     # considered when building each subtree. Tianqi Chen and Carlos Guestrin
#     # in their paper A Scalable Tree Boosting System recommend
#     'subsample': hp.uniform('subsample', 0.1, .9),

#     'subsample_freq': hp.choice('subsample_freq', list(range(0, 9, 1))),
#     # randomly select a fraction of the features.
#     # feature_fraction: controls the subsampling of features used
#     # for training (as opposed to subsampling the actual training data in
#     # the case of bagging). Smaller fractions reduce overfitting.
#     'feature_fraction': hp.uniform('feature_fraction', 0.1, .9),

#     # randomly bag or subsample training data.
#     'bagging_fraction': hp.uniform('bagging_fraction', 0.1, .9),
#     # bagging_fraction and bagging_freq: enables bagging (subsampling)
#     # of the training data. Both values need to be set for bagging to be used.
#     # The frequency controls how often (iteration) bagging is used. Smaller
#     # fractions and frequencies reduce overfitting.
# }

# # Set algoritm parameters
# best = fmin(fn=objective,
#             space=space,
#             algo=tpe.suggest,
#             max_evals=30)

# # Print best parameters
# best_params = space_eval(space, best)
# print("BEST PARAMS: ", best_params)

# %%
# test cell
# hlp.send_message_to_telegram('ouch')


#%%
