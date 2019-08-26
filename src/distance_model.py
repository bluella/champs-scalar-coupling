#!/usr/bin/env python3
"""Module with all data manipulations
   P.S. Can be converted to .ipynb"""

# %%
# all the imports
import multiprocessing
import math
import copy

import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt
import seaborn as sns

from lightgbm import LGBMRegressor

from src.lib import helpers as hlp


# %%
# Load datasets

ATOMIC_NUMBERS = {
    'H': 1,
    'C': 6,
    'N': 7,
    'O': 8,
    'F': 9
}
TRAIN_DTYPES = {
    'molecule_name': 'category',
    'atom_index_0': 'int8',
    'atom_index_1': 'int8',
    'type': 'category',
    'scalar_coupling_constant': 'float32'
}
STRUCTURES_DTYPES = {
    'molecule_name': 'category',
    'atom_index': 'int8',
    'atom': 'category',
    'x': 'float32',
    'y': 'float32',
    'z': 'float32'
}


train_csv = pd.read_csv(f'{hlp.DATASETS_ORIGINAL_PATH}train.csv',
                        index_col='id', dtype=TRAIN_DTYPES)
train_csv['molecule_index'] = train_csv.molecule_name.str.replace(
    'dsgdb9nsd_', '').astype('int32')
train_csv = train_csv[['molecule_index', 'atom_index_0',
                       'atom_index_1', 'type', 'scalar_coupling_constant']]

test_csv = pd.read_csv(f'{hlp.DATASETS_ORIGINAL_PATH}/test.csv',
                       index_col='id', dtype=TRAIN_DTYPES)
test_csv['molecule_index'] = test_csv['molecule_name'].str.replace(
    'dsgdb9nsd_', '').astype('int32')
test_csv = test_csv[['molecule_index', 'atom_index_0', 'atom_index_1', 'type']]

structures_csv = pd.read_csv(f'{hlp.DATASETS_ORIGINAL_PATH}/structures.csv',
                             dtype=STRUCTURES_DTYPES)
structures_csv['molecule_index'] = structures_csv.molecule_name.str.replace('dsgdb9nsd_', '')\
    .astype('int32')
structures_csv = structures_csv[[
    'molecule_index', 'atom_index', 'atom', 'x', 'y', 'z']]
structures_csv['atom'] = structures_csv['atom'].replace(
    ATOMIC_NUMBERS).astype('int8')

submission_csv = pd.read_csv(
    f'{hlp.DATASETS_PRED_PATH}/sample_submission.csv', index_col='id')

# %%
# Add number of atoms

n_atoms_tr = pd.DataFrame(train_csv['molecule_index'].value_counts().sort_values())
n_atoms_tst = pd.DataFrame(test_csv['molecule_index'].value_counts().sort_values())

n_atoms_tr.columns = ['n_atoms']
n_atoms_tr.index.names = ['molecule_index']
n_atoms_tst.columns = ['n_atoms']
n_atoms_tst.index.names = ['molecule_index']

train_csv = train_csv.join(n_atoms_tr, on='molecule_index')
test_csv = test_csv.join(n_atoms_tst, on='molecule_index')

# %%
# a lot of foreign functions

def build_type_dataframes(base, structures, coupling_type):
    base = base[base['type'] == coupling_type].drop('type', axis=1).copy()
    base = base.reset_index()
    base['id'] = base['id'].astype('int32')
    structures = structures[structures['molecule_index'].isin(
        base['molecule_index'])]
    return base, structures


def add_coordinates(base, structures, index):
    df = pd.merge(base, structures, how='inner',
                  left_on=['molecule_index', f'atom_index_{index}'],
                  right_on=['molecule_index', 'atom_index']).drop(['atom_index'], axis=1)
    df = df.rename(columns={
        'atom': f'atom_{index}',
        'x': f'x_{index}',
        'y': f'y_{index}',
        'z': f'z_{index}'
    })
    return df


def add_atoms(base, atoms):
    df = pd.merge(base, atoms, how='inner',
                  on=['molecule_index', 'atom_index_0', 'atom_index_1'])
    return df


def merge_all_atoms(base, structures):
    df = pd.merge(base, structures, how='left',
                  left_on=['molecule_index'],
                  right_on=['molecule_index'])
    df = df[(df.atom_index_0 != df.atom_index) &
            (df.atom_index_1 != df.atom_index)]
    return df


def add_center(df):
    df['x_c'] = ((df['x_1'] + df['x_0']) * np.float32(0.5))
    df['y_c'] = ((df['y_1'] + df['y_0']) * np.float32(0.5))
    df['z_c'] = ((df['z_1'] + df['z_0']) * np.float32(0.5))


def add_distance_to_center(df):
    df['d_c'] = ((
        (df['x_c'] - df['x'])**np.float32(2) +
        (df['y_c'] - df['y'])**np.float32(2) +
        (df['z_c'] - df['z'])**np.float32(2)
    )**np.float32(0.5))


def add_distance_between(df, suffix1, suffix2):
    df[f'd_{suffix1}_{suffix2}'] = ((
        (df[f'x_{suffix1}'] - df[f'x_{suffix2}'])**np.float32(2) +
        (df[f'y_{suffix1}'] - df[f'y_{suffix2}'])**np.float32(2) +
        (df[f'z_{suffix1}'] - df[f'z_{suffix2}'])**np.float32(2)
    )**np.float32(0.5))


def add_distances(df):
    n_atoms = 1 + max([int(c.split('_')[1])
                       for c in df.columns if c.startswith('x_')])

    for i in range(1, n_atoms):
        for vi in range(min(4, i)):
            add_distance_between(df, i, vi)


def add_n_atoms(base, structures):
    dfs = structures['molecule_index'].value_counts().rename(
        'n_atoms').to_frame()
    return pd.merge(base, dfs, left_on='molecule_index', right_index=True)


def build_couple_dataframe(some_csv, structures_arg, coupling_type, n_atoms=10):
    base, structures = build_type_dataframes(
        some_csv, structures_arg, coupling_type)
    base = add_coordinates(base, structures, 0)
    base = add_coordinates(base, structures, 1)

    base = base.drop(['atom_0', 'atom_1'], axis=1)
    atoms = base.drop('id', axis=1).copy()
    if 'scalar_coupling_constant' in some_csv:
        atoms = atoms.drop(['scalar_coupling_constant'], axis=1)

    add_center(atoms)
    atoms = atoms.drop(['x_0', 'y_0', 'z_0', 'x_1', 'y_1', 'z_1'], axis=1)

    atoms = merge_all_atoms(atoms, structures)

    add_distance_to_center(atoms)

    atoms = atoms.drop(['x_c', 'y_c', 'z_c', 'atom_index'], axis=1)
    atoms.sort_values(['molecule_index', 'atom_index_0',
                       'atom_index_1', 'd_c'], inplace=True)
    atom_groups = atoms.groupby(
        ['molecule_index', 'atom_index_0', 'atom_index_1'])
    atoms['num'] = atom_groups.cumcount() + 2
    atoms = atoms.drop(['d_c'], axis=1)
    atoms = atoms[atoms['num'] < n_atoms]

    atoms = atoms.set_index(
        ['molecule_index', 'atom_index_0', 'atom_index_1', 'num']).unstack()
    atoms.columns = [f'{col[0]}_{col[1]}' for col in atoms.columns]
    atoms = atoms.reset_index()

    # downcast back to int8
    for col in atoms.columns:
        if col.startswith('atom_'):
            atoms[col] = atoms[col].fillna(0).astype('int8')

    atoms['molecule_index'] = atoms['molecule_index'].astype('int32')

    full = add_atoms(base, atoms)
    add_distances(full)

    full.sort_values('id', inplace=True)

    return full


def take_n_atoms(df, n_atoms, four_start=4):
    labels = []
    for i in range(2, n_atoms):
        label = f'atom_{i}'
        labels.append(label)

    for i in range(n_atoms):
        num = min(i, 4) if i < four_start else 4
        for j in range(num):
            labels.append(f'd_{i}_{j}')
    if 'scalar_coupling_constant' in df:
        labels.append('scalar_coupling_constant')
    return df[labels]


# %%
# sandbox part

# create dataset for one type
full_df = build_couple_dataframe(train_csv, structures_csv, '3JHN', n_atoms=12)

# add nans count as feature
full_df['nulls'] = full_df.isnull().sum(axis=1)

# sanity check
# col = 'd_13_0'
# print(full_df[[col]].info())
# print(full_df[[col]].describe())
# print(full_df.columns)

# fill nans
full_df = full_df.fillna(-999)
# optional change in number of atoms
# df = take_n_atoms(full_df, 7)

# create & train model
X_data = full_df.drop(['scalar_coupling_constant'],
                      axis=1).values.astype('float32')
y_data = full_df['scalar_coupling_constant'].values.astype('float32')

X_train, X_val, y_train, y_val = train_test_split(
    X_data, y_data, test_size=0.2, random_state=128)
X_train.shape, X_val.shape, y_train.shape, y_val.shape

# configuration params are copied from @artgor kernel:
# https://www.kaggle.com/artgor/brute-force-feature-engineering
LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'mae',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'learning_rate': 0.2,
    'num_leaves': 128,
    'min_child_samples': 79,
    'max_depth': 9,
    'subsample_freq': 1,
    'subsample': 0.9,
    'bagging_seed': 11,
    'reg_alpha': 0.1,
    'reg_lambda': 0.3,
    'colsample_bytree': 1.0
}

model = LGBMRegressor(**LGB_PARAMS, n_estimators=1500, n_jobs=hlp.CPU_COUNT-1)
model.fit(X_train, y_train,
          eval_set=[(X_train, y_train), (X_val, y_val)], eval_metric='mae',
          verbose=100, early_stopping_rounds=200)

y_pred = model.predict(X_val)
print(np.log(mean_absolute_error(y_val, y_pred)))


# %%
# plot FE
cols = list(full_df.columns)
cols.remove('scalar_coupling_constant')
df_importance = pd.DataFrame({'feature': cols, 'importance': model.feature_importances_})
# sns.barplot(x="importance", y="feature",
#             data=df_importance.sort_values('importance', ascending=False));

display(df_importance.sort_values('importance', ascending=False))
# %%
# # train pipeline
LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'mae',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'learning_rate': 0.2,
    'num_leaves': 128,
    'min_child_samples': 79,
    'max_depth': 9,
    'subsample_freq': 1,
    'subsample': 0.9,
    'bagging_seed': 11,
    'reg_alpha': 0.1,
    'reg_lambda': 0.3,
    'colsample_bytree': 1.0
}
ATOMS_N = 20

MODEL_PARAMS = {
    '1JHN': 10,
    '1JHC': 15,
    '2JHH': 12,
    '2JHN': 8,
    '2JHC': 12,
    '3JHH': 11,
    '3JHC': 10,
    '3JHN': 10
}

N_FOLDS = 5


def build_x_y_data(some_csv, coupling_type, n_atoms, fill_value=-999, count_nans=True):
    full = build_couple_dataframe(
        some_csv, structures_csv, coupling_type, n_atoms=n_atoms)

    if count_nans:
        full['nulls'] = full.isnull().sum(axis=1)

    df = take_n_atoms(full, n_atoms)
    df = df.fillna(fill_value)
    print(df.columns)

    if 'scalar_coupling_constant' in df:
        X_data = df.drop(['scalar_coupling_constant'],
                         axis=1).values.astype('float32')
        y_data = df['scalar_coupling_constant'].values.astype('float32')
    else:
        X_data = df.values.astype('float32')
        y_data = None

    return X_data, y_data


def train_and_predict_for_one_coupling_type(coupling_type, submission, n_atoms,
                                            n_folds=5,
                                            n_splits=5,
                                            random_state=128):
    print(f'*** Training Model for {coupling_type} ***')

    X_data, y_data = build_x_y_data(train_csv, coupling_type, n_atoms)
    X_test, _ = build_x_y_data(test_csv, coupling_type, n_atoms)
    y_pred = np.zeros(X_test.shape[0], dtype='float32')

    cv_score = 0

    if n_folds > n_splits:
        n_splits = n_folds

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold, (train_index, val_index) in enumerate(kfold.split(X_data, y_data)):
        if fold >= n_folds:
            break

        X_train, X_val = X_data[train_index], X_data[val_index]
        y_train, y_val = y_data[train_index], y_data[val_index]

        model = LGBMRegressor(
            **LGB_PARAMS, n_estimators=1500, n_jobs=hlp.CPU_COUNT-1)
        model.fit(X_train, y_train,
                  eval_set=[(X_train, y_train), (X_val, y_val)
                            ], eval_metric='mae',
                  verbose=100, early_stopping_rounds=200)

        y_val_pred = model.predict(X_val)
        val_score = np.log(mean_absolute_error(y_val, y_val_pred))
        print(f'{coupling_type} Fold {fold}, logMAE: {val_score}')

        cv_score += val_score / n_folds
        y_pred += model.predict(X_test) / n_folds

    submission.loc[test_csv['type'] == coupling_type,
                   'scalar_coupling_constant'] = y_pred
    return cv_score


# %%
# actual train
submission_tmp = submission_csv.copy()
cv_scores = {}

for coupling_type_item in MODEL_PARAMS:
    tmp_cv_score = train_and_predict_for_one_coupling_type(
        coupling_type_item,
        submission_tmp,
        n_atoms=MODEL_PARAMS[coupling_type_item],
        n_folds=N_FOLDS)
    cv_scores[coupling_type_item] = tmp_cv_score

# %%
# print results
scores_df = pd.DataFrame({'type': list(cv_scores.keys()),
                          'cv_score': list(cv_scores.values())})
print(scores_df)
print(scores_df.mean())
hlp.send_message_to_telegram(f'Model for {hlp.PROJECT_DIR} was trained!')
hlp.send_message_to_telegram(str(scores_df))
hlp.send_message_to_telegram(str(scores_df.mean()))

# %%
# save results
print(submission_tmp.head())
submission_tmp.to_csv(f'{hlp.DATASETS_PRED_PATH}submission.csv')




#%%
