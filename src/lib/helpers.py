#!/usr/bin/env python3
"""Universal kernel blocks"""
import re
import os
import time
import datetime as dt
import numpy as np
import scipy as ss
import pandas as pd
import requests

import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn import metrics

from numba import jit

###################################################################################################
# constants
###################################################################################################

PROJECT_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
SRC_PATH = PROJECT_DIR + '/src/'
DATASETS_PATH = PROJECT_DIR + '/datasets/'
DATASETS_ORIGINAL_PATH = DATASETS_PATH + 'original/'
DATASETS_DEV_PATH = DATASETS_PATH + 'dev/'
DATASETS_PRED_PATH = DATASETS_PATH + 'predictions/'

###################################################################################################
# resources optimization
###################################################################################################


def reduce_mem_usage(df, verbose=True):
    """
    Reduce memory costs of df via changing numeric column types to more efficient ones
    Takes a lot of time, try only once
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
            end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


@jit
def fast_auc(y_true, y_prob):
    """
    fast roc_auc computation: https://www.kaggle.com/c/microsoft-malware-prediction/discussion/76013
    """
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    nfalse = 0
    auc = 0
    n = len(y_true)
    for i in range(n):
        y_i = y_true[i]
        nfalse += (1 - y_i)
        auc += y_i * nfalse
    auc /= (nfalse * (n - nfalse))
    return auc


def eval_auc(y_true, y_pred):
    """
    Fast auc eval function for lgb.
    """
    return 'auc', fast_auc(y_true, y_pred), True


def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):
    """
    Fast metric computation for this competition: https://www.kaggle.com/c/champs-scalar-coupling
    Code is from this kernel: https://www.kaggle.com/uberkinder/efficient-metric
    """
    maes = (y_true-y_pred).abs().groupby(types).mean()
    return np.log(maes.map(lambda x: max(x, floor))).mean()

###################################################################################################
# descibe & visualise
###################################################################################################


def resumetable(df):
    """
    Table about table
    """
    print(f"Dataset Shape: {df.shape}")
    summary = pd.DataFrame(df.dtypes, columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name', 'dtypes']]
    summary['Missing'] = df.isnull().sum().values
    summary['Uniques'] = df.nunique().values
    summary['First Value'] = df.loc[0].values
    summary['Second Value'] = df.loc[1].values
    summary['Third Value'] = df.loc[2].values

    for name in summary['Name'].value_counts().index:
        summary.loc[summary['Name'] == name, 'Entropy'] = \
            round(ss.stats.entropy(
                df[name].value_counts(normalize=True), base=2), 2)

    return summary


###################################################################################################
# preprocessing
###################################################################################################
def my_csv_read(csv_file):
    """Solve function pickle issues
    https://stackoverflow.com/questions/8804830/
    python-multiprocessing-picklingerror-cant-pickle-type-function
    """
    return pd.read_csv(csv_file)

def get_floats_from_string(string_to_parse):
    """finds all float numbers in string"""
    res_list = re.findall(r"[-+]?\d*\.\d+|\d+", string_to_parse)

    return res_list

def none_or_first(list_to_get):
    """gets first element of list of None"""
    if list_to_get:
        return list_to_get[0]
    else:
        return None


def clean_inf_nan(df):
    """nan instead of inf"""
    return df.replace([np.inf, -np.inf], np.nan)


def add_datetime_info(df_trans, ts_column='TransactionDT', start_date=dt.datetime(2017, 12, 1)):
    """adds _Weekdays, _Hours, _Days columns to df

    Args:
        df_trans (DataFrame):
            With timestamp column.
        ts_column (string):
            Column with second.
        start_date (datetime):
            Starting point if ts_column has no full timestamp

    Returns:
        df_trans (DataFrame):
            With 4 additional columns
    """

    if start_date:
        df_trans["_Date"] = df_trans[ts_column].apply(lambda x:
                                                      (start_date + dt.timedelta(seconds=x)))
    else:
        df_trans["_Date"] = df_trans[ts_column].apply(
            dt.datetime.fromtimestamp)
    df_trans['_Weekdays'] = df_trans['_Date'].dt.dayofweek
    df_trans['_Hours'] = df_trans['_Date'].dt.hour
    df_trans['_Days'] = df_trans['_Date'].dt.day
    df_trans.drop(['_Date'], axis=1, inplace=True)

    return df_trans


def correct_card_id(x):
    """Just replacement of characters"""

    x = x.replace('.0', '')
    x = x.replace('-999', 'NNNN')
    while len(x) < 4:
        x += 'N'

    return x


def add_card_id(df):
    """Apply correct_card_id to df columns"""
    cards_cols = ['card1', 'card2', 'card3', 'card5']
    for card in cards_cols:
        if '1' in card:
            df['Card_ID'] = df[card].map(str)
        else:
            df['Card_ID'] += ' ' + df[card].map(str)

    return df


def drop_columns_nan_null(df_drop, df_look,
                          keep_cols,
                          drop_proportion=0.9):
    """ drop columns with lots of nans or without values """

    one_value_cols = [
        col for col in df_look.columns if df_look[col].nunique() <= 1]

    many_null_cols = [col for col in df_look.columns if
                      df_look[col].isnull().sum() / df_look.shape[0] > drop_proportion]

    big_top_value_cols = [col for col in df_look.columns if
                          df_look[col].value_counts(dropna=False, normalize=True).
                          values[0] > drop_proportion]

    cols_to_drop = list(set(many_null_cols +
                            big_top_value_cols +
                            one_value_cols
                            ))

    for keep_col in keep_cols:
        if keep_col in cols_to_drop:
            cols_to_drop.remove(keep_col)
    print(len(cols_to_drop), ' columns were removed because of nulls and NaNs')
    print(f'dropped ones: {cols_to_drop}')
    df_drop.drop(cols_to_drop, axis=1, inplace=True)

    return df_drop

def drop_columns_corr(df_drop, df_look,
                      keep_cols,
                      drop_threshold=0.98):
    """drop columns with high correlation
    """


    # Absolute value correlation matrix
    corr_matrix = df_look[df_look['isFraud'].notnull()].corr().abs()

    # Getting the upper triangle of correlations
    upper = corr_matrix.where(np.array(np.triu(np.ones(corr_matrix.shape), k=1)).astype(np.bool))

    # Select columns with correlations above threshold
    cols_to_drop = [column for column in upper.columns if any(upper[column] > drop_threshold)]

    for keep_col in keep_cols:
        if keep_col in cols_to_drop:
            cols_to_drop.remove(keep_col)
    print(len(cols_to_drop), ' columns were removed because of high corr')
    print(f'dropped ones: {cols_to_drop}')
    df_drop.drop(cols_to_drop, axis=1, inplace=True)

    return df_drop

###################################################################################################
# training model
###################################################################################################


def train_model_regression(X, X_test, y, params, folds=None, model_type='lgb',
                           eval_metric='mae', columns=None,
                           plot_feature_importance=False, model=None,
                           verbose=10000, early_stopping_rounds=200,
                           n_estimators=50000, splits=None, n_folds=3):
    """
    A function to train a variety of regression models.
    Returns dictionary with oof predictions, test predictions,
    scores and, if necessary, feature importances.

    :params: X - training data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: X_test - test data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: y - target
    :params: folds - folds to split data
    :params: model_type - type of model to use
    :params: eval_metric - metric to use
    :params: columns - columns to use. If None - use all columns
    :params: plot_feature_importance - whether to plot feature importance of LGB
    :params: model - sklearn model, works only for "sklearn" model type

    """
    columns = X.columns if columns is None else columns
    X_test = X_test[columns]

    # check for different Kfolds
    if str(type(folds)) == "<class 'sklearn.model_selection._split.StratifiedKFold'>":
        splits = folds.split(X, y)
    elif str(type(folds)) == "<class 'sklearn.model_selection._split.TimeSeriesSplit'>":
        splits = folds.split(X) if splits is None else splits
    else:
        splits = folds.split(X) if splits is None else splits
    n_splits = folds.n_splits if splits is None else n_folds

    # to set up scoring parameters
    metrics_dict = {'mae': {'lgb_metric_name': 'mae',
                            'catboost_metric_name': 'MAE',
                            'sklearn_scoring_function': metrics.mean_absolute_error},
                    'group_mae': {'lgb_metric_name': 'mae',
                                  'catboost_metric_name': 'MAE',
                                  'scoring_function': group_mean_log_mae},
                    'mse': {'lgb_metric_name': 'mse',
                            'catboost_metric_name': 'MSE',
                            'sklearn_scoring_function': metrics.mean_squared_error}
                    }

    result_dict = {}

    # out-of-fold predictions on train data
    oof = np.zeros(len(X))

    # averaged predictions on train data
    prediction = np.zeros(len(X_test))

    # list of scores on folds
    scores = []
    feature_importance = pd.DataFrame()

    # split and train on folds
    for fold_n, (train_index, valid_index) in enumerate(splits):
        if verbose:
            print(f'Fold {fold_n + 1} started at {time.ctime()}')
        if isinstance(X, np.ndarray):
            X_train, X_valid = X[columns][train_index], X[columns][valid_index]
            y_train, y_valid = y[train_index], y[valid_index]
        else:
            X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        if model_type == 'lgb':
            model = lgb.LGBMRegressor(
                **params, n_estimators=n_estimators, n_jobs=-1)
            model.fit(X_train, y_train,
                      eval_set=[(X_train, y_train), (X_valid, y_valid)
                                ], eval_metric=metrics_dict[eval_metric]['lgb_metric_name'],
                      verbose=verbose, early_stopping_rounds=early_stopping_rounds)

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test, num_iteration=model.best_iteration_)

        if model_type == 'xgb':
            train_data = xgb.DMatrix(
                data=X_train, label=y_train, feature_names=X.columns)
            valid_data = xgb.DMatrix(
                data=X_valid, label=y_valid, feature_names=X.columns)

            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
            model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist,
                              early_stopping_rounds=200, verbose_eval=verbose, params=params)
            y_pred_valid = model.predict(xgb.DMatrix(
                X_valid, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
            y_pred = model.predict(xgb.DMatrix(
                X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)

        if model_type == 'sklearn':
            model = model
            model.fit(X_train, y_train)

            y_pred_valid = model.predict(X_valid).reshape(-1,)
            score = metrics_dict[eval_metric]['sklearn_scoring_function'](
                y_valid, y_pred_valid)
            print(f'Fold {fold_n}. {eval_metric}: {score:.4f}.')
            print('')

            y_pred = model.predict(X_test).reshape(-1,)

        if model_type == 'cat':
            model = CatBoostRegressor(iterations=20000,
                                      eval_metric=metrics_dict
                                      [eval_metric]['catboost_metric_name'],
                                      **params,
                                      loss_function=metrics_dict
                                      [eval_metric]['catboost_metric_name'])
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid),
                      cat_features=[], use_best_model=True, verbose=False)

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test)

        oof[valid_index] = y_pred_valid.reshape(-1,)
        if eval_metric != 'group_mae':
            scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](
                y_valid, y_pred_valid))
        else:
            scores.append(metrics_dict[eval_metric]['scoring_function'](
                y_valid, y_pred_valid, X_valid['type']))

        prediction += y_pred

        if model_type == 'lgb' and plot_feature_importance:
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat(
                [feature_importance, fold_importance], axis=0)

    prediction /= n_splits
    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(
        np.mean(scores), np.std(scores)))

    result_dict['oof'] = oof
    result_dict['prediction'] = prediction
    result_dict['scores'] = scores

    if model_type == 'lgb':
        if plot_feature_importance:
            feature_importance["importance"] /= n_splits
            cols = feature_importance[["feature", "importance"]]\
                .groupby("feature").mean().sort_values(
                    by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(
                cols)]

            plt.figure(figsize=(16, 12))
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(
                by="importance", ascending=False))
            plt.title('LGB Features (avg over folds)')

            result_dict['feature_importance'] = feature_importance

    return result_dict


def train_model_classification(X, X_test, y, params, folds, model_type='lgb',
                               eval_metric='auc', columns=None,
                               plot_feature_importance=False, model=None,
                               verbose=10000, early_stopping_rounds=200,
                               n_estimators=50000, splits=None,
                               n_folds=3, averaging='usual', n_jobs=-1):
    """
    A function to train a variety of classification models.
    Returns dictionary with oof predictions, test predictions,
    scores and, if necessary, feature importances.

    :params: X - training data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: X_test - test data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: y - target
    :params: folds - folds to split data
    :params: model_type - type of model to use
    :params: eval_metric - metric to use
    :params: columns - columns to use. If None - use all columns
    :params: plot_feature_importance - whether to plot feature importance of LGB
    :params: model - sklearn model, works only for "sklearn" model type

    """
    columns = X.columns if columns is None else columns

    # check for different Kfolds
    if str(type(folds)) == "<class 'sklearn.model_selection._split.StratifiedKFold'>":
        splits = folds.split(X, y)
    elif str(type(folds)) == "<class 'sklearn.model_selection._split.TimeSeriesSplit'>":
        splits = folds.split(X) if splits is None else splits
    else:
        splits = folds.split(X) if splits is None else splits

    n_splits = folds.n_splits if splits is None else n_folds
    X_test = X_test[columns]

    # to set up scoring parameters
    metrics_dict = {'auc': {'lgb_metric_name': eval_auc,
                            'catboost_metric_name': 'AUC',
                            'sklearn_scoring_function': metrics.roc_auc_score},
                    }

    result_dict = {}
    if averaging == 'usual':
        # out-of-fold predictions on train data
        oof = np.zeros((len(X), 1))

        # averaged predictions on train data
        prediction = np.zeros((len(X_test), 1))

    elif averaging == 'rank':
        # out-of-fold predictions on train data
        oof = np.zeros((len(X), 1))

        # averaged predictions on train data
        prediction = np.zeros((len(X_test), 1))

    # list of scores on folds
    scores = []
    feature_importance = pd.DataFrame()

    # split and train on folds
    for fold_n, (train_index, valid_index) in enumerate(splits):
        print(f'Fold {fold_n + 1} started at {time.ctime()}')
        if isinstance(X, np.ndarray):
            X_train, X_valid = X[columns][train_index], X[columns][valid_index]
            y_train, y_valid = y[train_index], y[valid_index]
        else:
            X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        if model_type == 'lgb':
            model = lgb.LGBMClassifier(
                **params, n_estimators=n_estimators, n_jobs=n_jobs)
            model.fit(X_train, y_train,
                      eval_set=[(X_train, y_train), (X_valid, y_valid)
                                ], eval_metric=metrics_dict[eval_metric]['lgb_metric_name'],
                      verbose=verbose, early_stopping_rounds=early_stopping_rounds)

            y_pred_valid = model.predict_proba(X_valid)[:, 1]
            y_pred = model.predict_proba(
                X_test, num_iteration=model.best_iteration_)[:, 1]

        if model_type == 'xgb':
            train_data = xgb.DMatrix(
                data=X_train, label=y_train, feature_names=X.columns)
            valid_data = xgb.DMatrix(
                data=X_valid, label=y_valid, feature_names=X.columns)

            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
            model = xgb.train(dtrain=train_data, num_boost_round=n_estimators, evals=watchlist,
                              early_stopping_rounds=early_stopping_rounds,
                              verbose_eval=verbose, params=params)
            y_pred_valid = model.predict(xgb.DMatrix(
                X_valid, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
            y_pred = model.predict(xgb.DMatrix(
                X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)

        if model_type == 'sklearn':
            model = model
            model.fit(X_train, y_train)

            y_pred_valid = model.predict(X_valid).reshape(-1,)
            score = metrics_dict[eval_metric]['sklearn_scoring_function'](
                y_valid, y_pred_valid)
            print(f'Fold {fold_n}. {eval_metric}: {score:.4f}.')
            print('')

            y_pred = model.predict_proba(X_test)

        if model_type == 'cat':
            model = CatBoostClassifier(iterations=n_estimators,
                                       eval_metric=metrics_dict
                                       [eval_metric]['catboost_metric_name'],
                                       **params,
                                       loss_function=metrics_dict
                                       [eval_metric]['catboost_metric_name'])
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid),
                      cat_features=[], use_best_model=True, verbose=False)

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test)

        if averaging == 'usual':

            oof[valid_index] = y_pred_valid.reshape(-1, 1)
            scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](
                y_valid, y_pred_valid))

            prediction += y_pred.reshape(-1, 1)

        elif averaging == 'rank':

            oof[valid_index] = y_pred_valid.reshape(-1, 1)
            scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](
                y_valid, y_pred_valid))

            prediction += pd.Series(y_pred).rank().values.reshape(-1, 1)

        if model_type == 'lgb' and plot_feature_importance:
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat(
                [feature_importance, fold_importance], axis=0)

    prediction /= n_splits

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(
        np.mean(scores), np.std(scores)))

    result_dict['oof'] = oof
    result_dict['prediction'] = prediction
    result_dict['scores'] = scores

    if model_type == 'lgb':
        if plot_feature_importance:
            feature_importance["importance"] /= n_splits
            cols = feature_importance[["feature", "importance"]]\
                .groupby("feature").mean().sort_values(
                    by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(
                cols)]

            plt.figure(figsize=(16, 12))
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(
                by="importance", ascending=False))
            plt.title('LGB Features (avg over folds)')

            result_dict['feature_importance'] = feature_importance
            result_dict['top_columns'] = cols

    return result_dict


###################################################################################################
# training model
###################################################################################################



# telegram API
######################################################################################
TELEGRAM_TOKEN = os.environ['TELEGRAM_DS_TOKEN']
TELEGRAM_ID_LIST = [os.environ['TELEGRAM_ID']]

def send_message_to_telegram(msg='hi'):
    """in: str, out: str"""
    METHOD_NAME = 'sendMessage'

    for my_id in TELEGRAM_ID_LIST:
        data_to_send = {
            'chat_id': my_id,
            'text': msg
        }

        header = {
            'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8'
        }

        msg_req = requests.post('https://api.telegram.org/bot{}/{}'
                                .format(TELEGRAM_TOKEN, METHOD_NAME),
                                data=data_to_send,
                                headers=header
                                )
    return msg_req


def send_photo_to_telegram(photo='graph.png'):
    """in: str, out: str"""
    METHOD_NAME = '/sendPhoto'
    REQUEST_URL = 'https://api.telegram.org/bot' + TELEGRAM_TOKEN

    for my_id in TELEGRAM_ID_LIST:
        files = {'photo': (photo, open(photo, "rb"))}
        data = {'chat_id': my_id}
        msg_req = requests.post(
            REQUEST_URL + METHOD_NAME, data=data, files=files)

    return msg_req


def send_file_to_telegram(document='''/home/path_to_file'''):
    """in: str, out: str"""
    METHOD_NAME = '/sendDocument'
    REQUEST_URL = 'https://api.telegram.org/bot' + TELEGRAM_TOKEN
    for my_id in TELEGRAM_ID_LIST:
        files = {'document': (document, open(document, "rb"))}
        data = {'chat_id': my_id}

        msg_req = requests.post(
            REQUEST_URL + METHOD_NAME, data=data, files=files)

    return msg_req
