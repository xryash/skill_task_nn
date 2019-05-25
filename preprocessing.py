import os
import pickle

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

TRAIN_DATA = 'data/train.csv'
SUBM_DATA = 'data/test.csv'

PREP_TRAIN_DATA = 'data/prep_train'
PREP_TEST_DATA = 'data/prep_test'
PREP_SUBM_DATA = 'data/prep_subm_test'

N_COMPONENTS = 256


def _max(data):
    """Return average max value without nan and inf"""
    max_vals = data.max(skipna=False).values
    max_vals = max_vals[~np.isnan(max_vals)]
    max_vals = max_vals[~np.isinf(max_vals)]
    max_val = np.max(max_vals)
    return max_val


def _min(data):
    """Return average min value without nan and inf"""
    min_vals = data.min(skipna=False).values
    min_vals = min_vals[~np.isnan(min_vals)]
    min_vals = min_vals[~np.isinf(min_vals)]
    min_val = np.min(min_vals)
    return min_val


def _mean(data):
    """Return mean value without nan and inf"""
    mean_vals = data.mean(skipna=False).values
    mean_vals = mean_vals[~np.isnan(mean_vals)]
    mean_vals = mean_vals[~np.isinf(mean_vals)]
    mean_val = np.mean(mean_vals)
    return mean_val


def _get_common_columns():
    """Return columns without zeros"""
    train_data = pd.read_csv(TRAIN_DATA, sep=',')
    test_data = pd.read_csv(SUBM_DATA, sep=',')

    train_data = train_data.drop('sample_id', 1)
    train_data = train_data.drop('y', 1)
    test_data = test_data.drop('sample_id', 1)

    # if all elements in column are zero, then skip it
    train_data = train_data.loc[:, (train_data != 0).any(axis=0)]
    test_data = test_data.loc[:, (test_data != 0).any(axis=0)]

    # find count of actual features
    if len(train_data.columns) > len(test_data.columns):
        columns = train_data.columns
    else:
        columns = test_data.columns
    return columns


def _load_train_data():
    """Load data from csv file"""
    train_data = pd.read_csv(TRAIN_DATA, sep=',')
    y_train = train_data.y.values

    # get actual features
    cols = _get_common_columns()
    train_data = train_data[cols]

    mean_val = _mean(train_data)
    max_val = _max(train_data)
    min_val = _min(train_data)

    # replace nan and inf with max, min and mean
    train_data = train_data.replace([np.inf], max_val)
    train_data = train_data.replace([-np.inf], min_val)
    train_data = train_data.replace([np.nan], mean_val)

    x_train = train_data.values

    return x_train, y_train


def _load_subm_data():
    """Load data from csv file"""
    test_data = pd.read_csv(SUBM_DATA, sep=',')
    ids = test_data.sample_id.values

    # get actual features
    cols = _get_common_columns()

    test_data = test_data[cols]

    mean_val = _mean(test_data)
    max_val = _max(test_data)
    min_val = _min(test_data)

    # replace nan and inf with max, min and mean
    test_data = test_data.replace([np.inf], max_val)
    test_data = test_data.replace([-np.inf], min_val)
    test_data = test_data.replace([np.nan], mean_val)

    x_test = test_data.values

    return x_test, ids


def _split_data(x, y, test_numb):
    """Split data on train and validation sets"""
    indexes = np.arange(len(x))
    np.random.shuffle(indexes)
    x_test = x[indexes[: int(test_numb)]]
    x_train = x[indexes[int(test_numb):]]
    y_test = y[indexes[: int(test_numb)]]
    y_train = y[indexes[int(test_numb):]]
    return x_train, y_train, x_test, y_test


def _normalize(x):
    """Normalize data  (x - mean) / std"""
    x_mean = np.mean(x)
    x_std = np.std(x)
    return (x - x_mean) / x_std


def _pca(n_components):
    """Reduce data dimension"""
    x, _ = _load_train_data()
    pca = PCA(n_components=n_components).fit(x)
    return pca


def val_batch(x, y, validation_numb, seed):
    """According to seed split data on train and val datasets"""
    x_train, y_train = [], []
    x_val, y_val = [], []

    # compute number of batches
    batch_count = len(x) // validation_numb

    while seed > batch_count:
        seed = int(seed / 2)
    for batch_index in range(len(x) // validation_numb):
        # split x, y batches
        batch_x = x[batch_index * validation_numb:min((batch_index + 1) * validation_numb, len(x))]
        batch_y = y[batch_index * validation_numb:min((batch_index + 1) * validation_numb, len(y))]

        if batch_index is seed:
            x_val.extend(batch_x)
            y_val.extend(batch_y)
        else:
            x_train.extend(batch_x)
            y_train.extend(batch_y)

    return np.array(x_train), np.array(y_train), np.array(x_val), np.array(y_val)


def _save(features, labels, path):
    """Save datasets to file"""
    pickle.dump((features, labels), open(str(path), 'wb'))


def train_prep(test_numb=200):
    """Get preprocessed train data"""

    # if file does not exist, preprocess and save it
    if not (os.path.exists(PREP_TRAIN_DATA)):
        x_train, y_train = _load_train_data()

        # use PCA for reducing dimensions
        pca = _pca(N_COMPONENTS)
        x_train = pca.transform(x_train)

        x_train = _normalize(x_train)
        y_train = np.array([y_train]).transpose(1, 0)

        x_train, y_train, x_test, y_test = _split_data(x_train, y_train, test_numb)

        # save data to csv files
        _save(x_train, y_train, PREP_TRAIN_DATA)
        _save(x_test, y_test, PREP_TEST_DATA)

    # if file exists, load it
    else:
        with open(PREP_TRAIN_DATA, mode='rb') as file:
            train_dataset = pickle.load(file)
        x_train, y_train = train_dataset[0], train_dataset[1]
        with open(PREP_TEST_DATA, mode='rb') as file:
            test_dataset = pickle.load(file)
        x_test, y_test = test_dataset[0], test_dataset[1]

    print([i.shape for i in [x_train, y_train, x_test, y_test]])

    return x_train, y_train, x_test, y_test


def test_prepr():
    """Get preprocessed test data"""

    # if file does not exist, preprocess and save it
    if not (os.path.exists(PREP_SUBM_DATA)):
        # load data
        x_test, ids = _load_subm_data()

        # use PCA for reducing dimension
        pca = _pca(N_COMPONENTS)
        x_test = pca.transform(x_test)
        x_test = _normalize(x_test)

        _save(x_test, ids, PREP_SUBM_DATA)

    # if file exists, load it
    else:
        with open(PREP_SUBM_DATA, mode='rb') as file:
            subm_dataset = pickle.load(file)

        x_test, ids = subm_dataset[0], subm_dataset[1]

    print([i.shape for i in [x_test, ids]])

    return x_test, ids
