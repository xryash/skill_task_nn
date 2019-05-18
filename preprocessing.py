import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt


def _load_train_data(path):
    """Load data from csv file"""
    train_data = pd.read_csv(path, sep=',')
    train_data = train_data.replace([np.inf, -np.inf], np.nan)
    train_data = train_data.fillna(0)
    y_train = train_data.y.values
    train_data = train_data.drop('sample_id', 1)
    train_data = train_data.drop('y', 1)
    x_train = train_data.values
    return x_train, y_train


def _load_test_data(path):
    """Load data from csv file"""
    test_data = pd.read_csv(path, sep=',')
    test_data = test_data.replace([np.inf, -np.inf], np.nan)
    test_data = test_data.fillna(0)
    ids = test_data.sample_id.values
    test_data = test_data.drop('sample_id', 1)
    x_test = test_data.values
    return x_test, ids


def _split_data(x, y, validation_numb):
    """Split data on train and validation sets"""
    indexes = np.arange(len(x))
    np.random.shuffle(indexes)
    x_val = x[indexes[: int(validation_numb)]]
    x_train = x[indexes[int(validation_numb):]]
    y_val = y[indexes[: int(validation_numb)]]
    y_train = y[indexes[int(validation_numb):]]
    return x_train, y_train, x_val, y_val

def _normalize(x):
    """Normalize data  (x - mean) / std"""
    x_mean = np.mean(x)
    x_std = np.std(x)
    return (x - x_mean) / x_std


def _dim_reduction(x, n_components):
    pca = PCA(n_components=n_components).fit(x)
    x = pca.transform(x)
    return x


def test_prepr():
    """Get preprocessed test data"""
    TEST_DATA = 'data/test.csv'
    x_test, ids = _load_test_data(TEST_DATA)
    x_test = _normalize(x_test)
    return x_test, ids


def train_prepr(validation_numb=200):
    """Get preprocessed train data"""
    TRAIN_DATA = 'data/train.csv'
    x_train, y_train = _load_train_data(TRAIN_DATA)
    x_train = _normalize(x_train)
    x_train, y_train, x_val, y_val = _split_data(x_train, y_train, validation_numb)
    y_train = np.array([y_train]).transpose(1, 0)
    y_val = np.array([y_val]).transpose(1, 0)
    print([i.shape for i in [x_train, y_train, x_val, y_val]])
    return x_train, y_train, x_val, y_val


TRAIN_DATA = 'data/train.csv'

x_train, y_train = _load_train_data(TRAIN_DATA)


