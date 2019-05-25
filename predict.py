import tensorflow as tf
import numpy as np
import pandas as pd

from constants import RESULT_FILE, MODELS_DIR, NAME_PREFIX
from net import get_networks_options
from oop import NeuralNet
from preprocessing import test_prepr


def save_predictions(ids, results):
    """Save predictions to csv file"""

    # reshape nd array to 1d array
    results = results.reshape([-1])
    d = {'sample_id': ids, 'y': results}

    # create a dataframe and save to file
    df = pd.DataFrame(data=d, index=None)
    df.to_csv(RESULT_FILE, sep=',', index=False)


def predict(options, x_test, ids):
    """Predict function for submition data"""
    results = []
    for option in options:
        nn_number = option['number']

        model_path = MODELS_DIR + str(nn_number) + '/' + NAME_PREFIX + str(nn_number)

        with tf.Session() as session:
            # load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(model_path))

            # restore a model
            saver.restore(session, model_path)

            # do predictions with gpu device
            with tf.device('/device:GPU:0'):
                predictions = NeuralNet.predict(x_test)

        results.append(predictions)
    results = np.array(results)
    results = np.mean(results, axis=0)
    print('Saving results....')
    save_predictions(ids, results)
    print('Saving complete')


if __name__ == "__main__":
    # load data
    x_test, ids = test_prepr()

    # get networks hyperparams
    options = get_networks_options()

    predict(options, x_test, ids)
