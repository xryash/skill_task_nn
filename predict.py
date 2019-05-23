import tensorflow as tf
import numpy as np
import pandas as pd

from net import get_networks_options
from oop import NeuralNet
from preprocessing import test_prepr

MODELS_DIR = 'models/'
NAME_PREFIX = 'model_'

RESULT_FILE = 'data/results.csv'


def save_predictions(ids, results):
    results = results.reshape([-1])
    d = {'sample_id': ids, 'y': results}
    print(results.shape)
    print(ids.shape)
    df = pd.DataFrame(data=d, index=None)
    df.to_csv(RESULT_FILE, sep=',', index=False)

def predict(options, x_test, ids):
    results = []
    for option in options:
        nn_number = option['number']

        model_path = MODELS_DIR + str(nn_number) + '/' + NAME_PREFIX + str(nn_number)

        with tf.Session() as session:

            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(model_path))

            saver.restore(session, model_path)
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

    options = get_networks_options()

    predict(options, x_test, ids)
