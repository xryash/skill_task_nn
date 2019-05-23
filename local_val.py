import tensorflow as tf

from net import get_networks_options
from oop import NeuralNet
from preprocessing import train_prep

from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np

MODELS_DIR = 'models/'
NAME_PREFIX = 'model_'


def local_val(options, x_test, y_test):
    results = []
    for option in options:
        # if option['skip']:
        #    continue

        nn_number = option['number']

        model_path = MODELS_DIR + str(nn_number) + '/' + NAME_PREFIX + str(nn_number)

        with tf.Session() as session:

            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(model_path))

            saver.restore(session, model_path)
            with tf.device('/device:GPU:0'):
                predictions, test_loss, test_accuracy = NeuralNet.local_val(x_test, y_test)

        print('Neural network number' + str(nn_number))
        print('Loss: {:.4f}'.format(test_loss))
        print('Accuracy: {:.2f}'.format(test_accuracy))
        print('Auc: {:.4f}'.format(roc_auc_score(y_test, predictions)))

        results.append(predictions)
    results = np.array(results)
    results = np.mean(results, axis=0)

    acc = accuracy_score(y_test, np.round(results))
    print(acc)

    auc = roc_auc_score(y_test, results)
    print(auc)


if __name__ == "__main__":

    # load data
    x_train, y_train, x_test, y_test = train_prep()

    options = get_networks_options()

    local_val(options, x_test, y_test)
