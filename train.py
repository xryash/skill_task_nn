import tensorflow as tf

from constants import MODELS_DIR, PLOTS_DIR, NAME_PREFIX, BATCH_SIZE, TEST_NUMB, VALIDATION_NUMB
from net import get_networks_options
from oop import NeuralNet
from preprocessing import train_prep, val_batch
from utils import loss_plot, accuracy_plot, check_dir


def train(options, features_count, labels_count, batch_size, validation_numb, x, y):
    for option in options:
        if option['skip']:
            continue

        nn_numb = option['number']
        epochs = option['epochs']
        learning_rate = option['learning_rate']
        dropout_rate = option['dropout_rate']

        print('Neural network number' + str(nn_numb) + '\n')

        x_train, y_train, x_val, y_val = val_batch(x, y, validation_numb, seed=nn_numb)

        # remove previous weights, bias, inputs, etc..
        tf.reset_default_graph()

        with tf.Graph().as_default() as graph:
            init = tf.global_variables_initializer()
            network = NeuralNet(dropout_rate, features_count, labels_count, epochs, batch_size, learning_rate)

        with tf.Session(graph=graph) as session:
            session.run(init)
            with tf.device('/device:GPU:0'):
                print('Training....\n')
                train_loss, test_loss, train_accuracy, test_accuracy = network.training(x_train, y_train, x_val, y_val)

            print('Training complete\n')

            model_path = MODELS_DIR + str(nn_numb) + '/' + NAME_PREFIX + str(nn_numb)
            loss_name_plot = 'loss_' + str(nn_numb) + '.png'
            accuracy_name_plot = 'accuracy__' + str(nn_numb) + '.png'

            print('Saving plots....\n')

            # save plots
            accuracy_plot(train_accuracy, test_accuracy, PLOTS_DIR + accuracy_name_plot)
            loss_plot(train_loss, test_loss, PLOTS_DIR + loss_name_plot)

            print('Saving plots complete\n')

            print('Saving model....\n')

            network.save_model(model_path)
            print('Saving model complete \n')

    print('Training complete')


if __name__ == "__main__":
    # load data
    x_train, y_train, x_test, y_test = train_prep(TEST_NUMB)

    # check model dir
    check_dir(MODELS_DIR)

    # check plots dir
    check_dir(PLOTS_DIR)

    features_count = x_train.shape[1]
    labels_count = y_train.shape[1]

    # get network options
    options = get_networks_options()

    train(options, features_count, labels_count, BATCH_SIZE, VALIDATION_NUMB, x_train, y_train)
