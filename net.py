import tensorflow as tf


def get_networks_options():
    """Get hyperparams for each neural network"""
    return (
        {'number': 0, 'epochs': 30, 'learning_rate': 0.0001, 'dropout_rate': 0.3, 'skip': False},
        {'number': 1, 'epochs': 25, 'learning_rate': 0.0001, 'dropout_rate': 0.3, 'skip': False},
        {'number': 2, 'epochs': 20, 'learning_rate': 0.0001, 'dropout_rate': 0.3, 'skip': False},
        {'number': 3, 'epochs': 30, 'learning_rate': 0.0001, 'dropout_rate': 0.3, 'skip': False},
        {'number': 4, 'epochs': 25, 'learning_rate': 0.0001, 'dropout_rate': 0.3, 'skip': False},
    )


def ensemble_arch(x, dropout_rate):
    """Neural network ensemble tensorflow graph"""

    net = tf.layers.batch_normalization(x)
    initializer = tf.contrib.layers.xavier_initializer()

    outs = []

    # compute branch tensors and add to an array
    for i in range(3):
        branch = tf.layers.dense(net, 2048, activation=tf.nn.relu, kernel_initializer=initializer)
        branch = tf.layers.batch_normalization(branch)
        branch = tf.layers.dropout(branch, dropout_rate)
        branch = tf.layers.dense(branch, 1024, activation=tf.nn.relu, kernel_initializer=initializer)
        branch = tf.layers.batch_normalization(branch)

        outs.append(branch)

    # merge all branch tensors
    net = tf.keras.layers.concatenate(outs)

    net = tf.layers.batch_normalization(net)
    net = tf.layers.dropout(net, dropout_rate)

    net = tf.layers.dense(net, 1, activation=None, name='output')

    return net
