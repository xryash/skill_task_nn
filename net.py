import tensorflow as tf


def get_networks_options():
    return (
        {'number': 0, 'epochs': 175, 'learning_rate': 0.0000035, 'dropout_rate': 0.5, 'skip': False},
        {'number': 1, 'epochs': 150, 'learning_rate': 0.0000035, 'dropout_rate': 0.5, 'skip': False},
        {'number': 2, 'epochs': 200, 'learning_rate': 0.0000035, 'dropout_rate': 0.5, 'skip': False},
        {'number': 3, 'epochs': 250, 'learning_rate': 0.0000035, 'dropout_rate': 0.5, 'skip': False},

    )


def net_arch(x, dropout_rate):
    initializer = tf.contrib.layers.xavier_initializer()
    net = tf.layers.batch_normalization(x, axis=1)
    net = tf.layers.dense(net, 4096, activation=tf.nn.relu, kernel_initializer=initializer)
    net = tf.layers.batch_normalization(net, axis=1)
    net = tf.layers.dropout(net, dropout_rate)
    net = tf.layers.dense(net, 2048, activation=tf.nn.relu, kernel_initializer=initializer)
    net = tf.layers.batch_normalization(net, axis=1)
    net = tf.layers.dropout(net, dropout_rate)
    net = tf.layers.dense(net, 1, activation=None, name='output')

    return net


def ensemble_arch(x, dropout_rate):
    net = tf.layers.batch_normalization(x)
    initializer = tf.contrib.layers.xavier_initializer()

    outs = []

    for i in range(3):
        branch = tf.layers.dense(net, 2048, activation=tf.nn.relu, kernel_initializer=initializer)
        branch = tf.layers.batch_normalization(branch)
        branch = tf.layers.dropout(branch, dropout_rate)

        branch = tf.layers.dense(branch, 1024, activation=tf.nn.relu, kernel_initializer=initializer)
        branch = tf.layers.batch_normalization(branch)

        outs.append(branch)

    net = tf.keras.layers.concatenate(outs)

    net = tf.layers.batch_normalization(net)
    net = tf.layers.dropout(net, dropout_rate)

    net = tf.layers.dense(net, 1, activation=None, name='output')

    return net


"""

    net = tf.layers.dense(x,  4096, activation=tf.nn.relu, kernel_initializer=initializer)
    net = tf.layers.batch_normalization(net)
    net = tf.layers.dropout(net, dropout_rate)
    net = tf.layers.dense(net,  2048, activation=tf.nn.relu, kernel_initializer=initializer)
    net = tf.layers.batch_normalization(net)
    net = tf.layers.dropout(net, dropout_rate)
    net = tf.layers.dense(net,  1024, activation=tf.nn.relu, kernel_initializer=initializer)
    net = tf.layers.batch_normalization(net)
    net = tf.layers.dropout(net, dropout_rate)
    net = tf.layers.dense(net, 1, activation=None)
"""


