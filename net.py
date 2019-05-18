import tensorflow as tf


def net(x, dropout_rate):
    initializer = tf.contrib.layers.xavier_initializer()

    net = tf.layers.dense(x, 2048, activation=tf.nn.relu, kernel_initializer=initializer)
    net = tf.layers.dropout(net, dropout_rate)
    net = tf.layers.dense(net, 2048, activation=tf.nn.relu, kernel_initializer=initializer)
    net = tf.layers.dropout(net, dropout_rate)
    net = tf.layers.dense(net, 2048, activation=tf.nn.relu, kernel_initializer=initializer)
    net = tf.layers.dropout(net, dropout_rate)
    net = tf.layers.dense(net, 2048, activation=tf.nn.relu, kernel_initializer=initializer)
    net = tf.layers.dropout(net, dropout_rate)
    net = tf.layers.dense(net, 1024, activation=tf.nn.relu, kernel_initializer=initializer)
    net = tf.layers.dropout(net, dropout_rate)
    net = tf.layers.dense(net, 1024, activation=tf.nn.relu, kernel_initializer=initializer)
    net = tf.layers.dropout(net, dropout_rate)
    net = tf.layers.dense(net, 512, activation=tf.nn.relu, kernel_initializer=initializer)
    net = tf.layers.dropout(net, dropout_rate)
    net = tf.layers.dense(net, 512, activation=tf.nn.relu, kernel_initializer=initializer)
    net = tf.layers.dropout(net, dropout_rate)
    net = tf.layers.dense(net, 1, activation=None)

    return net