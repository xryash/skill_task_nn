import tensorflow as tf


def eval_accuracy(y, predictions):
    y_true = tf.placeholder(tf.float32, shape=(None, 1), name='y_true')
    y_pred = tf.placeholder(tf.float32, shape=(None, 1), name='y_pred')

    correct_prediction = tf.equal(tf.round(y_pred), y_true)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

    session = tf.get_default_session()

    feed_dict_metrics = {y_true: y,
                         y_pred: predictions}

    accuracy = session.run(accuracy,
                           feed_dict=feed_dict_metrics)

    return accuracy


def eval_auc(y, predictions):
    y_true = tf.placeholder(tf.float32, shape=(None, 1), name='y_true')
    y_pred = tf.placeholder(tf.float32, shape=(None, 1), name='y_pred')

    # auc, update_op = tf.metrics.auc(y_true, y_pred, name='auc')
    auc = tf.metrics.auc(y_true, y_pred)

    session = tf.get_default_session()

    session.run(tf.local_variables_initializer())

    feed_dict_metrics = {y_true: y,
                         y_pred: predictions}

    auc = session.run([auc],
                                 feed_dict=feed_dict_metrics)

    return auc
