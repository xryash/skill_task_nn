import tensorflow as tf

from net import net
from utils import loss_plot, accuracy_plot
from preprocessing import train_prepr


def train(epochs, batch_size, learning_rate, model_replica_path, dropout_rate):


    # init dropout params
    train_dropout_rate = dropout_rate
    test_dropout_rate = 0.0

    # remove previous weights, bias, inputs, etc..
    tf.reset_default_graph()

    # init gpu device
    DEVICE = '/device:GPU:0'

    features_count = 1612
    labels_count = 1

    # load data
    x_train, y_train, x_val, y_val = train_prepr(200)

    x = tf.placeholder(tf.float32, shape=(None, features_count), name='x')
    y = tf.placeholder(tf.float32, shape=(None, labels_count), name='y')
    dropout_rate = tf.placeholder(tf.float32, name='dropout_rate')

    with tf.device(DEVICE):

        # init model
        model = net(x, dropout_rate=dropout_rate)

        # init optimization function
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=model)
        cost = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        # accuracy function
        predicted = tf.nn.sigmoid(model)
        correct_prediction = tf.equal(tf.round(predicted), y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # create tensorflow session
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            train_loss, test_loss, train_accuracy, test_accuracy = [], [], [], []

            print('Training....')
            for i in range(epochs):
                print('Epoch {} :'.format(i + 1))
                for batch in range(len(x_train) // batch_size):
                    # batching data
                    batch_x = x_train[batch * batch_size:min((batch + 1) * batch_size, len(x_train))]
                    batch_y = y_train[batch * batch_size:min((batch + 1) * batch_size, len(y_train))]

                    # start training
                    session.run(optimizer, feed_dict={x: batch_x,
                                                      y: batch_y,
                                                      dropout_rate: train_dropout_rate})

                    # compute metrics
                    train_loss_batch, train_accuracy_batch = session.run([cost, accuracy], feed_dict={x: batch_x,
                                                                                                      y: batch_y,
                                                                                                      dropout_rate: train_dropout_rate})

                    print(
                        'Batch range:{} - {}  Loss: {:>10.4f}  Accuracy: {:.6f}'.format(batch * batch_size,
                                                                                        min((batch + 1) * batch_size,
                                                                                            len(x_train)),
                                                                                        train_loss_batch,
                                                                                        train_accuracy_batch))

                test_accuracy_batch, test_loss_batch = session.run([accuracy, cost], feed_dict={x: x_val,
                                                                                                y: y_val,
                                                                                                dropout_rate: test_dropout_rate})

                print(
                    'Epoch {} finished, Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format((i + 1), test_loss_batch,
                                                                                            test_accuracy_batch))

                train_loss.append(train_loss_batch)
                test_loss.append(test_loss_batch)
                train_accuracy.append(train_accuracy_batch)
                test_accuracy.append(test_accuracy_batch)

            # draw plots
            loss_plot(train_loss, test_loss)
            accuracy_plot(train_accuracy, test_accuracy)


if __name__ == "__main__":
    epochs = 100
    batch_size = 8
    learning_rate = 0.003
    model_replica_path = 'tmp/model.ckpt'
    dropout_rate = 0.4
    train(epochs, batch_size, learning_rate, model_replica_path, dropout_rate)