import tensorflow as tf

from net import ensemble_arch


class NeuralNet(object):
    def __init__(self,
                 dropout_rate, features_count, labels_count, epochs, batch_size, learning_rate
                 ):
        self.dropout_rate = dropout_rate
        self.labels_count = labels_count
        self.features_count = features_count
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # X and Y tensorflow inputs
        self.x_input = tf.placeholder(tf.float32, shape=(None, self.features_count), name='x')
        self.y_input = tf.placeholder(tf.float32, shape=(None, self.labels_count), name='y')
        # dropout tensorflow input
        self.d_r_input = tf.placeholder(tf.float32, name='d_r')

    def __cost(self, y, model):
        """Return cost function tensor"""
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=model)
        cost = tf.reduce_mean(cross_entropy, name='cost')
        return cost

    def __optimizer(self, cost):
        """Return NN optimizer"""
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost, name='optimizer')
        return optimizer

    def __accuracy(self, y, model):
        """Return accuracy function tensor"""
        predictions = tf.nn.sigmoid(model)
        correct_prediction = tf.equal(tf.round(predictions), y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
        return accuracy

    def __compute_train_metrics(self, batch_x, batch_y, batch_index, data_len, accuracy, cost):
        """Compute cost and accuracy functions for train batch"""
        session = tf.get_default_session()

        # compute train metrics
        feed_dict_metrics = {self.x_input: batch_x,
                             self.y_input: batch_y,
                             self.d_r_input: self.dropout_rate}

        train_loss_batch, train_accuracy_batch = session.run([cost, accuracy],
                                                             feed_dict=feed_dict_metrics)
        # print train metrics
        print(
            'Batch range:{} - {}  Loss: {:>10.4f}  Accuracy: {:.6f}'.format(batch_index * self.batch_size,
                                                                            min((
                                                                                        batch_index + 1) * self.batch_size,
                                                                                data_len),
                                                                            train_loss_batch,
                                                                            train_accuracy_batch))
        return train_loss_batch, train_accuracy_batch

    def __compute_test_metrics(self, x_val, y_val, epoch, accuracy, cost):
        """Compute cost and accuracy functions for test batch"""
        session = tf.get_default_session()
        # compute test metrics
        feed_dict_metrics = {self.x_input: x_val,
                             self.y_input: y_val,
                             self.d_r_input: 0.0}

        test_accuracy_batch, test_loss_batch = session.run([accuracy, cost],
                                                           feed_dict=feed_dict_metrics)

        # print test metrics
        print(
            'Epoch {} finished, Loss: {:>10.4f} Validation Accuracy: {:.6f} '.format((epoch + 1),
                                                                                     test_loss_batch,
                                                                                     test_accuracy_batch))
        return test_accuracy_batch, test_loss_batch

    def training(self, x_train, y_train, x_val, y_val):
        """Neural network training function"""

        train_loss, test_loss = [], [],
        train_accuracy, test_accuracy = [], []

        # init model
        model = ensemble_arch(self.x_input, dropout_rate=self.d_r_input)

        # accuracy function
        accuracy = self.__accuracy(self.y_input, model)

        # init cost function
        cost = self.__cost(self.y_input, model)

        # init optimizer
        optimizer = self.__optimizer(cost)

        session = tf.get_default_session()
        session.run(tf.global_variables_initializer())

        for epoch in range(self.epochs):
            print('Epoch {} :'.format(epoch + 1))
            for batch_index in range(len(x_train) // self.batch_size):
                # batching data
                batch_x = x_train[
                          batch_index * self.batch_size:min((batch_index + 1) * self.batch_size, len(x_train))]
                batch_y = y_train[
                          batch_index * self.batch_size:min((batch_index + 1) * self.batch_size, len(y_train))]

                # start training
                feed_dict_train = {self.x_input: batch_x,
                                   self.y_input: batch_y,
                                   self.d_r_input: self.dropout_rate}
                session.run(optimizer, feed_dict=feed_dict_train)

                # compute metrics with train data
                train_loss_batch, train_accuracy_batch = self.__compute_train_metrics(batch_x, batch_y,
                                                                                      batch_index,
                                                                                      len(x_train), accuracy,
                                                                                      cost)
            # compute metrics with test data
            test_accuracy_batch, test_loss_batch = self.__compute_test_metrics(x_val, y_val, epoch, accuracy,
                                                                               cost)

            train_loss.append(train_loss_batch)
            test_loss.append(test_loss_batch)
            train_accuracy.append(train_accuracy_batch)
            test_accuracy.append(test_accuracy_batch)

        return train_loss, test_loss, train_accuracy, test_accuracy

    def save_model(self, model_path):
        """Save model by path"""
        saver = tf.train.Saver()
        session = tf.get_default_session()
        saver.save(session, model_path)

    @staticmethod
    def local_val(x_test, y_test):
        """Local validation function tensorflow graph"""

        session = tf.get_default_session()
        graph = tf.get_default_graph()

        # get the inputs from the graph by name
        x_input = graph.get_tensor_by_name('x:0')
        y_input = graph.get_tensor_by_name('y:0')
        d_r_input = graph.get_tensor_by_name('d_r:0')

        # get network output layer
        predictions = tf.nn.sigmoid(graph.get_tensor_by_name('output/BiasAdd:0'))

        # get metric tensors
        accuracy = graph.get_tensor_by_name('accuracy:0')
        cost = graph.get_tensor_by_name('cost:0')

        # compute predictions and metrics
        feed_dict_metrics = {x_input: x_test,
                             y_input: y_test,
                             d_r_input: 0.0}
        predictions, test_accuracy, test_loss = session.run([predictions, accuracy, cost],
                                                            feed_dict=feed_dict_metrics)

        return predictions, test_loss, test_accuracy

    @staticmethod
    def predict(x_test):
        """Predict function tensorflow graph"""

        session = tf.get_default_session()
        graph = tf.get_default_graph()

        # get the inputs from the graph by name
        x_input = graph.get_tensor_by_name('x:0')
        d_r_input = graph.get_tensor_by_name('d_r:0')

        # get network output layer
        predictions = tf.nn.sigmoid(graph.get_tensor_by_name('output/BiasAdd:0'))

        # do predictions
        feed_dict_metrics = {x_input: x_test,
                             d_r_input: 0.0}
        predictions = session.run(predictions,
                                  feed_dict=feed_dict_metrics)

        return predictions
