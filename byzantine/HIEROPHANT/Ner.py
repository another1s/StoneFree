import tensorflow as tf
import numpy as np
from byzantine.helperfunction.data_split import BatchGenerator


class BilstmNer:
    def __init__(self, config, embedding_pretrained, dropout_keep=1):
        # the keys of config:
        # {
        #     lr: learning rate (float)
        #     batch_size : number of instances being processed per iteration (int)
        #     embedding_dim: embedding dimension ( vector dimensions ) (int)
        #     embedding_size: number of words in dictionary   (int)
        #     sen len: sentence length     (int)
        #     tag_size: how many different name entity  (int)
        # }
        self.lr = config['learning_rate']
        self.batch_size = config['batch_size']
        self.embedding_dim = config['embedding_dimension']
        self.embedding_size = config['embedding_size']
        self.sentence_len = config['sentence_length']
        self.pretrained_model = config['pretrained_model']
        self.tag_size = config['tag_size']
        self.config = config

        self.embedding_pretrained = embedding_pretrained
        self.dropout_keep = dropout_keep
        self.input_data = tf.placeholder(tf.int32, shape=[self.batch_size, self.sentence_len], name="input_data")
        self.labels = tf.placeholder(tf.int32, shape=[self.batch_size, self.sentence_len], name="labels")
        self.embedding_placeholder = tf.placeholder(tf.float32, shape=[self.embedding_size, self.embedding_dim],
                                                    name="embedding_placeholder")
        with tf.variable_scope("bilstm_crf") as scope:
            self.build_graph()

    def build_graph(self):
        word_embeddings = tf.get_variable("word_embeddings", [self.embedding_size, self.embedding_dim])
        if self.embedding_pretrained:
            embeddings_init = word_embeddings.assign(self.embedding_pretrained)

        input_embedded = tf.nn.embedding_lookup(word_embeddings, self.input_data)
        input_embedded = tf.nn.dropout(input_embedded, self.dropout_keep)

        lstm_fw_cell = tf.keras.layers.LSTMCell(self.embedding_dim, unit_forget_bias=True, dropout=0.3)
        lstm_bw_cell = tf.keras.layers.LSTMCell(self.embedding_dim, unit_forget_bias=True, dropout=0.3)
        bilstm_out = tf.keras.layers.Bidirectional(lstm_fw_cell, merge_mode='concat', backward_layer=lstm_bw_cell,
                                                   input_shape=input_embedded)
        # Fully connected layer.
        self.Weights = tf.get_variable(name="W", shape=[self.batch_size, 2 * self.embedding_dim, self.tag_size], dtype=tf.float32)

        self.bias = tf.get_variable(name="b", shape=[self.batch_size, self.sentence_len, self.tag_size], dtype=tf.float32, initializer=tf.zeros_initializer())

        bilstm_out = tf.tanh(tf.matmul(bilstm_out, self.Weights) + self.bias)

        # Linear-CRF.
        log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(bilstm_out, self.labels, tf.tile(np.array([self.sentence_len]), np.array([self.batch_size])))
        loss = tf.reduce_mean(-log_likelihood)

        # Compute the viterbi sequence and score (used for prediction and test time).
        self.viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(bilstm_out, self.transition_params, tf.tile(np.array([self.sentence_len]), np.array([self.batch_size])))

        # Training ops.
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.minimize(loss)

    def train(self, sess, training_data: 'list', training_labels: 'list', epoches: 'int', batch_size: 'int'):
        data_batches = BatchGenerator(training_data, training_labels)
        batch_num = data_batches.y.shape[0] / batch_size
        acc_list = []
        for epoch in range(epoches):
            acc = 0
            instance_num = 0
            for batch in range(batch_num):
                x_batch, y_batch = data_batches.next_batch(batch_size)
                feed_dict = {'input_data': x_batch, 'labels': y_batch}
                pre, train_op = sess.run([self.viterbi_sequence, self.train_op], feed_dict=feed_dict)
                # print('prediction: ', pre, '\n')
                # print('loss ', train_op, '\n')
                for i in range(len(y_batch)):
                    for j in range(len(y_batch[0])):
                        instance_num = instance_num + 1
                        if pre[i][j] == y_batch[i][j]:
                            acc = acc + 1
            # print training result of every epoch
            print(acc/instance_num)
            acc_list.append(acc/instance_num)

        average_acc = 0
        for acc in acc_list:
            average_acc = average_acc + acc
        return acc_list, average_acc/len(acc_list)

    def testing(self, sess, testing_data: 'list', testing_labels: 'list', epoches: 'int', batch_size: 'int'):
        data_batches = BatchGenerator(testing_data, testing_labels)
        batch_num = data_batches.y.shape[0] / batch_size
        acc_list = []
        recognized_res = []
        for epoch in range(epoches):
            acc = 0
            instance_num = 0
            for batch in range(batch_num):
                x_batch, y_batch = data_batches.next_batch(batch_size)
                feed_dict = {'input_data': x_batch, 'labels': y_batch}
                pre, _ = sess.run([self.viterbi_sequence], feed_dict=feed_dict)
                for i in range(len(y_batch)):
                    for j in range(len(y_batch[0])):
                        instance_num = instance_num + 1
                        if pre[i][j] == y_batch[i][j]:
                            acc = acc + 1
                            recognized_res.append((x_batch[i][j], y_batch[i][j]))
            acc_list.append(acc/instance_num)

        average_acc = 0
        for acc in acc_list:
            average_acc = average_acc + acc
        return acc_list, recognized_res, average_acc / len(acc_list)

    def fine_tuning(self):

        return

    def save_to_local(self):

        return