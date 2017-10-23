import tensorflow as tf


class CNN:
    def __init__(self, x1_maxlen, x2_maxlen, y_len, embedding, filter_sizes, num_filters, hidden_size, state_size, x3_size):
        self.input_x1 = tf.placeholder(tf.int32, [None, x1_maxlen], name="post_text")
        self.input_x1_len = tf.placeholder(tf.int32, [None, ], name="post_text_len")
        self.input_x2 = tf.placeholder(tf.int32, [None, x2_maxlen], name="target_description")
        self.input_x2_len = tf.placeholder(tf.int32, [None, ], name="target_description_len")
        self.input_x3 = tf.placeholder(tf.float32, [None, x3_size], name="image_feature")
        self.input_y = tf.placeholder(tf.float32, [None, y_len], name="truth_class")
        self.input_z = tf.placeholder(tf.float32, [None, 1], name="truth_mean")
        self.dropout_rate_embedding = tf.placeholder(tf.float32, name="dropout_rate_embedding")
        self.dropout_rate_hidden = tf.placeholder(tf.float32, name="dropout_rate_hidden")
        self.dropout_rate_cell = tf.placeholder(tf.float32, name="dropout_rate_cell")
        self.batch_size = tf.placeholder(tf.int32, name="batch_size")

        self.W = tf.get_variable(shape=embedding.shape, initializer=tf.constant_initializer(embedding), name="embedding")
        self.embedded_input_x1 = tf.nn.embedding_lookup(self.W, self.input_x1)
        self.embedded_input_x1 = tf.layers.dropout(self.embedded_input_x1, rate=1-self.dropout_rate_embedding)
        self.embedded_input_x1_expanded = tf.expand_dims(self.embedded_input_x1, -1)

        pooled_outputs1 = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("1-conv-maxpool-%s" % filter_size):
                filter_shape = [filter_size, embedding.shape[1], 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="filter_weights")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="filter_biases")
                conv = tf.nn.conv2d(self.embedded_input_x1_expanded, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                pooled = tf.nn.max_pool(h, ksize=[1, x1_maxlen-filter_size+1, 1, 1], strides=[1, 1, 1, 1], padding="VALID", name="pool")
                pooled_outputs1.append(pooled)

        num_features = num_filters*len(filter_sizes)
        self.h_pool1 = tf.concat(pooled_outputs1, 3)
        self.h_pool_flat1 = tf.reshape(self.h_pool1, [-1, num_features])

        if x3_size:
            self.compressed_input_x3 = tf.layers.dense(tf.layers.dense(self.input_x3, 1024, activation=tf.nn.relu), 256, activation=tf.nn.relu)
            self.h_pool_flat1 = tf.concat([self.h_pool_flat1, self.compressed_input_x3], axis=-1)

        if hidden_size:
            self.h_pool_flat1 = tf.layers.dense(self.h_pool_flat1, hidden_size, activation=tf.nn.relu)

        self.h_drop1 = tf.layers.dropout(self.h_pool_flat1, rate=1-self.dropout_rate_hidden)

        self.scores = tf.layers.dense(inputs=self.h_drop1, units=y_len)

        if y_len == 1:
            self.predictions = tf.nn.sigmoid(self.scores, name="prediction")
            self.loss = tf.losses.mean_squared_error(self.input_z, self.predictions)
            self.mse = tf.losses.mean_squared_error(self.input_z, self.predictions)
            correct_predictions = tf.equal(tf.cast(tf.round(self.predictions), tf.int32), tf.cast(tf.round(self.input_y), tf.int32))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        elif y_len == 2:
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y))
            self.predictions = tf.slice(tf.nn.softmax(self.scores), [0, 0], [-1, 1], name="prediction")
            self.mse = tf.losses.mean_squared_error(self.input_z, self.predictions)
            correct_predictions = tf.equal(tf.argmax(self.scores, 1), tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        elif y_len == 4:
            self.normalised_scores = tf.nn.softmax(self.scores, name="distribution")
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.scores))
            self.predictions = tf.matmul(self.normalised_scores, tf.constant([0, 0.3333333333, 0.6666666666, 1.0], shape=[4, 1]), name="prediction")
            self.mse = tf.losses.mean_squared_error(self.input_z, self.predictions)
            correct_predictions = tf.equal(tf.argmax(tf.matmul(self.normalised_scores, tf.constant([1, 0, 1, 0, 0, 1, 0, 1], shape=[4, 2], dtype=tf.float32)), 1), tf.argmax(tf.matmul(self.input_y, tf.constant([1, 0, 1, 0, 0, 1, 0, 1], shape=[4, 2], dtype=tf.float32)), 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


class DAN:
    def __init__(self, x1_maxlen, x2_maxlen, y_len, embedding, filter_sizes, num_filters, hidden_size, state_size, x3_size):
        self.input_x1 = tf.placeholder(tf.int32, [None, x1_maxlen], name="post_text")
        self.input_x1_len = tf.placeholder(tf.int32, [None, ], name="post_text_len")
        self.input_x2 = tf.placeholder(tf.int32, [None, x2_maxlen], name="target_description")
        self.input_x2_len = tf.placeholder(tf.int32, [None, ], name="target_description_len")
        self.input_x3 = tf.placeholder(tf.float32, [None, x3_size], name="image_feature")
        self.input_y = tf.placeholder(tf.float32, [None, y_len], name="truth_class")
        self.input_z = tf.placeholder(tf.float32, [None, 1], name="truth_mean")
        self.dropout_rate_embedding = tf.placeholder(tf.float32, name="dropout_rate_embedding")
        self.dropout_rate_hidden = tf.placeholder(tf.float32, name="dropout_rate_hidden")
        self.dropout_rate_cell = tf.placeholder(tf.float32, name="dropout_rate_cell")
        self.batch_size = tf.placeholder(tf.int32, name="batch_size")

        self.W = tf.get_variable(shape=embedding.shape, initializer=tf.constant_initializer(embedding), name="embedding")
        self.embedded_input_x1 = tf.nn.embedding_lookup(self.W, self.input_x1)
        self.embedded_input_x1 = tf.layers.dropout(self.embedded_input_x1, rate=1-self.dropout_rate_embedding)

        # self.avg_input_x1 = tf.reduce_mean(self.embedded_input_x1, axis=1)
        mask = tf.cast(tf.contrib.keras.backend.repeat_elements(tf.expand_dims(tf.sequence_mask(self.input_x1_len, x1_maxlen), axis=-1), embedding.shape[1], axis=2), tf.float32)
        masked_embedded_input_x1 = tf.multiply(self.embedded_input_x1, mask)
        self.avg_input_x1 = tf.reduce_sum(masked_embedded_input_x1, axis=1)/tf.reduce_sum(mask, axis=1)

        if hidden_size:
            self.avg_input_x1 = tf.layers.dense(self.avg_input_x1, hidden_size, activation=tf.nn.relu)
        self.h_drop1 = tf.layers.dropout(self.avg_input_x1, rate=1-self.dropout_rate_hidden)
        self.scores = tf.layers.dense(inputs=self.h_drop1, units=y_len)

        if y_len == 1:
            self.predictions = tf.nn.sigmoid(self.scores, name="prediction")
            self.loss = tf.losses.mean_squared_error(self.input_z, self.predictions)
            self.mse = tf.losses.mean_squared_error(self.input_z, self.predictions)
            correct_predictions = tf.equal(tf.cast(tf.round(self.predictions), tf.int32), tf.cast(tf.round(self.input_y), tf.int32))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        elif y_len == 2:
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y))
            self.predictions = tf.slice(tf.nn.softmax(self.scores), [0, 0], [-1, 1], name="prediction")
            self.mse = tf.losses.mean_squared_error(self.input_z, self.predictions)
            correct_predictions = tf.equal(tf.argmax(self.scores, 1), tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        elif y_len == 4:
            self.normalised_scores = tf.nn.softmax(self.scores, name="distribution")
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.scores))
            self.predictions = tf.matmul(self.normalised_scores, tf.constant([0, 0.3333333333, 0.6666666666, 1.0], shape=[4, 1]), name="prediction")
            self.mse = tf.losses.mean_squared_error(self.input_z, self.predictions)
            correct_predictions = tf.equal(tf.argmax(tf.matmul(self.normalised_scores, tf.constant([1, 0, 1, 0, 0, 1, 0, 1], shape=[4, 2], dtype=tf.float32)), 1), tf.argmax(tf.matmul(self.input_y, tf.constant([1, 0, 1, 0, 0, 1, 0, 1], shape=[4, 2], dtype=tf.float32)), 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


def extract_last(output, lengths):
    batch_range = tf.range(tf.shape(output)[0])
    batch_idx = tf.stack([batch_range, lengths-1], axis=-1)
    return tf.gather_nd(output, batch_idx)


class BiRNN:
    def __init__(self, x1_maxlen, x2_maxlen, y_len, embedding, filter_sizes, num_filters, hidden_size, state_size, x3_size):
        self.input_x1 = tf.placeholder(tf.int32, [None, x1_maxlen], name="post_text")
        self.input_x1_len = tf.placeholder(tf.int32, [None, ], name="post_text_len")
        self.input_x2 = tf.placeholder(tf.int32, [None, x2_maxlen], name="target_description")
        self.input_x2_len = tf.placeholder(tf.int32, [None, ], name="target_description_len")
        self.input_x3 = tf.placeholder(tf.float32, [None, x3_size], name="image_feature")
        self.input_y = tf.placeholder(tf.float32, [None, y_len], name="truth_class")
        self.input_z = tf.placeholder(tf.float32, [None, 1], name="truth_mean")
        self.dropout_rate_embedding = tf.placeholder(tf.float32, name="dropout_rate_embedding")
        self.dropout_rate_hidden = tf.placeholder(tf.float32, name="dropout_rate_hidden")
        self.dropout_rate_cell = tf.placeholder(tf.float32, name="dropout_rate_cell")
        self.batch_size = tf.placeholder(tf.int32, [], name="batch_size")

        self.W = tf.get_variable(shape=embedding.shape, initializer=tf.constant_initializer(embedding), name="embedding")
        self.embedded_input_x1 = tf.nn.embedding_lookup(self.W, self.input_x1)
        self.embedded_input_x1 = tf.layers.dropout(self.embedded_input_x1, rate=1-self.dropout_rate_embedding)

        cell_fw = tf.contrib.rnn.GRUCell(state_size)
        cell_dropout_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=1-self.dropout_rate_cell)
        initial_state_fw = cell_fw.zero_state(self.batch_size, tf.float32)
        cell_bw = tf.contrib.rnn.GRUCell(state_size)
        cell_dropout_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=1-self.dropout_rate_cell)
        initial_state_bw = cell_bw.zero_state(self.batch_size, tf.float32)
        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_dropout_fw, cell_bw=cell_dropout_bw, inputs=self.embedded_input_x1, sequence_length=self.input_x1_len, initial_state_bw=initial_state_bw, initial_state_fw=initial_state_fw)
        bi_outputs = tf.concat(outputs, 2)
        mask = tf.cast(tf.contrib.keras.backend.repeat_elements(tf.expand_dims(tf.sequence_mask(self.input_x1_len, x1_maxlen), axis=-1), 2*state_size, axis=2), tf.float32)

        self.h_drop = tf.layers.dropout(tf.concat([extract_last(outputs[0], self.input_x1_len), outputs[1][:, 0, :]], -1), rate=1-self.dropout_rate_hidden)

        # self.h_drop = tf.layers.dropout(tf.reduce_sum(bi_outputs, axis=1)/tf.reduce_sum(mask, axis=1), rate=1-self.dropout_rate_hidden)
        #
        # self.h_drop = tf.layers.dropout(tf.reduce_max(bi_outputs, axis=1), rate=1-self.dropout_rate_hidden)

        self.scores = tf.layers.dense(inputs=self.h_drop, units=y_len)

        if y_len == 1:
            self.predictions = tf.nn.sigmoid(self.scores, name="prediction")
            self.loss = tf.losses.mean_squared_error(self.input_z, self.predictions)
            self.mse = tf.losses.mean_squared_error(self.input_z, self.predictions)
            correct_predictions = tf.equal(tf.cast(tf.round(self.predictions), tf.int32), tf.cast(tf.round(self.input_y), tf.int32))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        elif y_len == 2:
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y))
            self.predictions = tf.slice(tf.nn.softmax(self.scores), [0, 0], [-1, 1], name="prediction")
            self.mse = tf.losses.mean_squared_error(self.input_z, self.predictions)
            correct_predictions = tf.equal(tf.argmax(self.scores, 1), tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        elif y_len == 4:
            self.normalised_scores = tf.nn.softmax(self.scores, name="distribution")
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.scores))
            self.predictions = tf.matmul(self.normalised_scores, tf.constant([0, 0.3333333333, 0.6666666666, 1.0], shape=[4, 1]), name="prediction")
            self.mse = tf.losses.mean_squared_error(self.input_z, self.predictions)
            correct_predictions = tf.equal(tf.argmax(tf.matmul(self.normalised_scores, tf.constant([1, 0, 1, 0, 0, 1, 0, 1], shape=[4, 2], dtype=tf.float32)), 1), tf.argmax(tf.matmul(self.input_y, tf.constant([1, 0, 1, 0, 0, 1, 0, 1], shape=[4, 2], dtype=tf.float32)), 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


class SAN:
    def __init__(self, x1_maxlen, x2_maxlen, y_len, embedding, filter_sizes, num_filters, hidden_size, state_size, x3_size, attention_size, view_size=1, alpha=0, beta=0):
        if view_size == 1:
            beta = 0
        self.input_x1 = tf.placeholder(tf.int32, [None, x1_maxlen], name="post_text")
        self.input_x1_len = tf.placeholder(tf.int32, [None, ], name="post_text_len")
        self.input_x2 = tf.placeholder(tf.int32, [None, x2_maxlen], name="target_description")
        self.input_x2_len = tf.placeholder(tf.int32, [None, ], name="target_description_len")
        self.input_x3 = tf.placeholder(tf.float32, [None, x3_size], name="image_feature")
        self.input_y = tf.placeholder(tf.float32, [None, y_len], name="truth_class")
        self.input_z = tf.placeholder(tf.float32, [None, 1], name="truth_mean")
        self.dropout_rate_embedding = tf.placeholder(tf.float32, name="dropout_rate_embedding")
        self.dropout_rate_hidden = tf.placeholder(tf.float32, name="dropout_rate_hidden")
        self.dropout_rate_cell = tf.placeholder(tf.float32, name="dropout_rate_cell")
        self.batch_size = tf.placeholder(tf.int32, [], name="batch_size")
        with tf.variable_scope("embedding"):
            self.W = tf.get_variable(shape=embedding.shape, initializer=tf.constant_initializer(embedding), name="embedding")
            self.embedded_input_x1 = tf.nn.embedding_lookup(self.W, self.input_x1)
            self.embedded_input_x1 = tf.layers.dropout(self.embedded_input_x1, rate=1-self.dropout_rate_embedding)
        with tf.variable_scope("biRNN"):
            cell_fw = tf.contrib.rnn.GRUCell(state_size)
            cell_dropout_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=1-self.dropout_rate_cell)
            initial_state_fw = cell_fw.zero_state(self.batch_size, tf.float32)
            cell_bw = tf.contrib.rnn.GRUCell(state_size)
            cell_dropout_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=1-self.dropout_rate_cell)
            initial_state_bw = cell_bw.zero_state(self.batch_size, tf.float32)
            outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_dropout_fw, cell_bw=cell_dropout_bw, inputs=self.embedded_input_x1, sequence_length=self.input_x1_len, initial_state_bw=initial_state_bw, initial_state_fw=initial_state_fw)
            bi_outputs = tf.concat(outputs, 2)
        with tf.variable_scope("attention"):
            W_1 = tf.get_variable(shape=[2*state_size, attention_size], initializer=tf.contrib.layers.xavier_initializer(), name="W_1")
            W_2 = tf.get_variable(shape=[attention_size, view_size], initializer=tf.contrib.layers.xavier_initializer(), name="W_2")
            reshaped_bi_outputs = tf.reshape(bi_outputs, shape=[-1, 2*state_size])
            if x3_size:
                # self.compressed_input_x3 = tf.contrib.keras.backend.repeat(tf.layers.dense(tf.layers.dense(self.input_x3, 1024, activation=tf.nn.tanh), attention_size, activation=tf.nn.tanh), x1_maxlen)
                self.compressed_input_x3 = tf.contrib.keras.backend.repeat(tf.layers.dense(self.input_x3, attention_size, activation=tf.nn.tanh), x1_maxlen)
                self.compressed_input_x3 = tf.reshape(self.compressed_input_x3, shape=[-1, attention_size])
                self.attention = tf.nn.softmax(tf.reshape(tf.matmul(tf.nn.tanh(tf.matmul(reshaped_bi_outputs, W_1)+self.compressed_input_x3), W_2), shape=[self.batch_size, x1_maxlen, view_size]), dim=1)
            else:
                self.attention = tf.nn.softmax(tf.reshape(tf.matmul(tf.nn.tanh(tf.matmul(reshaped_bi_outputs, W_1)), W_2), shape=[self.batch_size, x1_maxlen, view_size]), dim=1)
            attention_output = tf.reshape(tf.matmul(tf.transpose(bi_outputs, perm=[0, 2, 1]), self.attention), shape=[self.batch_size, view_size*2*state_size])
        with tf.variable_scope("penalty"):
            attention_t = tf.transpose(self.attention, perm=[0, 2, 1])
            attention_t_attention = tf.matmul(attention_t, self.attention)
            identity = tf.reshape(tf.tile(tf.diag(tf.ones([view_size])), [self.batch_size, 1]), shape=[self.batch_size, view_size, view_size])
            self.penalised_term = tf.square(tf.norm(attention_t_attention-identity, ord="euclidean", axis=[1, 2]))
        self.h_drop = tf.layers.dropout(attention_output, rate=1-self.dropout_rate_hidden)
        self.scores = tf.layers.dense(inputs=self.h_drop, units=y_len)
        if y_len == 1:
            self.predictions = tf.nn.sigmoid(self.scores, name="prediction")
            self.loss = tf.reduce_mean(tf.square(tf.subtract(self.input_z, self.predictions))+beta*self.penalised_term)
            self.mse = tf.losses.mean_squared_error(self.input_z, self.predictions)
            correct_predictions = tf.equal(tf.cast(tf.round(self.predictions), tf.int32), tf.cast(tf.round(self.input_y), tf.int32))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        elif y_len == 2:
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)+beta*self.penalised_term)
            self.predictions = tf.slice(tf.nn.softmax(self.scores), [0, 0], [-1, 1], name="prediction")
            self.mse = tf.losses.mean_squared_error(self.input_z, self.predictions)
            correct_predictions = tf.equal(tf.argmax(self.scores, 1), tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        elif y_len == 4:
            self.normalised_scores = tf.nn.softmax(self.scores, name="distribution")
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.scores)+beta*self.penalised_term)
            self.predictions = tf.matmul(self.normalised_scores, tf.constant([0, 0.3333333333, 0.6666666666, 1.0], shape=[4, 1]), name="prediction")
            self.mse = tf.losses.mean_squared_error(self.input_z, self.predictions)
            correct_predictions = tf.equal(tf.argmax(tf.matmul(self.normalised_scores, tf.constant([1, 0, 1, 0, 0, 1, 0, 1], shape=[4, 2], dtype=tf.float32)), 1), tf.argmax(tf.matmul(self.input_y, tf.constant([1, 0, 1, 0, 0, 1, 0, 1], shape=[4, 2], dtype=tf.float32)), 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
