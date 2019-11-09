import tensorflow as tf

from HBLSTM_CRF.config import hidden_size_lstm_1, hidden_size_lstm_2, tags, word_dim, proj1, proj2, words
from HBLSTM_CRF.local_utils import select


class DAModel():
    def __init__(self):
        with tf.variable_scope("placeholder"):
            self.dialogue_lengths = tf.placeholder(tf.int32, shape=[None], name="dialogue_lengths")
            self.word_ids = tf.placeholder(tf.int32, shape=[None, None, None], name="word_ids")
            self.utterance_lengths = tf.placeholder(tf.int32, shape=[None, None], name="utterance_lengths")
            self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
            self.clip = tf.placeholder(tf.float32, shape=[], name='clip')

        with tf.variable_scope("embeddings"):
            _word_embeddings = tf.get_variable(
                name="_word_embeddings",
                dtype=tf.float32,
                shape=[words, word_dim],
                initializer=tf.random_uniform_initializer()
            )
            word_embeddings = tf.nn.embedding_lookup(_word_embeddings, self.word_ids, name="word_embeddings")
            self.word_embeddings = tf.nn.dropout(word_embeddings, 0.8)

        with tf.variable_scope("utterance_encoder"):
            s = tf.shape(self.word_embeddings)
            batch_size = s[0] * s[1]

            time_step = s[-2]
            word_embeddings = tf.reshape(self.word_embeddings, [batch_size, time_step, word_dim])
            length = tf.reshape(self.utterance_lengths, [batch_size])

            fw = tf.nn.rnn_cell.LSTMCell(hidden_size_lstm_1, forget_bias=0.8, state_is_tuple=True)
            bw = tf.nn.rnn_cell.LSTMCell(hidden_size_lstm_1, forget_bias=0.8, state_is_tuple=True)

            output, _ = tf.nn.bidirectional_dynamic_rnn(fw, bw, word_embeddings, sequence_length=length,
                                                        dtype=tf.float32)
            output = tf.concat(output, axis=-1)  # [batch_size, time_step, dim]

            # Select the last valid time step output as the utterance embedding,
            # this method is more concise than TensorArray with while_loop
            output = select(output, self.utterance_lengths)  # [batch_size, dim]
            output = tf.reshape(output, s[0], s[1], 2 * hidden_size_lstm_1)
            output = tf.nn.dropout(output, 0.8)

            # output_ta = tf.TensorArray(dtype = tf.float32, size = 1, dynamic_size = True)

            # def body(time, output_ta_1):
            #     if length[time] == 0:
            #         output_ta_1 = output_ta_1.write(time, output[time][0])
            #     else:
            #         output_ta_1 = output_ta_1.write(time, output[time][length[time] - 1])
            #     return time + 1, output_ta_1

            # def condition(time, output_ta_1):
            #     return time < batch_size

            # i = 0
            # [time, output_ta] = tf.while_loop(condition, body, loop_vars = [i, output_ta])
            # output = output_ta.stack()
            # output = tf.reshape(output, [s[0], s[1], 2*hidden_size_lstm_1])
            # output = tf.nn.dropout(output, 0.8)

        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.BasicLSTMCell(hidden_size_lstm_2, state_is_tuple=True)
            cell_bw = tf.contrib.rnn.BasicLSTMCell(hidden_size_lstm_2, state_is_tuple=True)

            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, output,
                                                                        sequence_length=self.dialogue_lengths,
                                                                        dtype=tf.float32)
            outputs = tf.concat([output_fw, output_bw], axis=-1)
            outputs = tf.nn.dropout(outputs, 0.8)

        with tf.variable_scope("proj1"):
            output = tf.reshape(outputs, [-1, 2 * hidden_size_lstm_2])
            W = tf.get_variable("W", dtype=tf.float32, shape=[2 * hidden_size_lstm_2, proj1],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable("b", dtype=tf.float32, shape=[proj1], initializer=tf.zeros_initializer())
            output = tf.nn.relu(tf.matmul(output, W) + b)

        with tf.variable_scope("proj2"):
            W = tf.get_variable("W", dtype=tf.float32, shape=[proj1, proj2],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable("b", dtype=tf.float32, shape=[proj2], initializer=tf.zeros_initializer())
            output = tf.nn.relu(tf.matmul(output, W) + b)

        with tf.variable_scope("logits"):
            nstep = tf.shape(outputs)[1]
            W = tf.get_variable("W", dtype=tf.float32, shape=[proj2, tags], initializer=tf.random_uniform_initializer())
            b = tf.get_variable("b", dtype=tf.float32, shape=[tags], initializer=tf.zeros_initializer())

            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, nstep, tags])

        with tf.variable_scope("loss"):
            log_likelihood, self.trans_params = tf.contrib.crf.crf_log_likelihood(
                self.logits, self.labels, self.dialogue_lengths)
            self.loss = tf.reduce_mean(-log_likelihood) + tf.nn.l2_loss(W) + tf.nn.l2_loss(b)
            # tf.summary.scalar("loss", self.loss)

        with tf.variable_scope("viterbi_decode"):
            viterbi_sequence, _ = tf.contrib.crf.crf_decode(self.logits, self.trans_params, self.dialogue_lengths)

            batch_size = tf.shape(self.dialogue_lengths)[0]

            output_ta = tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=True)

            def body(time, output_ta_1):
                length = self.dialogue_lengths[time]
                vcode = viterbi_sequence[time][:length]
                true_labs = self.labels[time][:length]
                accurate = tf.reduce_sum(tf.cast(tf.equal(vcode, true_labs), tf.float32))

                output_ta_1 = output_ta_1.write(time, accurate)
                return time + 1, output_ta_1

            def condition(time, output_ta_1):
                return time < batch_size

            i = 0
            [time, output_ta] = tf.while_loop(condition, body, loop_vars=[i, output_ta])
            output_ta = output_ta.stack()
            accuracy = tf.reduce_sum(output_ta)
            self.accuracy = accuracy / tf.reduce_sum(tf.cast(self.dialogue_lengths, tf.float32))
            # tf.summary.scalar("accuracy", self.accuracy)

        with tf.variable_scope("train_op"):
            optimizer = tf.train.AdagradOptimizer(0.1)
            # if tf.greater(self.clip , 0):
            grads, vs = zip(*optimizer.compute_gradients(self.loss))
            grads, gnorm = tf.clip_by_global_norm(grads, self.clip)
            self.train_op = optimizer.apply_gradients(zip(grads, vs))
            # else:
            #    self.train_op = optimizer.minimize(self.loss)
        # self.merged = tf.summary.merge_all()