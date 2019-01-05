import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import os
from data_utils import read_and_decode


class Model(object):
    def __init__(self, batch_size, num_epoch, num_step, dropout_rate,
                 target_vocab_size, seq_len, learning_rate, train_dataset, test_dataset,
                 is_training=True):
        self.batch_size = batch_size
        self.num_epochs = num_epoch
        self.num_steps = num_step
        self.dropout_rate = dropout_rate
        self.seq_len = seq_len
        self.learning_rate = learning_rate
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.hidden_dim = 256
        self.attention_hidden_dim = 100

        self.target_vocab_size = target_vocab_size
        self.global_step = tf.Variable(0, trainable=False)

        # attention
        self.attention_W = tf.Variable(tf.random_uniform([self.hidden_dim * 4, self.attention_hidden_dim],
                                                         0.0, 1.0), name='attention_W')
        self.attention_U = tf.Variable(tf.random_uniform([self.hidden_dim * 2, self.attention_hidden_dim],
                                                         0.0, 1.0), name='attention_U')
        self.attention_V = tf.Variable(tf.random_uniform([self.attention_hidden_dim, 1],
                                                         0.0, 1.0), name='attention_V')

        # softmax
        self.softmax_w = tf.Variable(tf.random_uniform([self.hidden_dim * 2, self.target_vocab_size],
                                                       0.0, 1.0), name='softmax_w', dtype=tf.float32)
        self.softmax_b = tf.Variable(tf.random_uniform([self.target_vocab_size],
                                                       0.0, 1.0), name='softmax_b', dtype=tf.float32)

        # LSTM cell
        self.encoder_lstm_cell_fw = rnn.BasicLSTMCell(self.hidden_dim, state_is_tuple=False)
        self.encoder_lstm_cell_bw = rnn.BasicLSTMCell(self.hidden_dim, state_is_tuple=False)
        self.decoder_lstm_cell = rnn.BasicLSTMCell(self.hidden_dim * 2, state_is_tuple=False)

        # dropout
        if is_training:
            self.decoder_lstm_cell = rnn.DropoutWrapper(self.decoder_lstm_cell,
                                                        output_keep_prob=self.dropout_rate)

    def CNN_VGG(self, inputs):
        ''' CNN extract feature from each input image, 网络架构选择的是VGG(CRNN)
        @param inputs: the input image
        @return: feature maps
        '''
        with tf.variable_scope('VGG_CNN'):
            conv1 = tf.layers.conv2d(inputs=inputs, filters=64, kernel_size=(3, 3),
                                     padding='SAME', activation=tf.nn.relu, name='conv_1')
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=(2, 2), strides=2, name='pool_1')

            conv2 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=(3, 3),
                                     padding='SAME', activation=tf.nn.relu, name='conv_2')
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=(2, 2), strides=2, name='pool_2')

            conv3 = tf.layers.conv2d(inputs=pool2, filters=256, kernel_size=(3, 3),
                                     padding='SAME', activation=tf.nn.relu, name='conv_3')

            conv4 = tf.layers.conv2d(inputs=conv3, filters=256, kernel_size=(3, 3),
                                     padding='SAME', activation=tf.nn.relu, name='conv_4')
            pool3 = tf.layers.max_pooling2d(inputs=conv4, pool_size=(1, 2), strides=2, name='pool_3')

            conv5 = tf.layers.conv2d(inputs=pool3, filters=512, kernel_size=(3, 3),
                                     padding='SAME', activation=tf.nn.relu, name='conv_5')
            bn1 = tf.layers.batch_normalization(conv5, name='bn1')

            conv6 = tf.layers.conv2d(inputs=bn1, filters=512, kernel_size=(3, 3),
                                     padding='SAME', activation=tf.nn.relu, name='conv_6')
            bn2 = tf.layers.batch_normalization(conv6, name='bn_2')
            pool4 = tf.layers.max_pooling2d(inputs=bn2, pool_size=(1, 2), strides=2, name='pool_4')

            conv7 = tf.layers.conv2d(inputs=pool4, filters=512, kernel_size=(3, 3),
                                     padding='SAME', activation=tf.nn.relu, name='conv_7')
        return conv7

    def bottleck(self, input, channel_out, stride, scope):
        channel_in = int.get_shape()[-1]
        channel = channel_out / 4
        with tf.variable_scope(scope):
            first_layer = tf.layers.conv2d(inputs=input, filters=channel, kernel_size=3,
                                           padding='VALID', activation=tf.nn.relu)
            second_layer = tf.layers.conv2d(inputs=first_layer, filters=channel, kernel_size=3,
                                            padding='VALID', activation=tf.nn.relu)
            if channel_in != channel_out:
                shortcut = tf.layers.conv2d(inputs=input, filters=channel_out, kernel_size=1,
                                            strides=stride, name='projects')
            else:
                shortcut = input
            output = tf.nn.relu(shortcut + second_layer)
            return output


    def residual_block(self, input, channel_out, stride, n_bottleneck, scope):
        with tf.variable_scope(scope):
            for i in range(1, n_bottleneck + 1):
                out = self.bottleck(input, channel_out, stride, scope='bottlenec_%i', i)
            return out

    def CNN_ResNet(self, inputs):
        '''CNN extract feature from each input image, 网络架构选择的是ResNet(FAN)
        @param inputs:
        @return:
        '''
        with tf.variable_scope('ResNet_CNN'):
            # conv1_x
            conv1_1 = tf.layers.conv2d(inputs=inputs, filters=32, kernel_size=3,
                                       strides=1, padding='SAME', activation=tf.nn.relu,
                                       name='conv1_1')
            conv1_2 = tf.layers.conv2d(inputs=conv1_1, filters=64, kernel_size=3,
                                       strides=1, padding='SAME', activation=tf.nn.relu,
                                       name='conv1_2')
            # conv2_x
            pool2_1 = tf.layers.max_pooling2d(inputs=conv1_2, pool_size=(2, 2), strides=2)
            conv2_2 = self.residual_block(input=pool2_1, channel_out=128, stride=1,
                                          n_bottleneck=1, scope='conv2_2')
            conv2_3 = tf.layers.conv2d(inputs=conv2_2, filters=128, kernel_size=3,
                                       strides=1, padding='SAME', activation=tf.nn.relu,
                                       name='conv2_3')
            # conv3_x
            pool3_1 = tf.layers.max_pooling2d(inputs=conv2_3, pool_size=(2, 2), strides=2)
            conv3_2 = self.residual_block(input=pool3_1, channel_out=256, stride=1,
                                          n_bottleneck=2, scope='copnv3_2')
            conv3_3 = tf.layers.conv2d(inputs=conv3_2, filters=256, kernel_size=3,
                                       strides=1, padding='SAME', activation=tf.nn.relu,
                                       name='conv3_3')
            # conv4_x
            pool4_1 = tf.layers.max_pooling2d(inputs=conv3_3, pool_size=(2, 2), strides=(1, 2))
            conv4_2 = self.residual_block(input=pool4_1, channel_out=512, stride=1,
                                          n_bottleneck=5, scope='conv4_2')
            conv4_3 = tf.layers.conv2d(inputs=conv4_2, filters=512, kernel_size=3,
                                       strides=1, padding='SAME', activation=tf.nn.relu,
                                       name='conv4_3')
            # conv5_x
            conv5_1 = self.residual_block(input=conv4_3, channel_out=512, stride=1,
                                          n_bottleneck=3, scope='conv5_1')
            conv5_2 = tf.layers.conv2d(inputs=conv5_1, filters=512, kernel_size=2,
                                       strides=(1, 2), padding='SMAE', activation=tf.nn.relu,
                                       name='conv5_2')
            conv5_3 = tf.layers.conv2d(inputs=conv5_2, filters=512, kernel_size=2,
                                       strides=1, padding='valid', activation=tf.nn.relu,
                                       name='conv5_3')
        return conv5_3

    def mapToSequence(self, cnnOutput):
        '''
        convy feature maps into sequential representations
        @param cnnOutput: the feature map of CNN
        @return: the feature sequence
        '''
        reshaped_CNN_output = tf.reshape(cnnOutput, [self.batch_size, -1, 512])
        return reshaped_CNN_output

    def encoder(self, cell_fw, cell_bw, inputs, seq_len):
        '''
        @param cell_fw: the forward cell
        @param cell_bw: the backward cell
        @param inputs: the input data
        @param seq_len: the length of input data
        @return:
        '''
        enc_outputs, (output_state_fw, output_state_bw) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw, cell_bw=cell_bw, inputs=inputs, sequence_length=seq_len,
            dtype=tf.float32
        )
        enc_state = tf.concat([output_state_fw, output_state_bw], axis=1)
        enc_outputs = tf.concat(enc_outputs, axis=2)
        return enc_outputs, enc_state

    def attention(self, prev_state, enc_outputs):
        '''
        attention model
        @param prev_state: the decoder hidden state at time i-1
        @param enc_outputs: the encode outputs, a length 'T' list
        '''
        e_i = []
        c_i = []
        for output in enc_outputs:
            atten_hidden = tf.tanh(tf.add(tf.matmul(prev_state, self.attention_W),
                                          tf.matmul(output, self.attention_U)))
            e_i_j = tf.matmul(atten_hidden, self.attention_V)
            e_i.append(e_i_j)
        e_i = tf.concat(e_i, axis=1)
        alpha_i = tf.nn.softmax(e_i)
        alpha_i = tf.split(alpha_i, self.num_steps, 1)

        for alpha_i_j, output in zip(alpha_i, enc_outputs):
            c_i_j = tf.multiply(alpha_i_j, output)
            c_i.append(c_i_j)
        c_i = tf.reshape(tf.concat(c_i, axis=1), [-1, self.num_steps, self.hidden_dim * 2])
        c_i = tf.reduce_mean(c_i, 1)
        return c_i

    def Focusing(self, alpha, c_i):
        '''
        @param alpha:
        @param c_i:
        @return:
        '''
    def decode(self, cell, init_state, enc_outputs, loop_function=None):
        '''
        @param cell:
        @param init_state:
        @param enc_outputs:
        @param loop_function:
        @return:
        '''
        outputs = []
        prev = None
        state = init_state
        for i, inp in enumerate():
            if loop_function is not None and prev is not None:
                with tf.variable_scope('loop_function', reuse=True):
                    inp = loop_function(prev, i)
            if i > 0:
                tf.get_variable_scope().reuse_variables()

            c_i = self.attention(state, enc_outputs)
            inp = tf.concat([inp, c_i], 1)
            output, state = cell(inp, state)
            outputs.append(output)
            if loop_function is not None:
                prev = output
        return outputs

    def loop_function(self, prev, _):
        '''
        @param prev: the output of t-1 time
        @return:
        '''
        prev = tf.add(tf.matmul(prev, self.softmax_w) + self.softmax_b)
        prev_symbol = tf.arg_max(prev, 1)
        return prev_symbol

    def train(self):
        '''
        train the total architecture
        '''
        train_image, train_label = read_and_decode(self.train_dataset, self.batch_size)
        #
        CNN_logits = self.CNN_VGG(train_image)
        # CNN_logits = self.CNN_ResNet(train_image)
        print('After CNN:', CNN_logits.shape)
        map_to_sequence = self.mapToSequence(CNN_logits)
        print('After map_to_sequence', map_to_sequence.shape)
        enc_output, enc_state = self.encoder(self.encoder_lstm_cell_fw,
                                             self.encoder_lstm_cell_bw,
                                             map_to_sequence,
                                             seq_len)
        dec_outputs = self.decode(self.decoder_lstm_cell, enc_state, enc_output)
        outputs= tf.reshape(tf.concat(dec_outputs, axis=1), [-1, self.hidden_dim * 2])
        logits = tf.add(tf.matmul(outputs, self.softmax_w), self.softmax_b)
        prediction = tf.nn.softmax(logits)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=train_label)
        self.cross_entropy_loss = tf.reduce_mean(cross_entropy)
        self.optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)\
            .minimize(self.cross_entropy_loss)

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=session, coord=coord)
            for index in range(self.num_epochs):
                _, loss = session.run([self.optimizer, self.cross_entropy_loss])

            coord.request_stop()
            coord.join(threads=threads)

    def test(self):
        test_image, test_label = read_and_decode(self.test_dataset, self.batch_size)