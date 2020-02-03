import tensorflow as tf


class VGGModel():

    def conv1(self, input_data):
        # layer 1 (convolutional layer)
        with tf.name_scope('conv_1'):
            w_conv1 = tf.Variable(tf.truncated_normal([3, 3, 1, 64], stddev=1e-2))
            b1 = tf.Variable(tf.truncated_normal([64], stddev=1e-2))
            h_conv1 = tf.nn.conv2d(input_data, w_conv1, strides=[1, 1, 1, 1], padding='SAME')
            h_conv1_relu = tf.nn.relu(tf.add(h_conv1, b1))
            h_conv1_maxpool = tf.nn.max_pool(h_conv1_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        return h_conv1_maxpool

    # convolutional network layer 2
    def conv2(self, input_data):
        with tf.name_scope('conv_2'):
            W_conv2 = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=1e-2))
            b2 = tf.Variable(tf.truncated_normal([128], stddev=1e-2))
            h_conv2 = tf.nn.conv2d(input_data, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
            h_conv2_relu = tf.nn.relu(tf.add(h_conv2, b2))
            h_conv2_maxpool = tf.nn.max_pool(h_conv2_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        return h_conv2_maxpool

    # convolutional network layer 3
    def conv3(self, input_data):
        with tf.name_scope('conv_3'):
            W_conv3_1 = tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev=1e-2))
            W_conv3_2 = tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev=1e-2))

            b3_1 = tf.Variable(tf.truncated_normal([256], stddev=1e-2))
            b3_2 = tf.Variable(tf.truncated_normal([256], stddev=1e-2))

            h_conv3_1 = tf.nn.conv2d(input_data, W_conv3_1, strides=[1, 1, 1, 1], padding='SAME')
            h_conv3_relu_1 = tf.nn.relu(tf.add(h_conv3_1, b3_1))

            h_conv3_2 = tf.nn.conv2d(h_conv3_relu_1, W_conv3_2, strides=[1, 1, 1, 1], padding='SAME')
            h_conv3_relu_2 = tf.nn.relu(tf.add(h_conv3_2, b3_2))

            h_conv3_maxpool = tf.nn.max_pool(h_conv3_relu_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        return h_conv3_maxpool

    # convolutional network layer 3
    def conv4(self, input_data):
        with tf.name_scope('conv_4'):
            W_conv4_1 = tf.Variable(tf.truncated_normal([3, 3, 256, 512], stddev=1e-2))
            W_conv4_2 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=1e-2))

            b4_1 = tf.Variable(tf.truncated_normal([512], stddev=1e-2))
            b4_2 = tf.Variable(tf.truncated_normal([512], stddev=1e-2))

            h_conv4_1 = tf.nn.conv2d(input_data, W_conv4_1, strides=[1, 1, 1, 1], padding='SAME')
            h_conv4_relu_1 = tf.nn.relu(tf.add(h_conv4_1, b4_1))

            h_conv4_2 = tf.nn.conv2d(h_conv4_relu_1, W_conv4_2, strides=[1, 1, 1, 1], padding='SAME')
            h_conv4_relu_2 = tf.nn.relu(tf.add(h_conv4_2, b4_2))

            h_conv4_maxpool = tf.nn.max_pool(h_conv4_relu_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        return h_conv4_maxpool

    # convolutional network layer 5
    def conv5(self, input_data):
        with tf.name_scope('conv_5'):
            W_conv5_1 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=1e-2))
            W_conv5_2 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=1e-2))

            b5_1 = tf.Variable(tf.truncated_normal([512], stddev=1e-2))
            b5_2 = tf.Variable(tf.truncated_normal([512], stddev=1e-2))

            h_conv5_1 = tf.nn.conv2d(input_data, W_conv5_1, strides=[1, 1, 1, 1], padding='SAME')
            h_conv5_relu_1 = tf.nn.relu(tf.add(h_conv5_1, b5_1))

            h_conv5_2 = tf.nn.conv2d(h_conv5_relu_1, W_conv5_2, strides=[1, 1, 1, 1], padding='SAME')
            h_conv5_relu_2 = tf.nn.relu(tf.add(h_conv5_2, b5_2))

            h_conv5_maxpool = tf.nn.max_pool(h_conv5_relu_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        return h_conv5_maxpool

    # fully connected layer 1
    def fc1(self, input_data):
        input_layer_size = 7 * 7 * 512
        with tf.name_scope('fc_1'):
            input_data_reshape = tf.reshape(input_data, [-1, input_layer_size])
            w_fc1 = tf.Variable(tf.truncated_normal([input_layer_size, 4096], stddev=1e-2))
            b_fc1 = tf.Variable(tf.truncated_normal([4096], stddev=1e-2))
            h_fc1 = tf.add(tf.matmul(input_data_reshape, w_fc1), b_fc1)  # h_fc1 = input_data*W_fc1 + b_fc1
            h_fc1_relu = tf.nn.relu(h_fc1)

        return h_fc1_relu

    # fully connected layer 2
    def fc2(self, input_data):
        with tf.name_scope('fc_2'):
            W_fc2 = tf.Variable(tf.truncated_normal([4096, 4096], stddev=1e-2))
            b_fc2 = tf.Variable(tf.truncated_normal([4096], stddev=1e-2))
            h_fc2 = tf.add(tf.matmul(input_data, W_fc2), b_fc2)  # h_fc1 = input_data*W_fc1 + b_fc1
            h_fc2_relu = tf.nn.relu(h_fc2)

        return h_fc2_relu

    # final layer
    def final_out(self, input_data):
        with tf.name_scope('final_out'):
            W_fo = tf.Variable(tf.truncated_normal([4096, 4], stddev=1e-2))
            b_fo = tf.Variable(tf.truncated_normal([4], stddev=1e-2))
            h_fo = tf.add(tf.matmul(input_data, W_fo), b_fo)  # h_fc1 = input_data*W_fc1 + b_fc1
        return h_fo

    # build cnn_graph
    def build_model(self, images, keep_prob):
        r_cnn1 = self.conv1(images)  # convolutional layer 1
        r_cnn2 = self.conv2(r_cnn1)  # convolutional layer 2
        r_cnn3 = self.conv3(r_cnn2)  # convolutional layer 3
        r_cnn4 = self.conv4(r_cnn3)  # convolutional layer 4
        r_cnn5 = self.conv5(r_cnn4)  # convolutional layer 5
        r_fc1 = self.fc1(r_cnn5)  # fully connected layer1

        if not keep_prob == 1.0:
            r_fc1 = tf.nn.dropout(r_fc1, keep_prob)
        r_fc2 = self.fc2(r_fc1)  # fully connected layer2

        if not keep_prob == 1.0:
            r_fc2 = tf.nn.dropout(r_fc2, keep_prob)
        r_out = self.final_out(r_fc2)  # final layer

        return r_out
