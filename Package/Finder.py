import numpy as np
import tensorflow as tf
from Package.VGG_A_Model import VGGModel


class Finder:

    def __init__(self):
        self.Model = VGGModel()
        self.X = tf.placeholder(tf.float32, shape=[None, 200, 200, 1])
        self.keep_prob = tf.placeholder(tf.float32)
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        self.prediction = tf.nn.softmax(self.Model.build_model(self.X, self.keep_prob))

        saver = tf.compat.v1.train.Saver()
        saver.restore(self.sess, 'Train/Lego_VGG_A')

    def find(self, img):

        self.img = img.reshape([-1, img.shape[0], img.shape[1], 1])
        self.img = self.img.astype(np.int32)
        self.img = tf.image.resize(self.img, [200, 200])
        self.img = self.sess.run(self.img)
        p_val = self.sess.run(self.prediction, feed_dict={self.X: self.img, self.keep_prob: 1.})

        name_labels = ['2357', '3003', '3004', '3005']
        i = 0
        lab = 0
        for x in p_val[0]:
            print('%s              %f' % (name_labels[i], float(x)))
            if x == max(p_val[0]):
                lab = i
            i += 1
        return float(max(p_val[0])), name_labels[lab]
