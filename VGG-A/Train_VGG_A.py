import tensorflow as tf
from Package.Data_read import DataRead
from Package.VGG_A_Model import VGGModel
from datetime import datetime

tf.debugging.set_log_device_placement(True)

learning_rate = 5e-5
batch_size = 50
epochs = 100
data_read = DataRead(batch_size)
VGGModel = VGGModel()

X = tf.placeholder(tf.float32, shape=[None, 200, 200, 1])
Y = tf.placeholder(tf.float32, shape=[None, 4])
keep_prob = tf.placeholder(tf.float32)
iterator = data_read.made_train_batch().make_initializable_iterator()
test_iterator = data_read.made_train_batch().make_initializable_iterator()

prediction = VGGModel.build_model(X, keep_prob)
# define loss function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=Y))

train = tf.train.AdamOptimizer(learning_rate).minimize(cost)
label_max = tf.argmax(Y, 1)
pre_max = tf.argmax(prediction, 1)
correct_pred = tf.equal(pre_max, label_max)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

train_next = iterator.get_next()
test_next = test_iterator.get_next()

#tf.summary.scalar('cost', cost)
#tf.summary.scalar('accuracy', accuracy)

#summary = tf.summary.merge_all()

with tf.Session() as sess:
    startTime = datetime.now()
    sess.run(iterator.initializer)
    sess.run(test_iterator.initializer)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    #summary_writer = tf.summary.FileWriter('tensor_board_lego', sess.graph)

    for step in range(epochs):
        avg_cost = 0
        for x in range(int(data_read.data_size / batch_size)):
            x_data, y_data = sess.run(train_next)
            cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data, keep_prob: 0.5})
            avg_cost += cost_val / int(data_read.data_size / batch_size)
            if x % 10 == 0:
                validate_images_, validate_labels_ = sess.run(test_next)
                rv = sess.run([label_max, pre_max, cost, accuracy], feed_dict={X: validate_images_, Y: validate_labels_, keep_prob: 1.0})
                print('Validation cost:', rv[2], ' accuracy:', rv[3])

        now = datetime.now() - startTime

        print('step: ', step, 'cost_val : ', avg_cost, 'time', now)

    #summary_str = sess.run(summary, feed_dict={X: validate_images_, Y: validate_labels_, keep_prob: 1.0})
    #summary_writer.flush()
    saver.save(sess, 'Train/Lego_VGG_A')  # save session
