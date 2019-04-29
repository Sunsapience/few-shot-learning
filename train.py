import tensorflow as tf

from model.siamese import Siamese_net 
from data_utils import batch_test_different, batch_test_same, batch_train_flow

siamese = Siamese_net('rnn')
# train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(siamese.loss)
train_op = tf.train.GradientDescentOptimizer(1e-2).minimize(siamese.loss)

def run_epoch(sess,epoch,Model,data_queue, train_op):
    steps = 0

    feed_dict = {}
    total_loss = 0
    for x_1,x_2,y in data_queue:
        feed_dict[Model.x1] = x_1
        feed_dict[Model.x2] = x_2
        feed_dict[Model.y_] = y
        output = [train_op,Model.loss]
        _,loss = sess.run(output, feed_dict=feed_dict)       

        if steps % 200 == 0:
            print('epoch:\t{}\tsteps:\t{}\ttrain_loss:\t{}'.format(
                epoch,steps,float('%.6f' % loss)))       
        steps += 1
        total_loss += loss

    return round(total_loss/steps,6)

saver = tf.train.Saver(max_to_keep=3)
base_line = 0.01
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # saver.restore(sess,'./save/model--20')

    for i in range(20):
        train_queue = batch_train_flow(100)
        train_loss = run_epoch(sess, i+1, siamese, train_queue, train_op)
        print('epoch:\t{}\ttrain_loss:\t{}'.format(i+1,float('%.6f' % train_loss))) 
        
        if train_loss < base_line and i>0:
            base_line = train_loss
            saver.save(sess,'./save/model-',global_step=i+1)
        
