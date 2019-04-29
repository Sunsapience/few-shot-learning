import pickle
import tensorflow as tf

import visualize
from model.siamese import Siamese_net 
from data_utils import batch_test_different, batch_test_same, batch_test_train

siamese = Siamese_net('fc')

test_x = batch_test_different()
# test_x = batch_test_train()
# test_x = batch_test_same()
# test_x = batch_test_different()+batch_test_same()

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess,'./save/model--15')
    embed = sess.run(siamese.o1,feed_dict={siamese.x1:test_x})

    visualize.visualize(embed, test_x)

