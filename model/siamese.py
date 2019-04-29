import tensorflow as tf 
from tensorflow.contrib import cudnn_rnn

class Siamese_net():
    def __init__(self,net='fc'):
        self.net = net
        self.x1 = tf.placeholder(tf.float32, [None, 28, 28])
        self.x2 = tf.placeholder(tf.float32, [None, 28, 28])
        self.y_ = tf.placeholder(tf.float32, [None])

        with tf.variable_scope('Siamese', reuse=None):
            self.o1 = self.network(self.x1)
        with tf.variable_scope('Siamese', reuse=True):
            self.o2 = self.network(self.x2)
        
        # Create loss
        self.loss = self.loss_function()

    
    def network(self, x):
        if self.net == 'fc':
            x = tf.reshape(x,[-1,28*28])
            y = tf.layers.dense(x,512,activation=tf.nn.relu)
            y = tf.layers.dense(y,256,activation=tf.nn.relu)
            output = tf.layers.dense(y,2,activation=None)

        elif self.net == 'rnn':
            x = tf.transpose(x,perm=[1,0,2])
            cell = cudnn_rnn.CudnnGRU(1,64)
            o, _ = cell(x)
            output = tf.layers.dense(o[-1,:,:],2,activation=None)
        else:
            ## conv1 layer ##
            x = tf.reshape(x, [-1, 28, 28, 1])
            W_conv1 = self.weight_variable([5,5, 1,32]) # patch 5x5, in size 1, out size 32
            b_conv1 = self.bias_variable([32])
            h_conv1 = tf.nn.relu(self.conv2d(x, W_conv1) + b_conv1) # output size 28x28x32
            h_pool1 = self.max_pool_2x2(h_conv1) 
            ## conv2 layer ##
            W_conv2 = self.weight_variable([5,5,32,64]) # patch 5x5, in size 1, out size 32
            b_conv2 = self.bias_variable([64])
            h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2) # output size 28x28x32
            h_pool2 = self.max_pool_2x2(h_conv2) 
            h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
            # FC_layer
            o = tf.layers.dense(h_pool2_flat,1024,activation=tf.nn.relu)
            output = tf.layers.dense(o,2,activation=None)

        return output

    def conv2d(self,x, W):
        # stride [1, x_movement, y_movement, 1]
        # Must have strides[0] = strides[3] = 1
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    def max_pool_2x2(self,x):
        # stride [1, x_movement, y_movement, 1]
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    def bias_variable(self,shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def loss_function(self):
        
        labels_t = self.y_
        labels_f = 1.0-self.y_        # labels_ = !labels;
        eucd2 = tf.pow((self.o1-self.o2), 2)
        eucd2 = tf.reduce_sum(eucd2, 1)  # 欧式距离
        eucd = tf.sqrt(eucd2+1e-6, name="eucd")
        #######################################################
        # margin = 5.0
        # C = tf.constant(margin, name="C")
        # pos = labels_t*eucd # 相同类别情形  y*||o1-o2||
        # # neg = labels_f * tf.maximum(0.0, C-eucd)  #不同类别情形 （1-y)*min(0,c-||o1-o2||)
        # neg = labels_f * tf.pow(tf.maximum(C-eucd, 0), 2)
        ######################################################
        # pos = labels_t*tf.pow(eucd,2)
        # neg = labels_f *tf.exp(-2.77*eucd)

        ########################################################
        pos = labels_t*tf.square(eucd)
        neg = labels_f *tf.square(tf.maximum((1 - eucd),0))

        losses = pos+neg
        loss = tf.reduce_mean(losses, name="loss")
        return loss 
