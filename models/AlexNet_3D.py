from base_model import BaseModel
import tensorflow as tf
from models.utils.ops_cnn import conv_layer_3d, fc_layer, dropout, max_pool_3d, relu, batch_normalization
from tensorflow.contrib.layers import flatten


class AlexNet3D(BaseModel):
    def __init__(self, sess, conf):
        super(AlexNet3D, self).__init__(sess, conf)
        self.build_network(self.x)
        if self.conf.mode != 'train_sequence':
            self.configure_network()

    def build_network(self, x):
        # Building network...
        with tf.variable_scope('CapsNet'):
            net = batch_normalization(relu(conv_layer_3d(x, kernel_size=7, stride=2, num_filters=96,
                                                         add_reg=self.conf.L2_reg, layer_name='CONV1')),
                                      training=self.is_training, scope='BN1')
            net = max_pool_3d(net, pool_size=3, stride=2, padding='SAME', name='MaxPool1')
            net = batch_normalization(relu(conv_layer_3d(net, kernel_size=5, stride=2, num_filters=256,
                                                         add_reg=self.conf.L2_reg, layer_name='CONV2')),
                                      training=self.is_training, scope='BN2')
            net = max_pool_3d(net, pool_size=3, stride=2, padding='SAME', name='MaxPool2')
            net = batch_normalization(relu(conv_layer_3d(net, kernel_size=3, stride=1, num_filters=384,
                                                         add_reg=self.conf.L2_reg, layer_name='CONV3')),
                                      training=self.is_training, scope='BN3')
            net = batch_normalization(relu(conv_layer_3d(net, kernel_size=3, stride=1, num_filters=384,
                                                         add_reg=self.conf.L2_reg, layer_name='CONV4')),
                                      training=self.is_training, scope='BN4')
            net = batch_normalization(relu(conv_layer_3d(net, kernel_size=3, stride=1, num_filters=256,
                                                         add_reg=self.conf.L2_reg, layer_name='CONV5')),
                                      training=self.is_training, scope='BN5')
            net = max_pool_3d(net, pool_size=3, stride=2, padding='SAME', name='MaxPool3')
            layer_flat = flatten(net)
            net = relu(fc_layer(layer_flat, num_units=200, add_reg=self.conf.L2_reg, layer_name='FC1'))
            net = dropout(net, self.conf.dropout_rate, training=self.is_training)
            net = relu(fc_layer(net, num_units=75, add_reg=self.conf.L2_reg, layer_name='FC2'))
            net = dropout(net, self.conf.dropout_rate, training=self.is_training)
            self.features = net
            self.logits = fc_layer(net, num_units=self.conf.num_cls, add_reg=self.conf.L2_reg, layer_name='FC3')
            # [?, num_cls]
            self.probs = tf.nn.softmax(self.logits)
            # [?, num_cls]
            self.y_pred = tf.to_int32(tf.argmax(self.probs, 1))
            # [?] (predicted labels)
