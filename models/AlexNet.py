from base_model import BaseModel
import tensorflow as tf
from models.utils.ops_cnn import conv_layer, fc_layer, dropout, max_pool, lrn, relu
from tensorflow.contrib.layers import flatten


class AlexNet(BaseModel):
    def __init__(self, sess, conf):
        super(AlexNet, self).__init__(sess, conf)
        self.build_network(self.x)
        if self.conf.mode != 'train_sequence':
            self.configure_network()

    def build_network(self, x):
        # Building network...
        with tf.variable_scope('CapsNet'):
            net = lrn(relu(conv_layer(x, kernel_size=7, stride=2, num_filters=96, trainable=self.conf.trainable,
                                      add_reg=self.conf.L2_reg, layer_name='CONV1')))
            net = max_pool(net, pool_size=3, stride=2, padding='SAME', name='MaxPool1')
            net = lrn(relu(conv_layer(net, kernel_size=5, stride=2, num_filters=256, trainable=self.conf.trainable,
                                      add_reg=self.conf.L2_reg, layer_name='CONV2')))
            net = max_pool(net, pool_size=3, stride=2, padding='SAME', name='MaxPool2')
            net = relu(conv_layer(net, kernel_size=3, stride=1, num_filters=384, trainable=self.conf.trainable,
                                  add_reg=self.conf.L2_reg, layer_name='CONV3'))
            net = relu(conv_layer(net, kernel_size=3, stride=1, num_filters=384, trainable=self.conf.trainable,
                                  add_reg=self.conf.L2_reg, layer_name='CONV4'))
            net = relu(conv_layer(net, kernel_size=3, stride=1, num_filters=256, trainable=self.conf.trainable,
                                  add_reg=self.conf.L2_reg, layer_name='CONV5'))
            net = max_pool(net, pool_size=3, stride=2, padding='SAME', name='MaxPool3')
            layer_flat = flatten(net)
            net = relu(fc_layer(layer_flat, num_units=512, add_reg=self.conf.L2_reg,
                                trainable=self.conf.trainable, layer_name='FC1'))
            net = dropout(net, self.conf.dropout_rate, training=self.is_training)
            net = relu(fc_layer(net, num_units=512, add_reg=self.conf.L2_reg,
                                trainable=self.conf.trainable, layer_name='FC2'))
            net = dropout(net, self.conf.dropout_rate, training=self.is_training)
            self.features = net
            self.logits = fc_layer(net, num_units=self.conf.num_cls, add_reg=self.conf.L2_reg,
                                   trainable=self.conf.trainable, layer_name='FC3')
            # [?, num_cls]
            self.probs = tf.nn.softmax(self.logits)
            # [?, num_cls]
            self.y_pred = tf.to_int32(tf.argmax(self.probs, 1))
            # [?] (predicted labels)

