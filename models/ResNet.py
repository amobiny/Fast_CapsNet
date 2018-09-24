from base_model import BaseModel
import tensorflow as tf
from models.utils.ops_cnn import batch_normalization, relu, conv_layer, dropout, average_pool, fc_layer, max_pool, \
    global_average_pool, flatten, concatenation
from collections import namedtuple


class ResNet(BaseModel):
    def __init__(self, sess, conf):
        super(ResNet, self).__init__(sess, conf)
        # Configurations for each bottleneck group.
        BottleneckGroup = namedtuple('BottleneckGroup',
                                     ['num_blocks', 'bottleneck_size', 'out_filters'])
        self.groups = [BottleneckGroup(3, 32, 128), BottleneckGroup(4, 64, 256),
                       BottleneckGroup(6, 128, 512), BottleneckGroup(3, 256, 512)]
        self.build_network(self.x)
        if self.conf.mode != 'train_sequence':
            self.configure_network()

    def build_network(self, x):
        # Building network...
        with tf.variable_scope('ResNet'):
            net = conv_layer(x, num_filters=64, kernel_size=7, stride=1, add_reg=self.conf.L2_reg, layer_name='CONV0')
            net = relu(batch_normalization(net, training=self.is_training, scope='BN1'))
            # net = max_pool(net, pool_size=3, stride=2, name='MaxPool0')

            # Create the bottleneck groups, each of which contains `num_blocks` bottleneck blocks.
            for group_i, group in enumerate(self.groups):
                first_block = True
                for block_i in range(group.num_blocks):
                    block_name = 'group_%d/block_%d' % (group_i, block_i)
                    net = self.bottleneck_block(net, group, block_name, is_first_block=first_block)
                    first_block = False

            net = global_average_pool(net)
            net = flatten(net)
            self.features = net
            self.logits = fc_layer(net, num_units=self.conf.num_cls, add_reg=self.conf.L2_reg, layer_name='Fc1')
            # [?, num_cls]
            self.probs = tf.nn.softmax(self.logits)
            # [?, num_cls]
            self.y_pred = tf.to_int32(tf.argmax(self.probs, 1))
            # [?] (predicted labels)

    def bottleneck_block(self, input_x, group, name, is_first_block=False):
        with tf.variable_scope(name):
            # 1x1 convolution responsible for reducing the depth
            with tf.variable_scope('conv_in'):
                stride = 2 if is_first_block else 1
                conv = conv_layer(input_x, num_filters=group.bottleneck_size, add_reg=self.conf.L2_reg,
                                  kernel_size=1, stride=stride, layer_name='CONV')
                conv = relu(batch_normalization(conv, self.is_training, scope='BN'))

            with tf.variable_scope('conv_bottleneck'):
                conv = conv_layer(conv, num_filters=group.bottleneck_size, kernel_size=3,
                                  add_reg=self.conf.L2_reg, layer_name='CONV')
                conv = relu(batch_normalization(conv, self.is_training, scope='BN'))

            # 1x1 convolution responsible for increasing the depth
            with tf.variable_scope('conv_out'):
                conv = conv_layer(conv, num_filters=group.out_filters, kernel_size=1,
                                  add_reg=self.conf.L2_reg, layer_name='CONV')
                conv = batch_normalization(conv, self.is_training, scope='BN')

            # shortcut connections that turn the network into its counterpart
            # residual function (identity shortcut)
            with tf.variable_scope('shortcut'):
                if is_first_block:
                    shortcut = conv_layer(input_x, num_filters=group.out_filters, stride=2, kernel_size=1,
                                          add_reg=self.conf.L2_reg, layer_name='CONV_shortcut')
                    shortcut = batch_normalization(shortcut, self.is_training, scope='BN_shortcut')
                    assert (shortcut.get_shape().as_list() == conv.get_shape().as_list()), \
                        "Tensor sizes of the two branches are not matched!"
                    res = shortcut + conv
                else:
                    res = conv + input_x
                    assert (input_x.get_shape().as_list() == conv.get_shape().as_list()), \
                        "Tensor sizes of the two branches are not matched!"
            return relu(res)

