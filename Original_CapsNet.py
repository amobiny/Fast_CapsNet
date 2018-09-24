from base_model import BaseModel
import tensorflow as tf
from ops import *


class OrigCapsNet(BaseModel):
    def __init__(self, sess, conf):
        super(OrigCapsNet, self).__init__(sess, conf)
        self.build_network(self.x)
        if self.conf.mode != 'train_sequence' and self.conf.mode != 'test_sequence':
            self.configure_network()

    def build_network(self, x):
        # Building network...
        with tf.variable_scope('CapsNet'):
            # Layer 1: A 2D conv layer
            conv1 = tf.keras.layers.Conv2D(filters=256, kernel_size=9, strides=1, trainable=self.conf.trainable,
                                           padding='valid', activation='relu', name='conv1')(x)

            # Layer 2: Primary Capsule Layer; simply a 2D conv + reshaping
            primary_caps = tf.keras.layers.Conv2D(filters=256, kernel_size=9, strides=2, trainable=self.conf.trainable,
                                                  padding='valid', activation='relu', name='primary_caps')(conv1)
            _, H, W, dim = primary_caps.get_shape()
            num_caps = H.value * W.value * dim.value / self.conf.prim_caps_dim
            primary_caps_reshaped = tf.keras.layers.Reshape((num_caps, self.conf.prim_caps_dim))(primary_caps)
            caps1_output = squash(primary_caps_reshaped)

            # Layer 3: Digit Capsule Layer; Here is where the routing takes place
            self.digit_caps = FCCapsuleLayer(num_caps=self.conf.num_cls, caps_dim=self.conf.digit_caps_dim,
                                             routings=3, name='digit_caps', trainable=self.conf.trainable)(caps1_output)
            # [?, 2, 16]

            epsilon = 1e-9
            self.v_length = tf.sqrt(tf.reduce_sum(tf.square(self.digit_caps), axis=2, keep_dims=True) + epsilon)
            # [?, 2, 1]
            y_prob_argmax = tf.to_int32(tf.argmax(self.v_length, axis=1))
            # [?, 1]
            self.y_pred = tf.squeeze(y_prob_argmax)
            # [?] (predicted labels)

            if self.conf.add_recon_loss:
                self.mask()
                self.decoder()

    def mask(self):  # used in capsule network
        with tf.variable_scope('Masking'):
            y_pred_ohe = tf.one_hot(self.y_pred, depth=self.conf.num_cls)
            # [?, 10] (one-hot-encoded predicted labels)

            reconst_targets = tf.cond(self.is_training,  # condition
                                      lambda: self.y,  # if True (Training)
                                      lambda: y_pred_ohe,  # if False (Test)
                                      name="reconstruction_targets")
            # [?, 10]
            self.output_masked = tf.multiply(self.digit_caps, tf.expand_dims(reconst_targets, -1))
            # [?, 2, 16]

    def decoder(self):
        with tf.variable_scope('Decoder'):
            decoder_input = tf.reshape(self.output_masked, [-1, self.conf.num_cls * self.conf.digit_caps_dim])
            # [?, 160]
            fc1 = tf.layers.dense(decoder_input, self.conf.h1, activation=tf.nn.relu, name="FC1",
                                  trainable=self.conf.trainable)
            # [?, 512]
            fc2 = tf.layers.dense(fc1, self.conf.h2, activation=tf.nn.relu, name="FC2", trainable=self.conf.trainable)
            # [?, 1024]
            self.decoder_output = tf.layers.dense(fc2, self.conf.width * self.conf.height * self.conf.channel,
                                                  activation=tf.nn.sigmoid, name="FC3", trainable=self.conf.trainable)
            # [?, 784]
