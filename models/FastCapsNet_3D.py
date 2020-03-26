from models.base_model import BaseModel
import tensorflow as tf
from models.utils.ops_caps import *
import numpy as np


class FastCapsNet3D(BaseModel):
    def __init__(self, sess, conf):
        super(FastCapsNet3D, self).__init__(sess, conf)
        self.build_network()
        self.configure_network()

    def build_network(self):
        # Building network...
        with tf.variable_scope('CapsNet'):
            with tf.variable_scope('Conv1_layer'):
                conv1 = tf.layers.conv3d(self.x, filters=256, kernel_size=9, strides=1,
                                         padding='valid', activation=tf.nn.relu, name="conv1")
                # [batch_size, 24, 24, 24, 256]

            with tf.variable_scope('PrimaryCaps_layer'):
                conv2 = tf.layers.conv3d(conv1, filters=256, kernel_size=9, strides=2,
                                         padding='valid', activation=tf.nn.relu, name="conv2")
                # [batch_size, 8, 8, 8, 256]
                shape = conv2.get_shape().as_list()
                num_prim_caps = int(shape[1] * shape[2] * shape[3] * shape[4] / self.conf.prim_caps_dim)
                caps1_raw = tf.reshape(conv2, (self.conf.batch_size, num_prim_caps,
                                               self.conf.prim_caps_dim, 1), name="caps1_raw")
                # [batch_size, 8*8*8, 256, 1]
                caps1_output = squash(caps1_raw, name="caps1_output")
                # [batch_size, 512, 256, 1]

            # DigitCaps layer, return [batch_size, 10, 16, 1]
            with tf.variable_scope('DigitCaps_layer'):
                caps2_input = tf.reshape(caps1_output,
                                         shape=(self.conf.batch_size, num_prim_caps, 1, self.conf.prim_caps_dim, 1))
                # [batch_size, 512, 1, 256, 1] 512 capsules of 256D
                b_IJ = tf.zeros([self.conf.batch_size, num_prim_caps, self.conf.num_cls, 1, 1],
                                dtype=np.float32, name="b_ij")
                # [batch_size, 512, 2, 1, 1]
                self.caps2_output, u_hat = routing(caps2_input, b_IJ, self.conf.digit_caps_dim)
                # [batch_size, 2, 16, 1], [batch_size, 512, 2, 16, 1]
                u_hat_shape = u_hat.get_shape().as_list()
                self.img_s = int(round(u_hat_shape[1] ** (1. / 3)))
                self.u_hat = tf.reshape(u_hat,
                                        (self.conf.batch_size, self.img_s, self.img_s, self.img_s, 1, self.conf.num_cls,
                                         -1))
                # [batch_size, 8, 8, 8, 1, 2, 16]

                epsilon = 1e-9
                self.v_length = tf.squeeze(tf.sqrt(tf.reduce_sum(
                    tf.square(self.caps2_output), axis=2, keep_dims=True) + epsilon))
                # [batch_size, 2]
                self.y_pred = tf.to_int32(tf.argmax(self.v_length, axis=1))
                # [batch_size,] (predicted labels)

            if self.conf.add_recon_loss:
                self.mask()
                self.decoder()

    def mask(self):
        with tf.variable_scope('Masking'):
            y_pred_ohe = tf.one_hot(self.y_pred, depth=self.conf.num_cls)
            # [batch_size, 2] (one-hot-encoded predicted labels)

            reconst_targets = tf.cond(self.is_training,  # condition
                                      lambda: self.y,  # if True (Training)
                                      lambda: y_pred_ohe,  # if False (Test)
                                      name="reconstruction_targets")
            # [batch_size, 2]
            reconst_targets = tf.reshape(reconst_targets, (self.conf.batch_size, 1, 1, 1, self.conf.num_cls))
            # [batch_size, 1, 1, 2]
            reconst_targets = tf.tile(reconst_targets, (1, self.img_s, self.img_s, self.img_s, 1))
            # [batch_size, 8, 8, 8, 2]
            indices = tf.argmax(self.v_length, axis=1)
            self.u_hat = tf.transpose(self.u_hat, perm=[5, 0, 1, 2, 3, 4, 6])
            # u_hat: [2, batch_size, 8, 8, 8, 1, 16]
            u_list = tf.unstack(self.u_hat, axis=1)
            ind_list = tf.unstack(indices, axis=0)
            a = tf.stack([tf.gather_nd(mat, [[ind]]) for mat, ind in zip(u_list, ind_list)])
            # [batch_size, 1, 8, 8, 8, 1, 16]
            feat = tf.reshape(tf.transpose(a, perm=[0, 2, 3, 4, 1, 5, 6]),
                              (self.conf.batch_size, self.img_s, self.img_s, self.img_s, -1))
            # [batch_size, 8, 8, 8, 16]
            self.cube = tf.concat([feat, reconst_targets], axis=-1)
            # [batch_size, 8, 8, 8, 18]

    def decoder(self):
        with tf.variable_scope('Decoder'):
            res1 = deconv3d(self.cube, [self.conf.batch_size, 16, 16, 16, 16],
                            k_h=4, k_w=4, k_d=4, d_h=2, d_w=2, d_d=2, stddev=0.02, name="deconv_1")
            self.decoder_output = deconv3d(res1, [self.conf.batch_size, 32, 32, 32, 1],
                                           k_h=4, k_w=4, k_d=4, d_h=2, d_w=2, d_d=2, stddev=0.02, name="deconv_2")
