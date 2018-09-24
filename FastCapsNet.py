from base_model import BaseModel
import tensorflow as tf
from config import *
from ops import *
import numpy as np


class FastCapsNet3D(BaseModel):
    def __init__(self, sess, conf):
        super(FastCapsNet3D, self).__init__(sess, conf)
        self.build_network()
        if self.conf.mode != 'train_sequence' and self.conf.mode != 'test_sequence':
            self.configure_network()

    def build_network(self):
        # Building network...
        with tf.variable_scope('CapsNet'):
            with tf.variable_scope('Conv1_layer'):
                conv1 = tf.layers.conv3d(self.x, name="conv1", **conv1_params)
                # [batch_size, 24, 24, 12, 256]

            with tf.variable_scope('PrimaryCaps_layer'):
                conv2 = tf.layers.conv3d(conv1, name="conv2", **conv2_params)
                # [batch_size, 8, 8, 8, 256]
                caps1_raw = tf.reshape(conv2, (args.batch_size, caps1_n_caps, caps1_n_dims, 1), name="caps1_raw")
                # [batch_size, 2048, 8, 1]
                # [batch_size, 8*8*8, 256, 1]
                caps1_output = squash(caps1_raw, name="caps1_output")
                # [batch_size, 2048, 8, 1]
                # [batch_size, 512, 256, 1]

            # DigitCaps layer, return [batch_size, 10, 16, 1]
            with tf.variable_scope('DigitCaps_layer'):
                caps2_input = tf.reshape(caps1_output, shape=(args.batch_size, caps1_n_caps, 1, caps1_n_dims, 1))
                # [batch_size, 2048, 1, 8, 1]
                # [batch_size, 512, 1, 256, 1] 512 capsules of 256D
                b_IJ = tf.zeros([args.batch_size, caps1_n_caps, caps2_n_caps, 1, 1], dtype=np.float32, name="b_ij")
                # [batch_size, 2048, 2, 1, 1]
                # [batch_size, 512, 2, 1, 1]
                self.caps2_output, u_hat = routing(caps2_input, b_IJ, caps2_n_dims)
                # [batch_size, 2, 16, 1], [batch_size, 512, 2, 16, 1]
                u_hat_shape = u_hat.get_shape().as_list()
                # img_s = np.sqrt(u_hat_shape[1]/caps1_n_maps).astype(int)
                img_s = int(round(u_hat_shape[1] ** (1. / 3)))
                u_hat = tf.reshape(u_hat, (args.batch_size, img_s, img_s, img_s, caps1_n_maps, caps2_n_caps, -1))
                # u_hat: [batch_size, 8, 8, 8, 32, 2, 16]
                # u_hat: [batch_size, 8, 8, 8, 1, 2, 16]

            # Decoder
            with tf.variable_scope('Masking'):
                epsilon = 1e-9
                self.v_length = tf.sqrt(tf.reduce_sum(tf.square(self.caps2_output), axis=2, keep_dims=True) + epsilon)
                # [batch_size, 2, 1, 1]
                indices = tf.argmax(tf.squeeze(self.v_length), axis=1)
                self.y_prob = tf.nn.softmax(tf.squeeze(self.v_length))
                y_prob_argmax = tf.to_int32(tf.argmax(self.v_length, axis=1))
                # [batch_size, 1, 1]
                self.y_pred = tf.reshape(y_prob_argmax, shape=(args.batch_size,))
                # [batch_size] (predicted labels)
                y_pred_ohe = tf.one_hot(self.y_pred, depth=caps2_n_caps)
                # [batch_size, 2] (one-hot-encoded predicted labels)

                reconst_targets = tf.cond(self.mask_with_labels,  # condition
                                          lambda: self.Y,  # if True (Training)
                                          lambda: y_pred_ohe,  # if False (Test)
                                          name="reconstruction_targets")
                # [batch_size, 2]
                reconst_targets = tf.reshape(reconst_targets, (args.batch_size, 1, 1, 1, args.n_cls))
                # [batch_size, 1, 1, 2]
                reconst_targets = tf.tile(reconst_targets, (1, img_s, img_s, img_s, 1))
                # [batch_size, 8, 8, 8, 2]

                self.u_hat = tf.transpose(u_hat, perm=[5, 0, 1, 2, 3, 4, 6])
                # u_hat: [2, batch_size, 8, 8, 8, 32, 16]
                # u_hat: [2, batch_size, 8, 8, 8, 1, 16]
                u_list = tf.unstack(self.u_hat, axis=1)
                ind_list = tf.unstack(indices, axis=0)
                a = tf.stack([tf.gather_nd(mat, [[ind]]) for mat, ind in zip(u_list, ind_list)])
                # [batch_size, 1, 8, 8, 8, 32, 16]
                # [batch_size, 1, 8, 8, 8, 1, 16]
                feat = tf.reshape(tf.transpose(a, perm=[0, 2, 3, 4, 1, 5, 6]),
                                  (args.batch_size, img_s, img_s, img_s, -1))
                # [batch_size, 8, 8, 8, 32*16]
                # [batch_size, 8, 8, 8, 16]
                self.cube = tf.concat([feat, reconst_targets], axis=-1)
                # [batch_size, 8, 8, 8, 514]
                # [batch_size, 8, 8, 8, 18]

            with tf.variable_scope('Decoder'):
                res1 = deconv3d(self.cube, [args.batch_size, 16, 16, 16, 16],
                                k_h=4, k_w=4, k_d=4, d_h=2, d_w=2, d_d=2, stddev=0.02, name="deconv_1")
                self.decoder_output = deconv3d(res1, [args.batch_size, 32, 32, 32, 1],
                                               k_h=4, k_w=4, k_d=4, d_h=2, d_w=2, d_d=2, stddev=0.02, name="deconv_2")
