import tensorflow as tf


def spread_loss(labels, activations, margin, name):
    """This adds spread loss to total loss.
  :param labels: [N, O], where O is number of output classes, one hot vector, tf.uint8.
  :param activations: [N, O], activations.
  :param margin: margin 0.2 - 0.9 fixed schedule during training.
  :return: spread loss
  """
    activations_shape = activations.get_shape().as_list()
    with tf.variable_scope(name):
        mask_t = tf.equal(labels, 1)
        mask_i = tf.equal(labels, 0)

        activations_t = tf.reshape(tf.boolean_mask(activations, mask_t), [activations_shape[0], 1])
        activations_i = tf.reshape(tf.boolean_mask(activations, mask_i),
                                   [activations_shape[0], activations_shape[1] - 1])
        gap_mit = tf.reduce_sum(tf.square(tf.nn.relu(margin - (activations_t - activations_i))))
        return gap_mit


def margin_loss(y, v_length, conf):
    with tf.variable_scope('Margin_Loss'):
        # max(0, m_plus-||v_c||)^2
        present_error = tf.square(tf.maximum(0., conf.m_plus - v_length))
        # [?, 10, 1]
        # max(0, ||v_c||-m_minus)^2
        absent_error = tf.square(tf.maximum(0., v_length - conf.m_minus))
        # [?, 10, 1]
        # reshape: [?, 10, 1] => [?, 10]
        present_error = tf.squeeze(present_error)
        absent_error = tf.squeeze(absent_error)
        T_c = y
        # [?, 10]
        L_c = T_c * present_error + conf.lambda_val * (1 - T_c) * absent_error
        # [?, 10]
        margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1), name="margin_loss")
    return margin_loss


def cross_entropy(y, logits):
    try:
        diff = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)
    except:
        diff = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(diff)
    return loss
