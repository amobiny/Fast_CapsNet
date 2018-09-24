import tensorflow as tf


def write_spec(args):
    if args.add_rnn:
        config_file = open(args.rnn_modeldir + args.rnn_run_name + '/config.txt', 'w')
    else:
        config_file = open(args.modeldir + args.run_name + '/config.txt', 'w')
    config_file.write('run_name: ' + args.run_name + '\n')
    config_file.write('model: ' + args.model + '\n')
    config_file.write('loss_type: ' + args.loss_type + '\n')
    config_file.write('add_recon_loss: ' + str(args.add_recon_loss) + '\n')
    config_file.write('data: ' + args.data + '\n')
    config_file.write('height: ' + str(args.height) + '\n')
    config_file.write('num_cls: ' + str(args.num_cls) + '\n')
    config_file.write('batch_size: ' + str(args.batch_size) + '\n')
    config_file.write('optimizer: ' + 'Adam' + '\n')
    config_file.write('learning_rate: ' + str(args.init_lr) + ' : ' + str(args.lr_min) + '\n')
    config_file.write('data_augmentation: ' + str(args.data_augment) + '\n')
    config_file.write('max_angle: ' + str(args.max_angle) + '\n')
    if args.model == 'original_capsule':
        config_file.write('prim_caps_dim: ' + str(args.prim_caps_dim) + '\n')
        config_file.write('digit_caps_dim: ' + str(args.digit_caps_dim) + '\n')
    elif args.model == 'matrix_capsule':
        config_file.write('use_bias: ' + str(args.use_bias) + '\n')
        config_file.write('batch_normalization: ' + str(args.use_BN) + '\n')
        config_file.write('add_coords: ' + str(args.add_coords) + '\n')
        config_file.write('L2_reg: ' + str(args.L2_reg) + '\n')
        config_file.write('A: ' + str(args.A) + '\n')
        config_file.write('B: ' + str(args.B) + '\n')
        config_file.write('C: ' + str(args.C) + '\n')
        config_file.write('D: ' + str(args.D) + '\n')
    elif args.model == 'alexnet':
        config_file.write('dropout_rate: ' + str(args.dropout_rate))
    if args.add_rnn:
        config_file.write('trainable: ' + str(args.trainable) + '\n')
        config_file.write('recurrent_model: ' + args.recurrent_model + '\n')
        config_file.write('num layers: ' + str(args.num_layers) + '\n')
        config_file.write('num hidden: ' + str(args.num_hidden) + '\n')
        config_file.write('in_keep_prob: ' + str(args.in_keep_prob) + '\n')
        config_file.write('out_keep_prob: ' + str(args.out_keep_prob) + '\n')
        config_file.write('rnn_run_name: ' + args.rnn_run_name + '\n')
    if args.recurrent_model == "MANN":
        config_file.write('memory_size: ' + str(args.memory_size) + '\n')
        config_file.write('memory_vector_dim: ' + str(args.memory_vector_dim) + '\n')
        config_file.write('read_head_num: ' + str(args.read_head_num) + '\n')
    config_file.close()


def weight_variable(shape):
    """
    Create a weight variable with appropriate initialization
    :param shape: weight shape
    :return: initialized weight variable
    """
    initer = tf.truncated_normal_initializer(stddev=0.01)
    return tf.get_variable('W',
                           dtype=tf.float32,
                           shape=shape,
                           initializer=initer)


def bias_variable(shape):
    """
    Create a bias variable with appropriate initialization
    :param shape: bias variable shape
    :return: initialized bias variable
    """
    initer = tf.constant(0., shape=shape, dtype=tf.float32)
    return tf.get_variable('b',
                           dtype=tf.float32,
                           initializer=initer)