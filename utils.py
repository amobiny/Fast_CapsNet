import tensorflow as tf


def write_spec(args):
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
    elif args.model == 'alexnet' or args.model == 'resnet':
        config_file.write('dropout_rate: ' + str(args.dropout_rate))
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