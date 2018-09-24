import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('mode', 'train', 'train, train_sequence, test, test_sequence or get_features')
flags.DEFINE_integer('reload_step', 0, 'model number to load (either for testing or continue training)')
flags.DEFINE_string('run_name', 'test', 'Run name')
flags.DEFINE_string('model', 'fast_capsule', 'alexnet, resnet, original_capsule, fast_capsule')
flags.DEFINE_string('loss_type', 'margin', 'cross_entropy, spread or margin')
flags.DEFINE_boolean('add_recon_loss', True, 'To add reconstruction loss')
flags.DEFINE_boolean('L2_reg', False, 'Adds L2-regularization to all the network weights')
flags.DEFINE_float('lmbda', 5e-04, 'L2-regularization coefficient')

# Training logs
flags.DEFINE_integer('max_step', 100000, '# of step for training (only for mnist)')
flags.DEFINE_integer('max_epoch', 1000, '# of step for training (only for nodule data)')
flags.DEFINE_boolean('epoch_based', True, 'Running the training in epochs')
flags.DEFINE_integer('SAVE_FREQ', 1000, 'Number of steps to save model')
flags.DEFINE_integer('SUMMARY_FREQ', 100, 'Number of step to save summary')
flags.DEFINE_integer('VAL_FREQ', 500, 'Number of step to evaluate the network on Validation data')

# For margin loss
flags.DEFINE_float('m_plus', 0.9, 'm+ parameter')
flags.DEFINE_float('m_minus', 0.1, 'm- parameter')
flags.DEFINE_float('lambda_val', 0.5, 'Down-weighting parameter for the absent class')
# For reconstruction loss
flags.DEFINE_float('alpha', 0.0005, 'Regularization coefficient to scale down the reconstruction loss')
# For training
flags.DEFINE_integer('batch_size', 4, 'training batch size')
flags.DEFINE_float('init_lr', 1e-4, 'Initial learning rate')
flags.DEFINE_float('lr_min', 1e-5, 'Minimum learning rate')

# data
flags.DEFINE_string('data', 'nodule', 'nodule')
flags.DEFINE_integer('num_cls', 2, 'Number of output classes')
flags.DEFINE_float('percent', 0.6, 'Percentage of training data to use')
flags.DEFINE_integer('dim', 2, '2D or 3D for nodule data')
flags.DEFINE_boolean('one_hot', False, 'one-hot-encodes the labels')
flags.DEFINE_boolean('data_augment', True, 'Adds augmentation to data')
flags.DEFINE_boolean('flip', True, 'Flips the data left to right and side to side')
flags.DEFINE_integer('max_angle', 180, 'Maximum rotation angle along each axis; when applying augmentation')
flags.DEFINE_integer('height', 32, 'Network input height size')
flags.DEFINE_integer('width', 32, 'Network input width size')
flags.DEFINE_integer('depth', 32, 'Network input depth size (in the case of 3D input images)')
flags.DEFINE_integer('channel', 1, 'Network input channel size')

# Directories
flags.DEFINE_string('logdir', './Results/log_dir/', 'Logs directory')
flags.DEFINE_string('modeldir', './Results/model_dir/', 'Saved models directory')
flags.DEFINE_string('model_name', 'model', 'Model file name')


# CapsNet architecture
flags.DEFINE_integer('prim_caps_dim', 8, 'Dimension of the PrimaryCapsules')
flags.DEFINE_integer('digit_caps_dim', 32, 'Dimension of the DigitCapsules')
flags.DEFINE_integer('h1', 512, 'Number of hidden units of the first FC layer of the reconstruction network')
flags.DEFINE_integer('h2', 1024, 'Number of hidden units of the second FC layer of the reconstruction network')

# Parameters of Conv1_layer
conv1_params = {"filters": 256,
                "kernel_size": 9,
                "strides": 1,
                "padding": "valid",
                "activation": tf.nn.relu}

flags.DEFINE_integer('prim_caps_dim', 8, 'Dimension of the PrimaryCapsules')
flags.DEFINE_integer('digit_caps_dim', 32, 'Dimension of the DigitCapsules')

# Parameters of PrimaryCaps_layer
caps1_n_maps = 1
caps1_n_caps = caps1_n_maps * 8 * 8 * 8  # 2048 primary capsules
caps1_n_dims = 256

# Parameters of Conv2_layer
conv2_params = {"filters": 256,
                "kernel_size": 9,
                "strides": 2,
                "padding": "valid",
                "activation": tf.nn.relu}

args = tf.app.flags.FLAGS
