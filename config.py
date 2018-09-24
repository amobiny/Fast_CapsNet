import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('mode', 'train', 'train, train_sequence, test, test_sequence or get_features')
flags.DEFINE_integer('reload_step', 0, 'model number to load (either for testing or continue training)')
flags.DEFINE_string('model', 'resnet', 'alexnet, resnet, original_capsule, fast_capsule')
flags.DEFINE_string('loss_type', 'cross_entropy', 'cross_entropy, spread or margin')
flags.DEFINE_boolean('add_recon_loss', False, 'To add reconstruction loss')
flags.DEFINE_boolean('L2_reg', False, 'Adds L2-regularization to all the network weights')
flags.DEFINE_float('lmbda', 5e-04, 'L2-regularization coefficient')

# Training logs
flags.DEFINE_integer('max_step', 100000, '# of step for training (only for mnist)')
flags.DEFINE_integer('max_epoch', 1000, '# of step for training (only for nodule data)')
flags.DEFINE_integer('SUMMARY_FREQ', 100, 'Number of step to save summary')

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
flags.DEFINE_float('percent', 1, 'Percentage of training data to use')
flags.DEFINE_boolean('one_hot', True, 'one-hot-encodes the labels (set to False if it is already one-hot-encoded)')
flags.DEFINE_boolean('normalize', True, 'Normalizes the data (set to False if it is already normalized)')
flags.DEFINE_boolean('data_augment', True, 'Adds augmentation to data')
flags.DEFINE_integer('max_angle', 180, 'Maximum rotation angle along each axis; when applying augmentation')
flags.DEFINE_integer('height', 32, 'Network input height size')
flags.DEFINE_integer('width', 32, 'Network input width size')
flags.DEFINE_integer('depth', 32, 'Network input depth size (in the case of 3D input images)')
flags.DEFINE_integer('channel', 1, 'Network input channel size')

# Directories
flags.DEFINE_string('run_name', '01', 'Run name')
flags.DEFINE_string('logdir', './Results/log_dir/', 'Logs directory')
flags.DEFINE_string('modeldir', './Results/model_dir/', 'Saved models directory')
flags.DEFINE_string('model_name', 'model', 'Model file name')

# CapsNet architecture
flags.DEFINE_integer('iter_routing', 3, 'Number of dynamic routing iterations')
flags.DEFINE_integer('prim_caps_dim', 256, 'Dimension of the PrimaryCapsules')
flags.DEFINE_integer('digit_caps_dim', 16, 'Dimension of the DigitCapsules')
flags.DEFINE_integer('h1', 512, 'Number of hidden units of the first FC layer of the reconstruction network')
flags.DEFINE_integer('h2', 1024, 'Number of hidden units of the second FC layer of the reconstruction network')

# cnn architectures
flags.DEFINE_float('dropout_rate', 0.2, 'Drop-out rate of the CNN models')

args = tf.app.flags.FLAGS
