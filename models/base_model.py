import tensorflow as tf
import os
import numpy as np
from loss_ops import margin_loss, spread_loss, cross_entropy
from sklearn.metrics import confusion_matrix
from DataLoader import DataLoader


class BaseModel(object):
    def __init__(self, sess, conf):
        self.sess = sess
        self.conf = conf
        self.summary_list = []
        self.input_shape = [conf.batch_size, conf.height, conf.width, conf.depth, conf.channel]
        self.output_shape = [self.conf.batch_size, self.conf.num_cls]
        self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        self.create_placeholders()

    def create_placeholders(self):
        with tf.name_scope('Input'):
            self.x = tf.placeholder(tf.float32, self.input_shape, name='input')
            self.y = tf.placeholder(tf.float32, self.output_shape, name='annotation')
            self.is_training = tf.placeholder_with_default(False, shape=(), name="is_training")

    def loss_func(self):
        with tf.variable_scope('Loss'):
            if self.conf.loss_type == 'margin':
                loss = margin_loss(self.y, self.v_length, self.conf)
                self.summary_list.append(tf.summary.scalar('margin', loss))
            elif self.conf.loss_type == 'spread':
                self.generate_margin()
                loss = spread_loss(self.y, self.act, self.margin, 'spread_loss')
                self.summary_list.append(tf.summary.scalar('spread_loss', loss))
            elif self.conf.loss_type == 'cross_entropy':
                loss = cross_entropy(self.y, self.logits)
                tf.summary.scalar('cross_entropy', loss)
            if self.conf.L2_reg:
                with tf.name_scope('l2_loss'):
                    l2_loss = tf.reduce_sum(self.conf.lmbda * tf.stack([tf.nn.l2_loss(v)
                                                                        for v in tf.get_collection('weights')]))
                    loss += l2_loss
                self.summary_list.append(tf.summary.scalar('l2_loss', l2_loss))
            if self.conf.add_recon_loss:
                with tf.variable_scope('Reconstruction_Loss'):
                    squared = tf.square(self.decoder_output - self.x)
                    self.recon_err = tf.reduce_mean(squared)
                    self.total_loss = loss + self.conf.alpha * self.recon_err
                    self.summary_list.append(tf.summary.scalar('reconstruction_loss', self.recon_err))
                    self.summary_list.append(tf.summary.image('reconstructed', self.decoder_output))
                    self.summary_list.append(tf.summary.image('original', self.x))
            else:
                self.total_loss = loss
            self.mean_loss, self.mean_loss_op = tf.metrics.mean(self.total_loss)

    def accuracy_func(self):
        with tf.variable_scope('Accuracy'):
            correct_prediction = tf.equal(tf.to_int32(tf.argmax(self.y, axis=1)), self.y_pred)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self.mean_accuracy, self.mean_accuracy_op = tf.metrics.mean(accuracy)

    def generate_margin(self):
        # margin schedule
        # margin increase from 0.2 to 0.9 after margin_schedule_epoch_achieve_max
        NUM_STEPS_PER_EPOCH = int(self.conf.N / self.conf.batch_size)
        margin_schedule_epoch_achieve_max = 10.0
        self.margin = tf.train.piecewise_constant(tf.cast(self.global_step, dtype=tf.int32),
                                                  boundaries=[int(NUM_STEPS_PER_EPOCH *
                                                                  margin_schedule_epoch_achieve_max * x / 7)
                                                              for x in xrange(1, 8)],
                                                  values=[x / 10.0 for x in range(2, 10)])

    def configure_network(self):
        self.loss_func()
        self.accuracy_func()

        with tf.name_scope('Optimizer'):
            with tf.name_scope('Learning_rate_decay'):
                learning_rate = tf.train.exponential_decay(self.conf.init_lr,
                                                           self.global_step,
                                                           decay_steps=3000,
                                                           decay_rate=0.97,
                                                           staircase=True)
                self.learning_rate = tf.maximum(learning_rate, self.conf.lr_min)
            self.summary_list.append(tf.summary.scalar('learning_rate', self.learning_rate))
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            grads = optimizer.compute_gradients(self.total_loss)
            self.train_op = optimizer.apply_gradients(grads, global_step=self.global_step)
        self.sess.run(tf.global_variables_initializer())
        trainable_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(var_list=trainable_vars, max_to_keep=1000)
        self.train_writer = tf.summary.FileWriter(self.conf.logdir + self.conf.run_name + '/train/', self.sess.graph)
        self.valid_writer = tf.summary.FileWriter(self.conf.logdir + self.conf.run_name + '/valid/')
        self.configure_summary()
        print('*' * 50)
        print('Total number of trainable parameters: {}'.
              format(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))
        print('*' * 50)

    def configure_summary(self):
        summary_list = [tf.summary.scalar('Loss/total_loss', self.mean_loss),
                        tf.summary.scalar('Accuracy/average_accuracy', self.mean_accuracy)] + self.summary_list
        self.merged_summary = tf.summary.merge(summary_list)

    def save_summary(self, summary, step, mode):
        if mode == 'train':
            self.train_writer.add_summary(summary, step)
        elif mode == 'valid':
            self.valid_writer.add_summary(summary, step)
        self.sess.run(tf.local_variables_initializer())

    def train(self):
        self.sess.run(tf.local_variables_initializer())
        self.best_validation_accuracy = 0
        self.data_reader = DataLoader(self.conf)
        self.data_reader.get_data(mode='train')
        self.data_reader.get_data(mode='valid')
        self.train_loop()

    def train_loop(self):
        if self.conf.reload_step > 0:
            self.reload(self.conf.reload_step)
            print('*' * 50)
            print('----> Continue Training from step #{}'.format(self.conf.reload_step))
            print('*' * 50)
        else:
            print('*' * 50)
            print('----> Start Training')
            print('*' * 50)
        self.num_val_batch = self.data_reader.count_num_batch(self.conf.batch_size, mode='valid')
        if self.conf.epoch_based:
            self.num_train_batch = self.data_reader.count_num_batch(self.conf.batch_size, mode='train')
            for epoch in range(self.conf.max_epoch):
                self.data_reader.randomize()
                for train_step in range(self.num_train_batch):
                    glob_step = epoch * self.num_train_batch + train_step
                    start = train_step * self.conf.batch_size
                    end = (train_step + 1) * self.conf.batch_size
                    x_batch, y_batch = self.data_reader.next_batch(start, end, mode='train')
                    feed_dict = {self.x: x_batch, self.y: y_batch, self.is_training: True}
                    if train_step % self.conf.SUMMARY_FREQ == 0:
                        _, _, _, summary = self.sess.run([self.train_op,
                                                          self.mean_loss_op,
                                                          self.mean_accuracy_op,
                                                          self.merged_summary], feed_dict=feed_dict)
                        loss, acc = self.sess.run([self.mean_loss, self.mean_accuracy])
                        self.save_summary(summary, glob_step + self.conf.reload_step, mode='train')
                        print('step: {0:<6}, train_loss= {1:.4f}, train_acc={2:.01%}'.format(train_step, loss, acc))
                    else:
                        self.sess.run([self.train_op, self.mean_loss_op, self.mean_accuracy_op], feed_dict=feed_dict)
                self.evaluate(glob_step)
        else:
            self.data_reader.randomize()
            for train_step in range(1, self.conf.max_step + 1):
                # print(train_step)
                if train_step % self.conf.SUMMARY_FREQ == 0:
                    x_batch, y_batch = self.data_reader.next_batch()
                    feed_dict = {self.x: x_batch, self.y: y_batch, self.is_training: True}
                    _, _, _, summary = self.sess.run([self.train_op,
                                                      self.mean_loss_op,
                                                      self.mean_accuracy_op,
                                                      self.merged_summary], feed_dict=feed_dict)
                    loss, acc = self.sess.run([self.mean_loss, self.mean_accuracy])
                    self.save_summary(summary, train_step + self.conf.reload_step, mode='train')
                    print('step: {0:<6}, train_loss= {1:.4f}, train_acc={2:.01%}'.format(train_step, loss, acc))
                else:
                    x_batch, y_batch = self.data_reader.next_batch()
                    feed_dict = {self.x: x_batch, self.y: y_batch, self.is_training: True}
                    self.sess.run([self.train_op, self.mean_loss_op, self.mean_accuracy_op], feed_dict=feed_dict)
                if train_step % self.conf.VAL_FREQ == 0:
                    self.evaluate(train_step)

    def evaluate(self, train_step):
        self.sess.run(tf.local_variables_initializer())
        y_pred = np.zeros((self.data_reader.y_valid.shape[0]))
        for step in range(self.num_val_batch):
            start = step * self.conf.batch_size
            end = (step + 1) * self.conf.batch_size
            x_val, y_val = self.data_reader.next_batch(start, end, mode='valid')
            feed_dict = {self.x: x_val, self.y: y_val, self.is_training: False}
            yp, _, _ = self.sess.run([self.y_pred, self.mean_loss_op, self.mean_accuracy_op], feed_dict=feed_dict)
            y_pred[start:end] = yp
        summary_valid = self.sess.run(self.merged_summary, feed_dict=feed_dict)
        valid_loss, valid_acc = self.sess.run([self.mean_loss, self.mean_accuracy])
        self.save_summary(summary_valid, train_step + self.conf.reload_step, mode='valid')
        if valid_acc > self.best_validation_accuracy:
            self.best_validation_accuracy = valid_acc
            improved_str = '(improved)'
            self.save(train_step + self.conf.reload_step)
        else:
            improved_str = ''

        print('-' * 25 + 'Validation' + '-' * 25)
        print('After {0} training step: val_loss= {1:.4f}, val_acc={2:.01%}{3}'
              .format(train_step, valid_loss, valid_acc, improved_str))
        print(confusion_matrix(np.argmax(self.data_reader.y_valid, axis=1), y_pred))
        print('-' * 60)

    def test(self, step_num):
        self.sess.run(tf.local_variables_initializer())
        self.reload(step_num)
        self.data_reader = DataLoader(self.conf)
        self.data_reader.get_data(mode='test')
        self.num_test_batch = self.data_reader.count_num_batch(self.conf.batch_size, mode='test')
        self.is_train = False
        self.sess.run(tf.local_variables_initializer())
        y_pred = np.zeros((self.data_reader.y_test.shape[0]))
        img_recon = np.zeros((self.data_reader.y_test.shape[0], self.conf.height*self.conf.width))
        for step in range(self.num_test_batch):
            start = step * self.conf.batch_size
            end = (step + 1) * self.conf.batch_size
            x_test, y_test = self.data_reader.next_batch(start, end, mode='test')
            feed_dict = {self.x: x_test, self.y: y_test, self.is_training: False}
            yp, _, _, img = self.sess.run([self.y_pred, self.mean_loss_op, self.mean_accuracy_op, self.decoder_output],
                                     feed_dict=feed_dict)
            y_pred[start:end] = yp
            img_recon[start:end] = img
        test_loss, test_acc = self.sess.run([self.mean_loss, self.mean_accuracy])
        print('-' * 18 + 'Test Completed' + '-' * 18)
        print('test_loss= {0:.4f}, test_acc={1:.01%}'.format(test_loss, test_acc))
        print(confusion_matrix(np.argmax(self.data_reader.y_test, axis=1), y_pred))
        print('-' * 50)

    def save(self, step):
        print('----> Saving the model at step #{0}'.format(step))
        checkpoint_path = os.path.join(self.conf.modeldir + self.conf.run_name, self.conf.model_name)
        self.saver.save(self.sess, checkpoint_path, global_step=step)

    def reload(self, step):
        checkpoint_path = os.path.join(self.conf.modeldir + self.conf.run_name, self.conf.model_name)
        model_path = checkpoint_path + '-' + str(step)
        if not os.path.exists(model_path + '.meta'):
            print('----> No such checkpoint found', model_path)
            return
        print('----> Restoring the CNN model...')
        self.saver.restore(self.sess, model_path)
        print('----> CNN Model successfully restored')
