import random
import scipy
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import h5py


class DataLoader(object):

    def __init__(self, cfg):
        self.cfg = cfg
        self.augment = cfg.data_augment
        self.max_angle = cfg.max_angle
        self.batch_size = cfg.batch_size
        if cfg.percent == 1:
            self.data_path = './data/data_' + str(cfg.height) + '.h5'
        else:
            self.data_path = './data/data_' + str(cfg.height) + '_' + str(cfg.percent) + '.h5'

    def get_data(self, mode='train'):
        h5f = h5py.File(self.data_path, 'r')
        if mode == 'train':
            x_train = h5f['X_train'][:]
            y_train = h5f['Y_train'][:]
            if self.cfg.flip:
                self.x_train, self.y_train = flip_aug(x_train, y_train)
            else:
                self.x_train, self.y_train = x_train, y_train
        elif mode == 'valid':
            self.x_valid = h5f['X_test'][:]
            self.y_valid = h5f['Y_test'][:]
        elif mode == 'test':
            self.x_test = h5f['X_test'][:]
            self.y_test = h5f['Y_test'][:]
        h5f.close()

    def next_batch(self, start=None, end=None, mode='train'):
        if mode == 'train':
            x = self.x_train[start:end]
            y = self.y_train[start:end]
            if self.augment:
                x = random_rotation_2d(x, self.cfg.max_angle)
        elif mode == 'valid':
            x = self.x_valid[start:end]
            y = self.y_valid[start:end]
        elif mode == 'test':
            x = self.x_test[start:end]
            y = self.y_test[start:end]
        return x, y

    def count_num_batch(self, batch_size, mode='train'):
        if mode == 'train':
            num_batch = int(self.y_train.shape[0] / batch_size)
        elif mode == 'valid':
            num_batch = int(self.y_valid.shape[0] / batch_size)
        elif mode == 'test':
            num_batch = int(self.y_test.shape[0] / batch_size)
        return num_batch

    def randomize(self):
        """ Randomizes the order of data samples and their corresponding labels"""
        permutation = np.random.permutation(self.y_train.shape[0])
        self.x_train = self.x_train[permutation, :, :, :]
        self.y_train = self.y_train[permutation]

    def get_stats(self):
        h5f = h5py.File(self.data_path, 'r')
        x_train = h5f['X_train'][:]
        h5f.close()
        self.mean = np.mean(x_train, axis=0)
        self.std = np.std(x_train, axis=0)


def random_rotation_2d(batch, max_angle):
    """ Randomly rotate an image by a random angle (-max_angle, max_angle).
    Arguments:
    max_angle: `float`. The maximum rotation angle.
    Returns:
    batch of rotated 2D images
    """
    size = batch.shape
    batch = np.squeeze(batch)
    batch_rot = np.zeros(batch.shape)
    for i in range(batch.shape[0]):
        if bool(random.getrandbits(1)):
            image = np.squeeze(batch[i])
            angle = random.uniform(-max_angle, max_angle)
            batch_rot[i] = scipy.ndimage.interpolation.rotate(image, angle, mode='nearest', reshape=False)
        else:
            batch_rot[i] = batch[i]
    return batch_rot.reshape(size)


def flip_aug(x, y):
    x_90, x_180, x_270 = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
    for i in range(x.shape[0]):
        x_90[i] = scipy.ndimage.interpolation.rotate(np.squeeze(x[i]), 90).reshape(x.shape[1:])
        x_180[i] = scipy.ndimage.interpolation.rotate(np.squeeze(x[i]), 180).reshape(x.shape[1:])
        x_270[i] = scipy.ndimage.interpolation.rotate(np.squeeze(x[i]), 270).reshape(x.shape[1:])
    x_new = np.concatenate((x, x_90, x_180, x_270), axis=0)
    y_new = np.concatenate((y, y, y, y), axis=0)
    return x_new, y_new
