import random
import scipy
import numpy as np
import h5py
import scipy.ndimage


class DataLoader(object):

    def __init__(self, cfg):
        self.cfg = cfg
        self.mean = None
        self.std = None
        if cfg.percent == 1:
            self.data_path = './data/Lung_Nodule.h5'
        else:
            self.data_path = './data/Lung_Nodule_' + str(cfg.percent) + '.h5'

    def get_data(self, mode='train'):
        h5f = h5py.File(self.data_path, 'r')
        if mode == 'train':
            x_train = h5f['X_train'][:]
            y_train = h5f['Y_train'][:]
            self.x_train, self.y_train = self.prepare_data(x_train, y_train)
        elif mode == 'valid':
            x_valid = h5f['X_valid'][:]
            y_valid = h5f['Y_valid'][:]
            self.x_valid, self.y_valid = self.prepare_data(x_valid, y_valid)
        elif mode == 'test':
            x_test = h5f['X_valid'][:]
            y_test = h5f['Y_valid'][:]
            self.x_test, self.y_test = self.prepare_data(x_test, y_test)
        h5f.close()

    def prepare_data(self, x, y):
        if self.cfg.normalize:
            x = np.maximum(np.minimum(x, 4096.), 0.)
            try:
                _ = self.mean.shape
            except AttributeError:
                self.get_stats()
            x = (x - self.mean) / self.std
            x = x.reshape((-1, self.cfg.height, self.cfg.width, self.cfg.depth, self.cfg.channel)).astype(np.float32)
        if self.cfg.one_hot:
            y = (np.arange(self.cfg.num_cls) == y[:, None]).astype(np.float32)
        return x, y

    def next_batch(self, start=None, end=None, mode='train'):
        if mode == 'train':
            x = self.x_train[start:end]
            y = self.y_train[start:end]
            if self.cfg.data_augment:
                x = random_rotation_3d(x, self.cfg.max_angle)
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
        """ Randomizes the order of training data samples and their corresponding labels"""
        permutation = np.random.permutation(self.y_train.shape[0])
        self.x_train = self.x_train[permutation, :, :, :]
        self.y_train = self.y_train[permutation]

    def get_stats(self):
        """
        compute and store the mean and std of training samples (to be used for normalization)
        """
        h5f = h5py.File(self.data_path, 'r')
        x_train = np.maximum(np.minimum(h5f['X_train'][:], 4096.), 0.)
        h5f.close()
        self.mean = np.mean(x_train, axis=0)
        self.std = np.std(x_train, axis=0)


def random_rotation_3d(batch, max_angle):
    """
    Randomly rotate an image by a random angle (-max_angle, max_angle).
    :param batch: batch of images of shape (batch_size, height, width, depth, channel)
    :param max_angle: maximum rotation angle in degree
    :return: array of rotated batch of images of the same shape as 'batch'
    """
    size = batch.shape
    batch_rot = np.squeeze(batch)
    for i in range(batch.shape[0]):
        image = np.squeeze(batch[i])
        if bool(random.getrandbits(1)):  # rotate along x-axis
            angle = random.uniform(-max_angle, max_angle)
            image = scipy.ndimage.interpolation.rotate(image, angle, mode='nearest', axes=(1, 2), reshape=False)
        if bool(random.getrandbits(1)):  # rotate along y-axis
            angle = random.uniform(-max_angle, max_angle)
            image = scipy.ndimage.interpolation.rotate(image, angle, mode='nearest', axes=(0, 2), reshape=False)
        if bool(random.getrandbits(1)):  # rotate along z-axis
            angle = random.uniform(-max_angle, max_angle)
            image = scipy.ndimage.interpolation.rotate(image, angle, mode='nearest', axes=(0, 1), reshape=False)
        batch_rot[i] = image
    return batch_rot.reshape(size)
