import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import utils

from PIL import Image

data_dir = "./data/"
train_dir = "trainset/JPEGImages"
test_dir = "testset/JPEGImages"

image_size = (224, 224)


class VOC(object):
    def __init__(self):
        self.records_trainval_path = "voc_trainval_records.pkl"
        self.records_test_path = "voc_test_records.pkl"

        self.image_name_to_cls_name_train = None
        self.image_name_to_cls_id_train = None
        self.cls_name_to_cls_id_train = None
        self.cls_id_to_cls_name_train = None

        self.image_name_to_cls_name_test = None
        self.image_name_to_cls_id_test = None
        self.cls_name_to_cls_id_test = None
        self.cls_id_to_cls_name_test = None

        self.load_records()

    def load_records(self, train=True):

        records_trainval_path = os.path.join(data_dir, self.records_trainval_path)
        with open(records_trainval_path, mode='rb') as file:
            records_trainval = pickle.load(file)
            self.image_name_to_cls_name_train = records_trainval[0]
            self.image_name_to_cls_id_train = records_trainval[1]
            self.cls_name_to_cls_id_train = records_trainval[3]
            self.cls_id_to_cls_name_train = records_trainval[2]

        records_test_path = os.path.join(data_dir, self.records_test_path)
        with open(records_test_path, mode='rb') as file:
            records_test = pickle.load(file)
            self.image_name_to_cls_name_test = records_test[0]
            self.image_name_to_cls_id_test = records_test[1]
            self.cls_name_to_cls_id_test = records_test[3]
            self.cls_id_to_cls_name_test = records_test[2]

    def show_image(self, list=[], train=True):
        if train:
            dir = data_dir + train_dir

            for index in list:
                filename = self.image_name_to_cls_name_train[index][0] + '.jpg'
                cls_true_name = self.image_name_to_cls_name_train[index][1]

                # Path for the image-file.
                path = os.path.join(dir, filename)

                # Load the image and plot it.
                img = load_image(path)
                plot_images_(img, filename, cls_true_name)

        else:
            dir = data_dir + test_dir

            for index in list:
                filename = self.image_name_to_cls_name_test[index][0] + '.jpg'
                cls_true_name = self.image_name_to_cls_name_test[index][1]

                # Path for the image-file.
                path = os.path.join(dir, filename)

                # Load the image and plot it.
                img = load_image(path)
                plot_images_(img, filename, cls_true_name, train)

    def pre_posscess_data(self, train=True):
        if train:
            image_num = len(self.image_name_to_cls_id_train[:])
            VOCtrainval_path = data_dir + "voc_trainval_data_uint8.pkl"
        else:
            image_num = len(self.image_name_to_cls_id_test[:])
            VOCtrainval_path = data_dir + "voc_test_data_uint8.pkl"
        shape = [image_num, 224, 224, 3]
        x_train = np.zeros(shape=shape, dtype=np.uint8)
        y_train = np.zeros(image_num, dtype=np.uint8)
        x_train_names = []
        for i in range(image_num):
            print(i)
            if train:
                dir = data_dir + train_dir
                filename = self.image_name_to_cls_id_train[i][0] + '.jpg'
                cls_true_id = self.image_name_to_cls_id_train[i][1]
            else:
                dir = data_dir + test_dir
                filename = self.image_name_to_cls_id_test[i][0] + '.jpg'
                cls_true_id = self.image_name_to_cls_id_test[i][1]

            path = os.path.join(dir, filename)
            img = load_image(path, image_size)

            x_train_names.append(filename)
            x_train[i] = img
            y_train[i] = cls_true_id

        obj = (x_train_names, x_train, y_train)
        print(VOCtrainval_path)
        with open(VOCtrainval_path, mode='wb')as file:
            pickle.dump(obj, file, protocol=4)

    def load_data(self, train=True):
        if train:
            data_path = data_dir + "voc_trainval_data_uint8.pkl"

        else:
            data_path = data_dir + "voc_test_data_uint8.pkl"

        with open(data_path, mode='rb')as file:
            data = pickle.load(file)

        x_name = data[0]
        x_ = data[1]
        y_ = data[2]

        return x_name, x_, y_

    def pre_posscess_data_tf(self, train=True):
        if train:
            image_num = len(self.image_name_to_cls_id_train[:])
            VOCtrainval_path = data_dir + "voc_trainval_data_uint8_tf.pkl"
        else:
            image_num = len(self.image_name_to_cls_id_test[:])
            VOCtrainval_path = data_dir + "voc_test_data_uint8_tf.pkl"
        shape = [image_num, 224, 224, 3]
        x_train = np.zeros(shape=shape, dtype=np.uint8)
        y_train = np.zeros(image_num, dtype=np.uint8)
        x_train_names = []
        for i in range(image_num):
            print(i)
            if train:
                dir = data_dir + train_dir
                filename = self.image_name_to_cls_id_train[i][0] + '.jpg'
                cls_true_id = self.image_name_to_cls_id_train[i][1]
            else:
                dir = data_dir + test_dir
                filename = self.image_name_to_cls_id_test[i][0] + '.jpg'
                cls_true_id = self.image_name_to_cls_id_test[i][1]

            path = os.path.join(dir, filename)
            img = utils.load_image(path)

            x_train_names.append(filename)
            x_train[i] = img
            y_train[i] = cls_true_id

        obj = (x_train_names, x_train, y_train)
        print(VOCtrainval_path)
        with open(VOCtrainval_path, mode='wb')as file:
            pickle.dump(obj, file, protocol=4)

    def load_data_tf(self, train=True):
        if train:
            data_path = data_dir + "voc_trainval_data_uint8_tf.pkl"

        else:
            data_path = data_dir + "voc_test_data_uint8_tf.pkl"

        with open(data_path, mode='rb')as file:
            data = pickle.load(file)

        x_name = data[0]
        x_ = data[1]
        y_ = data[2]

        return x_name, x_, y_


voc = VOC()


def load_image(path, size=None):
    img = Image.open(path)

    if not size is None:
        img = img.resize(size=size, resample=Image.LANCZOS)

    img = np.array(img)

    # img = img / 255.0

    if (len(img.shape) == 2):
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

    return img


def plot_images_(img, filename, cls_true_name, cls_pred_name=None, pred_prob=None, train=True):
    plt.imshow(img)
    if train:

        if (type(cls_true_name) == np.uint8 or type(cls_true_name) == np.int64):
            cls_true_name = voc.cls_id_to_cls_name_train[cls_true_name]
            x_label = "Shape:{2},  Filename: {0},  Ture: {1}".format(filename, cls_true_name, img.shape)
        else:
            x_label = "Shape:{2},  Filename: {0},  Ture: {1}".format(filename, cls_true_name, img.shape)
    else:

        if (type(cls_true_name) == np.uint8 or type(cls_true_name) == np.int64):
            cls_true_name = voc.cls_id_to_cls_name_test[cls_true_name]
            x_label = "Shape:{2},  Filename: {0},  Ture: {1}".format(filename, cls_true_name, img.shape)
        else:
            x_label = "Shape:{2},  Filename: {0},  Ture: {1}".format(filename, cls_true_name, img.shape)
    plt.xlabel(x_label)
    plt.show()


def plot_images(img, filename, cls_true_name, cls_pred_name=None, pred_prob=None, trainset=True):
    plt.imshow(img)
    if trainset:
        cls_true_name = voc.cls_id_to_cls_name_train[cls_true_name]

        if cls_pred_name == None:
            x_label = "Filename: {0}, Ture: {1}".format(filename, cls_true_name)
        else:
            cls_pred_name = voc.cls_id_to_cls_name_train[cls_pred_name]
            x_label = "Filename: {0}, Ture: {1},   Pred: {2}, Prob: {3}".format(filename, cls_true_name, cls_pred_name,
                                                                                pred_prob)
    else:
        cls_true_name = voc.cls_id_to_cls_name_test[cls_true_name]

        if cls_pred_name == None:
            x_label = "Filename: {0}, Ture: {1}".format(filename, cls_true_name)
        else:
            cls_pred_name = voc.cls_id_to_cls_name_test[cls_pred_name]
            x_label = "Filename: {0}, Ture: {1},   Pred: {2}, Prob: {3}".format(filename, cls_true_name, cls_pred_name,
                                                                                pred_prob)

    plt.xlabel(x_label)
    plt.show()


def normalize(X_train, X_test):
    # this function normalize inputs for zero mean and unit variance
    # it is used when training a model.
    # Input: training set and test set
    # Output: normalized training set and test set according to the trianing set statistics.
    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    X_train = (X_train - mean) / (std + 1e-7)
    X_test = (X_test - mean) / (std + 1e-7)
    return X_train, X_test
