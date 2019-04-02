import sys
import numpy as np
import tensorflow as tf
import cv2
import scipy.io as sio
import pose_utils as pu
import os
import os.path
import glob
import time
import scipy
import scipy.io as sio
import ST_model_nonTrainable_AlexNetOnFaces as Pose_model
import utils_3DMM
import csv
import argparse

# sys.path.append('./ResNet')
from ThreeDMM_shape import ResNet_101 as resnet101_shape
from ThreeDMM_expr import ResNet_101 as resnet101_expr


class m4_3DMM:
    def __init__(self, cfg):
        self.cfg = cfg

        # Get training image/labels mean/std for pose CNN
        try:
            file = np.load(self.cfg.train_imgs_mean_file_path, )
            self.train_mean_vec = file["train_mean_vec"]  # [0,1]
            print('Load ' + self.cfg.train_imgs_mean_file_path + ' successful....')
        except:
            raise Exception('Load ' + self.cfg.train_imgs_mean_file_path + ' failed....')
        del file

        try:
            file = np.load(self.cfg.train_labels_mean_std_file_path)
            self.mean_labels = file["mean_labels"]
            self.std_labels = file["std_labels"]
            print('Load ' + self.cfg.train_labels_mean_std_file_path + ' successful....')
        except:
            raise Exception('Load ' + self.cfg.train_labels_mean_std_file_path + ' failed....')
        del file

        try:
            # Get training image mean for Shape CNN
            mean_image_shape = np.load(self.cfg.ThreeDMM_shape_mean_file_path)  # 3 x 224 x 224
            self.mean_image_shape = np.transpose(mean_image_shape, [1, 2, 0])  # 224 x 224 x 3, [0,255]
            print('Load ' + self.cfg.ThreeDMM_shape_mean_file_path +' successful....')
        except:
            raise Exception('Load ' + self.cfg.ThreeDMM_shape_mean_file_path +' failed....')

        # 程序中没有用到
        # try:
        #     # Get training image mean for Expression CNN
        #     mean_image_exp = np.load('./Expression_Model/3DMM_expr_mean.npy')  # 3 x 224 x 224
        #     self.mean_image_exp = np.transpose(mean_image_exp, [1, 2, 0])  # 224 x 224 x 3, [0,255]
        #     print('Load ' + self.cfg.ThreeDMM_shape_mean_file_path + ' successful....')
        # except:
        #     raise Exception('Load ' + self.cfg.ThreeDMM_shape_mean_file_path + ' failed....')

    def extract_PSE_feats(self, x, reuse=False):
        '''
        :param x: x format is RGB and is value range is [-1,1].
        :return: fc1ls: shape, fc1le: expression, pose_model.preds_unNormalized: pose
        '''
        x = tf.image.resize_images(x, [227, 227])

        # x is RGB and is value range is [-1,1].
        # first we need to change RGB to BGR;
        batch_, height_, width_, nc = x.get_shape().as_list()
        R = tf.reshape(x[:, :, :, 0], [batch_, height_, width_, 1])
        G = tf.reshape(x[:, :, :, 1], [batch_, height_, width_, 1])
        B = tf.reshape(x[:, :, :, 2], [batch_, height_, width_, 1])
        x = tf.concat([B, G, R], axis=3)
        # second change range [-1,1] to [0,255]
        x = (x + 1.0) * 127.5

        ###################
        # Face Pose-Net
        ###################
        try:
            net_data = np.load(self.cfg.PAM_frontal_ALexNet_file_path, encoding="latin1").item()
            pose_labels = np.zeros([self.cfg.batch_size, 6])
            print('Load ' + self.cfg.PAM_frontal_ALexNet_file_path+ ' successful....')
        except:
            raise Exception('Load ' + self.cfg.PAM_frontal_ALexNet_file_path+ ' failed....')
        x1 = tf.image.resize_bilinear(x, tf.constant([227, 227], dtype=tf.int32))

        # Image normalization
        x1 = x1 / 255.  # from [0,255] to [0,1]
        # subtract training mean
        mean = tf.reshape(self.train_mean_vec, [1, 1, 1, 3])
        mean = tf.cast(mean, 'float32')
        x1 = x1 - mean

        pose_model = Pose_model.Pose_Estimation(x1, pose_labels, 'valid', 0, 1, 1, 0.01, net_data, self.cfg.batch_size,
                                                self.mean_labels, self.std_labels)
        pose_model._build_graph(reuse=reuse)
        self.pose = pose_model.preds_unNormalized
        del net_data

        ###################
        # Shape CNN
        ###################
        x2 = tf.image.resize_bilinear(x, tf.constant([224, 224], dtype=tf.int32))
        x2 = tf.cast(x2, 'float32')
        x2 = tf.reshape(x2, [self.cfg.batch_size, 224, 224, 3])

        # Image normalization
        mean = tf.reshape(self.mean_image_shape, [1, 224, 224, 3])
        mean = tf.cast(mean, 'float32')
        x2 = x2 - mean

        with tf.variable_scope('shapeCNN', reuse=reuse):
            net_shape = resnet101_shape({'input': x2}, trainable=True)
            pool5 = net_shape.layers['pool5']
            pool5 = tf.squeeze(pool5)
            pool5 = tf.reshape(pool5, [self.cfg.batch_size, -1])
            try:
                npzfile = np.load(self.cfg.ShapeNet_fc_weights_file_path)
                print('Load ' + self.cfg.ShapeNet_fc_weights_file_path + ' successful....')
            except:
                raise Exception('Load ' + self.cfg.ShapeNet_fc_weights_file_path + ' failed....')

            ini_weights_shape = npzfile['ini_weights_shape']
            ini_biases_shape = npzfile['ini_biases_shape']
            with tf.variable_scope('shapeCNN_fc1'):
                # fc1ws = tf.Variable(tf.reshape(ini_weights_shape, [2048, -1]), trainable=True, name='weights')
                # fc1bs = tf.Variable(tf.reshape(ini_biases_shape, [-1]), trainable=True, name='biases')
                fc1ws = tf.get_variable(initializer=tf.reshape(ini_weights_shape, [2048, -1]), trainable=True, name='weights')
                fc1bs = tf.get_variable(initializer=tf.reshape(ini_biases_shape, [-1]), trainable=True, name='biases')
                self.fc1ls = tf.nn.bias_add(tf.matmul(pool5, fc1ws), fc1bs)

        ###################
        # Expression CNN
        ###################
        with tf.variable_scope('exprCNN', reuse=reuse):
            net_expr = resnet101_expr({'input': x2}, trainable=True)
            pool5 = net_expr.layers['pool5']
            pool5 = tf.squeeze(pool5)
            pool5 = tf.reshape(pool5, [self.cfg.batch_size, -1])

            try:
                npzfile = np.load(self.cfg.ExpNet_fc_weights_file_path)
                ini_weights_expr = npzfile['ini_weights_expr']
                ini_biases_expr = npzfile['ini_biases_expr']
                print('Load ' + self.cfg.ExpNet_fc_weights_file_path + '  successful....')
            except:
                raise Exception('Load ' + self.cfg.ExpNet_fc_weights_file_path + '  failed....')
            # time.sleep(30)

            with tf.variable_scope('exprCNN_fc1'):
                # fc1we = tf.Variable(tf.reshape(ini_weights_expr, [2048, 29]), trainable=True, name='weights')
                # fc1be = tf.Variable(tf.reshape(ini_biases_expr, [29]), trainable=True, name='biases')
                fc1we = tf.get_variable(initializer=tf.reshape(ini_weights_expr, [2048, 29]), trainable=True, name='weights')
                fc1be = tf.get_variable(initializer=tf.reshape(ini_biases_expr, [29]), trainable=True, name='biases')
                self.fc1le = tf.nn.bias_add(tf.matmul(pool5, fc1we), fc1be)



        # return fc1ls, fc1le, pose_model.preds_unNormalized



