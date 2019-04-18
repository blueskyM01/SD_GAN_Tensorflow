from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
import os
import datetime
from utils import *
from ops import *
from networks import *
import time
import ExpShapePoseNet as ESP
import scipy
import scipy.io as sio
import utils_3DMM
import argparse
import param
import time
import cv2


# listf = np.loadtxt('/media/yang/F/DataSet/Face/Label/lfw-deepfunneled.txt',dtype=str)
# print(listf.shape)
# image = cv2.imread('/media/yang/F/DataSet/Face/CASIA-WebFace_align/0000121/019.png')
# cv2.imshow('pic',image)
# cv2.waitKey(0)
#
# print('hhha')


def m4_np_one_change_to_np_two(np_1, np_2, steps):
    '''
    :param np_1:
    :param np_2:
    :param steps:
    :return:
    '''
    m4_distance = np_2 - np_1
    m4_step_distance = m4_distance / steps
    m4_intermediary_list = []
    for i in range(steps):
        np_1 = np_1 + m4_step_distance
        m4_intermediary_list.append(np_1)
    return m4_intermediary_list


np_1 = np.array([[1, 2, 3, 4],
                 [9, 4, 6, 9]], dtype=np.float32)

np_2 = np.array([[18, 12, 34, 44],
                 [94, 54, 61, 90]], dtype=np.float32)

cccc = m4_np_one_change_to_np_two(np_1, np_2, 10)

for i in cccc:
    print(i)

[-0.4, 0.4, -0.4, 0.4, -0.4, 0.4, 0.4, -0.4, 0.4, -0.4, -0.4, -0.4, -0.4, 0.4, -0.4, 0.4, 0.4, 0.4, 0.4, -0.4, 0.4,
 -0.4, 0.4, 0.4, -0.4, 0.4, -0.4, 0.4, 0.4]

[-0.4, -0.4, 0.4, -0.4, -0.4, 0.4, 0.4, 0.4, -0.4, -0.4, -0.4, -0.4, -0.4, 0.4, -0.4, 0.4, 0.4, -0.4, -0.4, -0.4, 0.4,
 0.4, -0.4, -0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
