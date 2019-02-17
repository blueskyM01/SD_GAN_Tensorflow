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

listf = np.loadtxt('/media/yang/F/DataSet/Face/Label/lfw-deepfunneled.txt',dtype=str)
print(listf.shape)