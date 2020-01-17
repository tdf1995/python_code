

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import os
import re
import numpy as np
import time
from tensorflow.python.platform import gfile

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('test_batch_size', 1000,
                            "Number of images to test process in a batch")
tf.app.flags.DEFINE_integer('batch_size', 1,
                            "Number of images to process in a batch.")
tf.app.flags.DEFINE_string('dataset_dir', '',
                           "Path to the tfrecord directory")
tf.app.flags.DEFINE_string('train_dir', r'D:\OCR\traindir',
                           "Directory where to write event logs and checkpoint.")
tf.app.flags.DEFINE_integer('image_height', 200,
                            "")
tf.app.flags.DEFINE_integer('image_width', 625,
                            "")
tf.app.flags.DEFINE_integer('num_class', 127,
                            "")
tf.app.flags.DEFINE_integer('num_train_pic', 200000,
                            "")
tf.app.flags.DEFINE_integer('num_test_pic', 6000,
                            "")
tf.app.flags.DEFINE_integer('Initial_Learning_Rate', 0.001,
                            "")
tf.app.flags.DEFINE_integer('num_test_pic', 6000,
                            "")