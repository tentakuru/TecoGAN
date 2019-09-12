# uncompyle6 version 3.3.5
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.7.1 (v3.7.1:260ec2c36a, Oct 20 2018, 14:57:15) [MSC v.1915 64 bit (AMD64)]
# Embedded file name: lib\dataloader.py
import tensorflow as tf
from lib.ops import *
import cv2 as cv, collections, os, math, scipy.misc as sic, numpy as np
from scipy import signal

def inference_data_loader(FLAGS):
    filedir = FLAGS.input_dir_LR
    downSP = False
    if FLAGS.input_dir_LR is None or not os.path.exists(FLAGS.input_dir_LR):
        if FLAGS.input_dir_HR is None or not os.path.exists(FLAGS.input_dir_HR):
            raise ValueError('Input directory not found')
        filedir = FLAGS.input_dir_HR
        downSP = True
    image_list_LR_temp = os.listdir(filedir)
    image_list_LR_temp = [_ for _ in image_list_LR_temp if _.endswith('.png')]
    image_list_LR_temp = sorted(image_list_LR_temp)
    image_list_LR_temp.sort(key=(lambda f: int(''.join(list(filter(str.isdigit, f))) or -1)))
    image_list_LR = [os.path.join(filedir, _) for _ in image_list_LR_temp]

    def preprocess_test(name):
        im = cv.imread(name, 3).astype(np.float32)
        if downSP:
            icol_blur = cv.GaussianBlur(im, (0, 0), sigmaX=1.5)
            im = icol_blur[::4, ::4, :]
        im = im / 255.0
        return im

    image_LR = [preprocess_test(_) for _ in image_list_LR]
    image_list_LR = image_list_LR[5:0:-1] + image_list_LR
    image_LR = image_LR[5:0:-1] + image_LR
    Data = collections.namedtuple('Data', 'paths_LR, inputs')
    return Data(paths_LR=image_list_LR,
      inputs=image_LR)