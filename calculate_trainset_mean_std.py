
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    FILE NAME: calculate_trainset_mean_std.py
    AUTHOR: Michalis Meyer
    DATE CREATED: 28.06.2020
    DATE LAST MODIFIED: 25.03.2020
    PYTHON VERSION: 3.6.3
    SCRIPT PURPOSE: The script calculates the image per channel mean and standard
    deviation in the training set, do not calculate the statistics on the whole
    dataset, as per here http://cs231n.github.io/neural-networks-2/#datapre
    
    Source: https://gist.github.com/jdhao/9a86d4b9e4f79c5330d54de991461fd6#file-calculate_trainset_mean_std-py
"""

import numpy as np
from os import listdir
from os.path import join, isdir
from glob import glob
import cv2
import timeit

# number of channels of the dataset image, 3 for color jpg, 1 for grayscale img
# you need to change it to reflect your dataset
CHANNEL_NUM = 3


def cal_dir_stat(root):
    cls_dirs = [d for d in listdir(root) if isdir(join(root, d))]
    pixel_num = 0 # store all pixel number in the dataset
    channel_sum = np.zeros(CHANNEL_NUM)
    channel_sum_squared = np.zeros(CHANNEL_NUM)

    for idx, d in enumerate(cls_dirs):
        print("#{} class".format(idx))
        im_pths = glob(join(root, d, "*.jpg"))

        for path in im_pths:
            im = cv2.imread(path) # image in M*N*CHANNEL_NUM shape, channel in BGR order
            im = im/255.0
            pixel_num += (im.size/CHANNEL_NUM)
            channel_sum += np.sum(im, axis=(0, 1))
            channel_sum_squared += np.sum(np.square(im), axis=(0, 1))

    bgr_mean = channel_sum / pixel_num
    bgr_std = np.sqrt(channel_sum_squared / pixel_num - np.square(bgr_mean))
    
    # change the format from bgr to rgb
    rgb_mean = list(bgr_mean)[::-1]
    rgb_std = list(bgr_std)[::-1]
    
    return rgb_mean, rgb_std

# The script assumes that under train_root, there are separate directories for each class
# of training images.
train_root = "C:/Users/username/path_to_project/flowers/train"
start = timeit.default_timer()
mean, std = cal_dir_stat(train_root)
end = timeit.default_timer()
print("elapsed time: {}".format(end-start))
print("mean:{}\nstd:{}".format(mean, std))


