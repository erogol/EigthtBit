# -*- coding: utf-8 -*-

import numpy as np
import hashlib
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.io import imread
from scipy.fftpack import dct

def phash(img):
    """
      Perceptual hashing, implemented by following
      http://www.hackerfactor.com/blog/?/archives/432-Looks-Like-It.html

      Takes an image matrix preferablly loaded by skimage
      Returns base64 integer value (img_signature) and binary
      representation of the hash value (hash_val)
    """
    img = resize(img,[32,32])
    img = rgb2gray(img)

    # Apply DCT tranformation
    coefs = dct(img)

    # Keep only lower frequencies
    lower_coefs = coefs[0:8, 0:8]

    # Ignore first coeff value since it throws off the average value
    lower_coefs_tmp = lower_coefs.copy()
    lower_coefs_tmp[0,0] = 0

    # Take the mean and compute binary mask
    mean = lower_coefs_tmp.mean()
    hash_val = np.zeros([8,8])
    hash_val[lower_coefs>mean] = 1
    str_rep = [ str(int(val)) for val in list(hash_val.flatten())]
    str_rep = ''.join(str_rep)
    img_signature = int(str_rep,2)
    return img_signature, hash_val.flatten()

def dhash(img, hash_size = 8):
    """
      Computes dhash value for the given image. It is good for detecting
      duplicate images in differetn sizes and possible deformations
    """

    import cv2
    if img.ndim == 3:
      img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_g = cv2.resize(img, (hash_size+1, hash_size))

    diff = img_g[1:,:] > img_g[:-1,:]
    str_rep = [ str(int(val)) for val in list(diff.flatten())]
    str_rep = ''.join(str_rep)
    # print str_rep
    # img_signature = '%016X' % long(str_rep, 2)
    return str_rep, diff.astype('uint8')

def md5(img):
    img_idx = hashlib.md5(img.tostring()).hexdigest()
    return img_idx
