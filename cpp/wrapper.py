# pylint: disable=C0103
"""This folder contains some c++ implementations that either make code run
faster or handles some numpy tricky issues.
"""
import ctypes as ct
import numpy as np
import os

# first, let's import the library
try:
    _DLL = np.ctypeslib.load_library('libcpputil.so',
            os.path.join(os.path.dirname(__file__)))
except Exception as error:
    raise error
try:
    _OMP_NUM_THREADS=int(os.environ['OMP_NUM_THREADS'])
except KeyError:
    try:
        import multiprocessing
        _OMP_NUM_THREADS=multiprocessing.cpu_count()
    except ImportError:
        _OMP_NUM_THREADS=1

################################################################################
# im2col operation
################################################################################
_DLL.im2col.restype = None

def im2col(im, col, psize, stride):
    num, height, width, channels = im.shape
    
    assert col.dtype  in [np.float32, np.float64, np.double]
      
    assert im.dtype  in [np.float32, np.float64, np.double]

    assert im.dtype is col.dtype
      
    _DLL.im2col(ct.c_int(im.itemsize),
                im.ctypes.data_as(ct.c_void_p),
                col.ctypes.data_as(ct.c_void_p),
                ct.c_int(num),
                ct.c_int(height),
                ct.c_int(width),
                ct.c_int(channels),
                ct.c_int(psize),
                ct.c_int(stride))