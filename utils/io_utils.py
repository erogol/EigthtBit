import numpy as np
import urllib
from skimage.io import imread, imsave
import multiprocessing 
from multiprocessing import Pool
import dill


def imgread(url):
    img = imread(url)
    return img

def imgsave(img, img_path):
    imsave(img_path, img)


def save_obj(obj, file_path):
    dill.dump(obj, open(file_path,'wb'))

def load_obj(file_path):
    return dill.load(open(file_path, 'rb'))

def save_feature(file_path, feat_vec, check_exist = False):
	"""
		file_path - file path to output_ext
		feat_vec - feature vector in numpy format
		check_exist - check whether the file exist
	"""
	if check_exist and os.path.exists(file_path):
			print 'Exists : ',file_path
			return 
	np.savetxt(file_path, feat_vec, delimiter=',', fmt='%.4e')

def _read_file(file_path, count):
    if count % 250 == 0:
        print multiprocessing.current_process(), ' --- ', count 
    x = np.loadtxt(file_path, delimiter='\n')
    x = x.astype('float64')
    return x

def _read_file_helper(args):
    return _read_file(*args)

def read_files(file_paths, num_proc):
    """
        Read files from the givne file_path list, possibly with 
        multiprocessing support
    """
    args = [(file_path, count) for count,file_path in enumerate(file_paths)]
    data = []
    if num_proc > 1:
        p = Pool(num_proc)
        try:
            data = p.map(_read_file_helper, args)
            p.close()
            return data
        except KeyboardInterrupt:
            print 'parent received control-c'
            return
    else:
        x = np.loadtxt(args[0][0], delimiter='\n')
        data = np.zeros([len(args), x.shape[0]])
        for arg in args:
            x = _read_file(arg[0], arg[1])
            data[arg[1],:] = x[None,:]
        return data