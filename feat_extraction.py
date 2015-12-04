from progressbar import ProgressBar
from utils.io_utils import save_feature, read_img
from utils.data_utils import change_filepaths
from pathos.multiprocessing import ProcessingPool
import os
import cv2

def extract_to_file(img_paths, img_file_ext, output_file_ext, extractor_obj):
	"""
		It calls feature extractor function and saves the features to files with the given 
		file extension and at the same path as images

		img_paths : list of img file paths 
		img_file_ext : image file extension to replace it with output_file_ext
		output_file_ext : output file extension
		extractor_obj : feature extractor object with transform function
	"""
	feat_paths = change_filepaths(img_paths, img_file_ext, output_file_ext)
	ProcessingPool().map(extract_to_file_helper, img_paths, feat_paths, [extractor_obj]*len(img_paths))
	print 'Finished!!'


def extract_to_file_helper(img_path, feat_path, extractor_obj):
	if os.path.basename(img_path).split('.')[0] != os.path.basename(feat_path).split('.')[0]:
		print img_path
		print feat_path
		return 
	img = cv2.imread(img_path)
	feat_vec = extractor_obj.transform(img)
	save_feature(feat_path, feat_vec, False)