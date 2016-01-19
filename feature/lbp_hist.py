#import numpy as np
#from skimage.feature import local_binary_pattern

class LbpHist():
	"""
		TODO: add grid wise implementation 
	"""
	def __init__(self):

	def transform(self, img):
		METHOD = 'uniform'
		radius = 4
		n_points = 8 * radius
		lbp = local_binary_pattern(img, n_points, radius, METHOD)
		n_bins = lbp.max() + 1
		hist, _ = np.histogram(lbp, normed=True, bins=n_bins, range=(0, n_bins))
		return hist