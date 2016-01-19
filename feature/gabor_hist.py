import cv2
#from img_utils import create_img_grids

class GaborHist():
	def __init__(sigmas=[2], theta_div=8, ksize = 10, ngrids = [2,2], pre_pipeline = None):
		self.sigmas = sigmas
		self.theta_div = theta_div
		self.ksize = ksize
		self.filters = []

		self.ngrids = ngrids
		self.pre_pipeline = pre_pipeline

	def build_filters(self):
		self.filters = []
	    for sigma in self.sigmas:
	        for theta in np.arange(0, np.pi, np.pi / self.theta_div):
	         kern = cv2.getGaborKernel((self.ksize, self.ksize), sigma, theta, 5, 0.5, 0, ktype=cv2.CV_32F)
	         kern /= 1.5*kern.sum()
	         self.filters.append(kern)

	def transform(self, img):
		"""
	        Compute gabor histogram from the given image.
	        1. Compute gabor filter response of the given image
	        2. Extract image grids from the response image
	        3. Compute value histogram of each grid 
	        4. Concanate histograms from the each grid
	        5. Do it for the other filters
	        
	        img : 3D image tensor
	        filters : filters created by build_filters()
	        num_grids : number of horizontal and vertical grids fromt eh given image
	        num_hist_bins :  number of bins for the histograms of each image grid

	        TODO: change hist comput. to cv2 implementation but currently it raises buggy error
	    """
	    if pre_pipeline != None:
		    img = self.pre_pipeline.transform(img)
	    img_feat = []
	    for kern in self.filters:
	        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
	        fpatches = create_img_grids(fimg, self.ngrids)
	        for count,patch in enumerate(fpatches):
	            hist, _ = np.histogram(patch, normed=False, bins=num_hist_bins, range=(0, 255))
	            hist = hist / hist.sum() # l1 norm
	            img_feat.append(hist)
	    return np.array(img_feat).flatten().astype(np.float32)
