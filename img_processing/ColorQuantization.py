from pylab import imread,imshow,figure,show,subplot
from ..utils.img_utils import img_resize
from numpy import reshape,uint8,flipud
from scipy.cluster.vq import kmeans,vq
from matplotlib.pylab as plt
import cv2


def quantize_color(img, centroids = None, use_lab=True, use_sklearn=False, debug=True):
	"""
	   img 			: img read by cv2
	   centroids 	: precomputed color value centroids. If not provided, it is learned implicitly
	   use_lab 		: convert img into lab color space before quantization. Lab space has perceptual
	   			 meaning with euclidean distance
	   use_sklearn	: prefer sklearn KMeans. It is better in precision but loss speed
	   debug		: show before and after images
	"""
	
	if use_lab:
		img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
		# reshaping the pixels matrix
		pixel = reshape(img_lab,(img_lab.shape[0]*img_lab.shape[1],3))
	else:
		pixel = reshape(img,(img.shape[0]*img.shape[1],3))

	# performing the clustering
	if centroids is None:
		if use_sklearn:
			kmeans = KMeans(n_clusters=8, random_state=0).fit(pixel)
			centroids = kmeans.cluster_centers_
		else:
			centroids,_ = kmeans(pixel,8) # six colors will be found
		
	#quantization
	qnt,_ = vq(pixel,centroids)

	# reshaping the result of the quantization
	centers_idx = reshape(qnt,(img.shape[0],img.shape[1]))
	img_q = centroids[centers_idx]

	if use_lab:
		img_q = cv2.cvtColor(img_q, cv2.COLOR_LAB2BGR)


	if debug:
		plt.figure()
		plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

		plt.figure()
		plt.imshow(cv2.cvtColor(img_q, cv2.COLOR_BGR2RGB))


