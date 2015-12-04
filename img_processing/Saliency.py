from methods.SaliencyMap import *
from ..utils import *
from ..utils.img_utils import img_resize
from skimage.segmentation import mark_boundaries,slic, felzenszwalb
from skimage.util import img_as_float
from skimage.morphology import convex_hull_image

class SaliencyMask():
	def __init__(self):
		self.mask = None
		self.cropped_mask = None
		self.crop_coords = None

	def transform(self,img, cropped = True):
	    #print img.shape
	    sm = SaliencyMap(img)
	    sm.map[sm.map < sm.map.mean()] = 0
	    sm.map[sm.map > sm.map.mean()] = 1
	    sm.map = sm.map.astype('uint8')
	    sm.map = convex_hull_image(sm.map).astype('uint8')
	    self.mask = sm.map
	    if cropped:
	        self.cropped_mask , self.crop_coords  = remove_img_border(self.mask)
	        img = img[self.crop_coords[0]:self.crop_coords[1], self.crop_coords[2]:self.crop_coords[3], :]
	        return img, self.cropped_mask
	    else:
	        return self.mask

	def apply_mask(self, img):
		"""
			Apply predefined mask to new image with possible different size
		"""
		mask_resized = img_resize(self.mask, img.shape)
		cropped_mask, crop_coords = remove_img_border(mask_resized.astype('uint8'))
		img = img[crop_coords[0]:crop_coords[1], crop_coords[2]:crop_coords[3], :]
		return img, cropped_mask


class InverseSaliencyMask():
	def __init__(self):
		self.mask = None
		self.cropped_mask = None
		self.crop_coords = None
	
	def transform(self, img, cropped = True):
	    #print img.shape
	    sm = SaliencyMap(img)
	    A = np.zeros_like(sm.map)
	    A[sm.map < sm.map.mean()] = 1
	    A = convex_hull_image(A).astype('uint8')
	    self.mask = A
	    if cropped:
	        self.cropped_mask , self.crop_coords = remove_img_border(A.astype('uint8'))
	        img = img[self.crop_coords[0]:self.crop_coords[1], self.crop_coords[2]:self.crop_coords[3], :]
	        return img, self.cropped_mask
	    else:
	        return self.mask

	def apply_mask(self, img):
		"""
			Apply predefined mask to new image with possible different size
		"""
		mask_resized = img_resize(self.mask, img.shape)
		cropped_mask, crop_coords = remove_img_border(mask_resized.astype('uint8'))
		img = img[crop_coords[0]:crop_coords[1], crop_coords[2]:crop_coords[3], :]
		return img, cropped_mask


class SaliencyMaskSlic():
	def __init__(self):
		self.mask = None
		self.cropped_mask = None
		self.crop_coords = None

	def transform(self, img, cropped=True):
		image = img_as_float(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
		cont = True
		# loop over the number of segments
		numSegments = 5

		while cont:
		#         print 'count'
		    # apply SLIC and extract (approximately) the supplied number
		    # of segments
		    segments = slic(image, n_segments = numSegments, multichannel = True, sigma = 1., enforce_connectivity=True, slic_zero=True)
		    segments[segments == 0] = segments.max()+1

		    #find superpixels lying at the borders
		    pixel_vals = np.unique(np.hstack([segments[:,0], segments[:,-1],segments[-1,:],segments[0,:]]))
		    #print pixel_vals

		    # Remvoe them
		    mask = np.in1d(segments, pixel_vals)
		    mask = mask.reshape(segments.shape)
		    segments[mask] = 0

		    if np.unique(np.hstack([segments[0,:], segments[:,-1], segments[:,0], segments[-1,:]])).shape[0] == 1 and np.count_nonzero(segments) > segments.size/2:
		        cont = False
		    else:
		        numSegments += 2
		segments[segments>0] = 1
		#segments = convex_hull_image(segments).astype('uint8')
		self.mask = segments
		if cropped:
			self.cropped_mask , self.crop_coords = remove_img_border(self.mask.astype('uint8'))
			img = img[self.crop_coords[0]:self.crop_coords[1], self.crop_coords[2]:self.crop_coords[3], :]
			return img, self.cropped_mask
		else:
		    return self.mask

	def apply_mask(self, img):
		"""
			Apply predefined mask to new image with possible different size
		"""
		mask_resized = img_resize(self.mask, img.shape)
		cropped_mask, crop_coords = remove_img_border(mask_resized.astype('uint8'))
		img = img[crop_coords[0]:crop_coords[1], crop_coords[2]:crop_coords[3], :]
		return img, cropped_mask

class SaliencyMaskFal():
	"""
		Perform felzenszwalb's graph based image segmentation.
		It is the best for retail product images
	"""
	def __init__(self):
		self.mask = None
		self.cropped_mask = None
		self.crop_coords = None

	def transform(self, img, cropped=True):
		image = img_as_float(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
		cont = True
		segments = felzenszwalb(img, min_size=20)
		segments[segments == 0] = segments.max()+1

		pixel_vals = np.unique(np.hstack([segments[:,0], segments[:,-1],segments[-1,:],
		                                  segments[0,:]]))
		#print pixel_vals

		# Remvoe them
		m = np.in1d(segments, pixel_vals)
		m = m.reshape(segments.shape)
		segments[m] = 0
		segments[segments>0] = 1

		self.cropped_mask , self.crop_coords = remove_img_border(segments.astype('uint8'))
		self.mask = segments
		if cropped:
			self.cropped_mask , self.crop_coords = remove_img_border(self.mask.astype('uint8'))
			img = img[self.crop_coords[0]:self.crop_coords[1], self.crop_coords[2]:self.crop_coords[3], :]
			return img, self.cropped_mask
		else:
		    return self.mask

	def apply_mask(self, img):
		"""
			Apply predefined mask to new image with possible different size
		"""
		mask_resized = img_resize(self.mask, img.shape)
		cropped_mask, crop_coords = remove_img_border(mask_resized.astype('uint8'))
		img = img[crop_coords[0]:crop_coords[1], crop_coords[2]:crop_coords[3], :]
		return img, cropped_mask

