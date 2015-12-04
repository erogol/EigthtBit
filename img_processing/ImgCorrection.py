import cv2


def hist_equalization(img):
	"""
		Corrects color distribution by applyign adaptive histogram 
		equalization to given image. If img is in BGR space, it is 
		converted to YCR_CB space and Y channel is equalized. If img is
		in gray scale then direct hist equalization is applied

		img : 3D BGR img array or 2D GRay img array 
	"""
	clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
	if len(img.shape) > 3: # check color img

		# Convert BGR to YCR_CB color space
		img2 = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

		# Take Y channel and apply adaptive histogram equalization
		[Y,C,R] = cv2.split(img2)
		Y = clahe.apply(Y)

		# Merge Channels andcovert img to BGR again
		img_equalized = cv2.merge([Y,C,R])
		img_equalized = cv2.cvtColor(img_equalized, cv2.COLOR_YCR_CB2BGR)
	
	elif len(img.shape) = 2:
		img_equalized = clahe.apply(img)

	return img_equalized

