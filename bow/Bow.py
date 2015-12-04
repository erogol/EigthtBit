import cv2
from ..utils import *
from sklearn.cluster import MiniBatchKMeans, KMeans
from progressbar import ProgressBar
import time
from pathos.multiprocessing import ProcessingPool

class Bow(object):
	"""
	    Bag Of Words for defined feature type
	    It takes images as URL for memory efficiency
	"""
	def __init__(self, num_words = 1200, num_feats = 500000, desc_name = 'SIFT', ngrids = [2,2], center = True, pre_pipeline = None):
		"""
			num_words 		: size of Vocab
			num_feats 		: size of random crowd of feature set to learn Vocab
			desc_name 		: name of the descriptor
			ngrids    		: spatial gridding size. Default is [2,2] four quadrants
			center    		: if true take a center spatial grid from the image as well.
			pre_pipeline 	: Pipeline object for image preprocessing. It is applied to any image
							before feature extraction and vocab learning 
		"""
		#super(Bow, self).__init__()
		self.num_words = num_words
		self.num_feats = num_feats
		self.desc_name = desc_name
		# if desc_name == 'SIFT':
		# 	self.desc = cv2.SIFT()
		# 	self.feat_length = 128
		# elif desc_name == "SURF":
		# 	self.desc = cv2.SURF()
		# 	self.feat_length = 64
		# elif desc_name == "ORB":
		# 	self.desc = cv2.ORB()
		# 	self.feat_length = 32

		self.VOCAB = []
		#self.kmeans = MiniBatchKMeans(n_clusters = self.num_words, 
		#                         init='k-means++', 
		#                         batch_size = 5, 
		#                         init_size = 3*self.num_words,
		#                         reassignment_ratio = 0.05,
		#                         max_no_improvement = 50)

		self.kmeans = KMeans(n_clusters=self.num_words, 
							init='k-means++', 
							n_init=10, 
							max_iter=300, 
							tol=0.0001, 
							precompute_distances=True, 
							verbose=0, 
							random_state=None, 
							copy_x=True, 
							n_jobs=3)
		self.ngrids = ngrids
		self.center = center

		self.pre_pipeline = pre_pipeline

	def _collect_feature(self, img_path, num_feat_per_img, img_count):
		if self.desc_name == 'SIFT':
			desc_compute = cv2.SIFT()
		elif self.desc_name == "SURF":
			desc_compute = cv2.SURF()
		elif self.desc_name == "ORB":
			desc_compute = cv2.ORB()

		if img_count % 100 == 0:
			print img_count
		detector = cv2.FastFeatureDetector()
		img = read_img(img_path)
		if self.pre_pipeline is not None:
			img = self.pre_pipeline.transform(img)[0]
		kp = detector.detect(img, None)
		kp,desc =desc_compute.compute(img,kp)
		#     print desc.shape
		if desc == None:
		    print img_path," has no SIFT feat"
		    return
		if desc.shape[0] > num_feat_per_img:
		    feat_indices = np.random.permutation(desc.shape[0])[:num_feat_per_img]
		    return desc[feat_indices,:]
		else:
		    return desc 
		  
	def learn_vocab(self, img_paths):
		num_feat_per_img = int(np.ceil(float(self.num_feats) / len(img_paths)))
		feat_vecs = np.zeros([0, self.num_words])

		if self.pre_pipeline is not None:
			print 'Preprocessing Pipeline Available  ... '

		print "Collecting Features for Vocab Learning ..."
		# Extract Features for Vocabulary Learning
		feat_vecs = ProcessingPool().map(self._collect_feature, img_paths, [num_feat_per_img]*len(img_paths), range(len(img_paths)))   
		
		feat_vecs_np = numpy.concatenate( feat_vecs, axis=0 )
		# Learn the Vocab
		print "Quantization ..."
		self.kmeans.fit(feat_vecs_np)
		self.VOCAB = self.kmeans.cluster_centers_

	def compute_feat_hist(self, img):
		if self.desc_name == 'SIFT':
			desc_compute = cv2.SIFT()
		elif self.desc_name == "SURF":
			desc_compute = cv2.SURF()
		elif self.desc_name == "ORB":
			desc_compute = cv2.ORB()

		# detect and compute features
		start_k = time.time()
		#dense=cv2.FeatureDetector_create("Dense")
		detector = cv2.FastFeatureDetector()
		kp = detector.detect(img)
		kp,desc =desc_compute.compute(img,kp)
		print 'feature extract :', time.time() - start_k

		# check null feature. If no feature detected return zero hist
		if desc is None:
		    print " has no SIFT feat"
		    return  np.zeros([self.num_words])

		# compute feat histogram 
		start_k = time.time()
		print 'Feat shape', desc.shape
		K = self.kmeans.predict(desc)
		print 'Kmeans :', time.time() - start_k
		hist,_ = np.histogram(K, normed=False, bins = self.num_words, range=[0,self.num_words])
		return hist

	def transform(self, img):
		start_t = time.time()
		# read image
		if type(img) is str:
			img = read_img(img)
			print 'Bow image reading'
		if self.pre_pipeline is not None:
		    img = self.pre_pipeline.transform(img)[0]
		    print 'Bow preprocessing pipeline '
		print 'Image read :', time.time() - start_t

		# img gridding 
		img_grids = create_img_grids(img, num_grids=self.ngrids, center=self.center)
		print 'Image gridding :', time.time() - start_t
		
		# feature transformation
		feat = []
		feat = ProcessingPool(nodes = self.ngrids[0]*self.ngrids[1]).map(self.compute_feat_hist, img_grids)				
		
		# merge grid features and l1 normalize
		feat = np.array(feat).flatten().astype('float32')
		feat = feat / feat.sum() #L1 normalization
		print 'Finished :', time.time() - start_t
		return feat

	def batch_transform(self, img_paths):
		feat_vecs = []
		for img_path in img_paths:
			feat_vec = self.transform(img_path)
			feat_vecs.append([img_path,feat_vec])
		return feat_vecs