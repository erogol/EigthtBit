import numpy as np
import cv2
from scipy.spatial import distance

def hist_intersection(x, y):
    return np.minimum(x, y).sum()

def bhattacharyya(A,B):
    A = A.astype('float32')
    B = B.astype('float32')
    return cv2.compareHist(A,B,cv2.cv.CV_COMP_BHATTACHARYYA)

def kl_divergence(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    filt = np.logical_and(p != 0, q != 0)
    return np.sum(p[filt] * np.log2(p[filt] / q[filt]))

# def rank_by_similarity(query_feat_list = [], data_list = [], DIST_FUNC = 'euclidean'):
# 	distances = []
# 	for count,X in enumerate(data_list):
# 		query = query_feat_list[count]
# 		dist = distance.cdist(query[None,:], X,DIST_FUNC)
# 		# Normalize Distances
# 		dist = dist / query.shape[1]
# 		distances.append(dist)

#     whole_dist = 1
# 	for dist in distances
# 		whole_dist *= dist
# 	ranking_img_indices = np.argsort(whole_dist)[0].astype(int)
# 	ranking_dist = np.sort(whole_dist)
# 	return ranking_img_indices, ranking_dist
