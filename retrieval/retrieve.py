import numpy as np
import cv2
from scipy.spatial import distance

def find_similar_images(query, X, feat_ext_func, post_processing = None, num_sim=10, dist='cosine'):    
    """
    Parameters
    ----------
    query : str or numpy array
        Query image to be compared with the given set
    X : numpy array
        Data collection to be compared with the query
    is_retrieval : bool, optional
        if True, model constructs feature extractor.
    post_processing : sklearn.Pipeline.pipeline, Optional
        post processing ot the feature vector of the query. If not
        defined this stage is skipped.
    num_sim : int,
        how many of the similar items will be returned
    dist : str,
        defines the distance function used by scipy.spatial.distance.cdist

    Return
    ----------
    sim_idx : list
        Similar idx list from most similar to less
    dist : numpy array
        unsorted distances to each data item

    Example
    ----------
    normalizer = preprocessing.Normalizer(norm='l1')
    post_processing = Pipeline([('norm', normalizer)])
    net = ImageNet1KInceptionMXNet(gpu_mode=False, crop_center=True, is_retrieval=True)
    sim_img_index, dist = retrieve_similar_images(query,
                                      X_norm, 
                                      net.feature_extraction,
                                      post_processing, 
                                      10, 
                                      dist='cosine')
    """
    # if query is URL then read img and extract feature
    # else query is feature vector
    query_feat = feat_ext_func(query)
    query_feat = query_feat.flatten()[None,:]
    print query_feat.shape
    if post_processing != None:
        print 'Data Preprocessing ... !'
        print type(post_processing)
        query_feat = post_processing.transform(query_feat)

    # Compute Similarity
    print query_feat.shape
    print X.shape
    dist = distance.cdist(query_feat, X,dist)
    
    # Find similar item idxs
    print "Dist",dist.shape
    idxs = np.argsort(dist)[0]
    
    sim_idxs = list(idxs)
    
    return  sim_idxs[:num_sim],dist
