import os
import numpy as np
import pandas as pd
import cPickle
import time

import mxnet as mx
import caffe

import logging
from skimage import io, transform
from matplotlib import pylab as plt
from config import Config

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

REPO_DIRNAME = os.path.abspath(os.path.dirname(__file__) + '/../..')
config = Config()

class ImageNet1KInceptionMXNet(object):
    """
    Interface for pretrained mxnet inception model on 1000 concepts used by ImageNet
    challenge. It is able to extract features and classify the given image
    """
    def __init__(self, gpu_mode, crop_center=False, is_retrieval=False):
        """
        Parameters
        ----------
        gpu_mode : bool
            If True model runs on GPU
        crop_center : bool, optional
            if True, model crops the image center by resizing the image regarding 
            shortest side.
        is_retrieval : bool, optional
            if True, model constructs feature extractor.
        """
        ROOT_PATH = config.NN_MODELS_ROOT_PATH+'Models/MxNet/Inception/'
        
        # Load the pre-trained model
        prefix = ROOT_PATH+"Inception_BN"
        num_round = 39
        if gpu_mode:
            self.model = mx.model.FeedForward.load(prefix, num_round, ctx=mx.gpu(), numpy_batch_size=1)
        else:
            self.model = mx.model.FeedForward.load(prefix, num_round, ctx=mx.cpu(), numpy_batch_size=1)
        
        # Load mean file
        self.mean_img = mx.nd.load(ROOT_PATH+"mean_224.nd")["mean_img"]
        
        # Load synset (text label)
        self.synset = [l.strip() for l in open(ROOT_PATH+'synset.txt').readlines()]

        # Crop the image center if defined
        self.crop_center = crop_center

        if is_retrieval:
            # get internals from model's symbol
            internals = self.model.symbol.get_internals()
            # get feature layer symbol out of internals
            fea_symbol = internals["global_pool_output"]
            if gpu_mode:
                self.feature_extractor = mx.model.FeedForward(ctx=mx.gpu(), symbol=fea_symbol, numpy_batch_size=1,
                                         arg_params=self.model.arg_params, aux_params=self.model.aux_params,
                                         allow_extra_params=True)
            else:
                self.feature_extractor = mx.model.FeedForward(ctx=mx.cpu(), symbol=fea_symbol, numpy_batch_size=1,
                                         arg_params=self.model.arg_params, aux_params=self.model.aux_params,
                                         allow_extra_params=True)
        
        
    def preprocess_image(self, img, show_img=False):
        # load image
        if type(img) == str:
            img = io.imread(img)
        #print("Original Image Shape: ", img.shape)
        if self.crop_center:
            # we crop image from center
            short_egde = min(img.shape[:2])
            yy = int((img.shape[0] - short_egde) / 2)
            xx = int((img.shape[1] - short_egde) / 2)
            crop_img = img[yy : yy + short_egde, xx : xx + short_egde]
        else:
            crop_img = img
        # resize to 224, 224
        resized_img = transform.resize(crop_img, (224, 224))
        if show_img:
            io.imshow(resized_img)
        # convert to numpy.ndarray
        sample = np.asarray(resized_img) * 256
        # swap axes to make image from (224, 224, 4) to (3, 224, 224)
        sample = np.swapaxes(sample, 0, 2)
        sample = np.swapaxes(sample, 1, 2)
        # sub mean 
        normed_img = sample - self.mean_img.asnumpy()
        normed_img.resize(1, 3, 224, 224)
        return normed_img
    
    def classify_image(self, img):
        start = time.time()
        img = self.preprocess_image(img)
        # Get prediction probability of 1000 classes from model
        prob = self.model.predict(img)[0]
        end   = time.time()
        # Argsort, get prediction index from largest prob to lowest
        pred = np.argsort(prob)[::-1]
        # Get top5 label
        top5 = [self.synset[pred[i]] for i in range(5)]
        top5 = [top_str[top_str.find(' ')::].split(',')[0] for top_str in top5]
        top5_probs = ["%.2f" % pr for pr in prob[pred[0:5]]]
        top5 = zip(top5, top5_probs)
        return (True, top5, '%.3f' % (end - start))

    def feature_extraction(self, img):
        query_img = self.preprocess_image(img, show_img=False)
        query_feat = self.feature_extractor.predict(query_img)
        return query_feat

class ImageNet1KInceptionV3MXNet(object):
    """
    Interface for pretrained mxnet inception model on 1000 concepts used by ImageNet
    challenge. It is able to extract features and classify the given image
    """
    def __init__(self, gpu_mode, crop_center=False, is_retrieval=False):
        """
        Parameters
        ----------
        gpu_mode : bool
            If True model runs on GPU
        crop_center : bool, optional
            if True, model crops the image center by resizing the image regarding 
            shortest side.
        is_retrieval : bool, optional
            if True, model constructs feature extractor.
        """
        ROOT_PATH = config.NN_MODELS_ROOT_PATH+'Models/MxNet/imagenet-1k-Inceptionv3/'
        
        # Load the pre-trained model
        prefix = ROOT_PATH+"Inception-7"
        num_round = 1
        if gpu_mode:
            self.model = mx.model.FeedForward.load(prefix, num_round, ctx=mx.gpu(), numpy_batch_size=1)
        else:
            self.model = mx.model.FeedForward.load(prefix, num_round, ctx=mx.cpu(), numpy_batch_size=1)
        
        # Load synset (text label)
        self.synset = [l.strip() for l in open(ROOT_PATH+'synset.txt').readlines()]

        # Crop the image center if defined
        self.crop_center = crop_center

        if is_retrieval:
            # get internals from model's symbol
            internals = self.model.symbol.get_internals()
            # get feature layer symbol out of internals
            fea_symbol = internals["global_pool_output"]
            if gpu_mode:
                self.feature_extractor = mx.model.FeedForward(ctx=mx.gpu(), symbol=fea_symbol, numpy_batch_size=1,
                                         arg_params=self.model.arg_params, aux_params=self.model.aux_params,
                                         allow_extra_params=True)
            else:
                self.feature_extractor = mx.model.FeedForward(ctx=mx.cpu(), symbol=fea_symbol, numpy_batch_size=1,
                                         arg_params=self.model.arg_params, aux_params=self.model.aux_params,
                                         allow_extra_params=True)
        
        
    def preprocess_image(self,path):
        "first resize image to 384 x 384"
        # load image
        img = io.imread(path)
        #print("Original Image Shape: ", img.shape)
        # we crop image from center
        short_egde = min(img.shape[:2])
        yy = int((img.shape[0] - short_egde) / 2)
        xx = int((img.shape[1] - short_egde) / 2)
        crop_img = img[yy : yy + short_egde, xx : xx + short_egde]
        # resize to 299, 299
        resized_img = transform.resize(crop_img, (299, 299))
        # convert to numpy.ndarray
        sample = np.asarray(resized_img) * 256
        # swap axes to make image from (299, 299, 3) to (3, 299, 299)
        sample = np.swapaxes(sample, 0, 2)
        sample = np.swapaxes(sample, 1, 2)
        # sub mean
        normed_img = sample - 128.
        normed_img /= 128.

        return np.reshape(normed_img, (1, 3, 299, 299))
    
    def classify_image(self, img):
        start = time.time()
        img = self.preprocess_image(img)
        # Get prediction probability of 1000 classes from model
        prob = self.model.predict(img)[0]
        end   = time.time()
        # Argsort, get prediction index from largest prob to lowest
        pred = np.argsort(prob)[::-1]
        # Get top5 label
        top5 = [self.synset[pred[i]] for i in range(5)]
        top5 = [top_str[top_str.find(' ')::].split(',')[0] for top_str in top5]
        top5_probs = ["%.2f" % pr for pr in prob[pred[0:5]]]
        top5 = zip(top5, top5_probs)
        return (True, top5, '%.3f' % (end - start))

    def feature_extraction(self, img):
        query_img = self.preprocess_image(img)
        query_feat = self.feature_extractor.predict(query_img)
        return query_feat
    
class ImageNet21KInceptionMXNet(object):
    def __init__(self, gpu_mode, crop_center=False):
        """
        Parameters
        ----------
        gpu_mode : bool
            If True model runs on GPU
        crop_center : bool, optional
            if True, model crops the image center by resizing the image regarding 
            shortest side.
        """
        ROOT_PATH = config.NN_MODELS_ROOT_PATH+'Models/MxNet/imagenet-21k-inception/Inception-Full/'
        
        # Load the pre-trained model
        prefix = ROOT_PATH+"Inception"
        num_round = 9
        if gpu_mode:
            self.model = mx.model.FeedForward.load(prefix, num_round, ctx=mx.gpu(), numpy_batch_size=1)
        else:
            self.model = mx.model.FeedForward.load(prefix, num_round, ctx=mx.cpu(), numpy_batch_size=1)
        
        self.mean_img = self.mean_img = np.ones([3,224,224])*117.0

        self.crop_img = crop_img
        
        # Load synset (text label)
        self.synset = [l.strip() for l in open(ROOT_PATH+'synset.txt').readlines()]
        
        
    def preprocess_image(self, img, show_img=False):
        # load image
        if type(img) == str:
            img = io.imread(img)
        # we crop image from center
        if self.crop_img:
            short_egde = min(img.shape[:2])
            yy = int((img.shape[0] - short_egde) / 2)
            xx = int((img.shape[1] - short_egde) / 2)
            crop_img = img[yy : yy + short_egde, xx : xx + short_egde]
        else:
            crop_img = img
        # resize to 224, 224
        resized_img = transform.resize(crop_img, (224, 224))
        if show_img:
            io.imshow(resized_img)
        # convert to numpy.ndarray
        sample = np.asarray(resized_img) * 256
        # swap axes to make image from (224, 224, 4) to (3, 224, 224)
        sample = np.swapaxes(sample, 0, 2)
        sample = np.swapaxes(sample, 1, 2)
        # sub mean 
        normed_img = sample - self.mean_img
        normed_img.resize(1, 3, 224, 224)
        print img.mean()
        return normed_img
    
    def classify_image(self, img):
        start = time.time()
        img = self.preprocess_image(img)
        # Get prediction probability of 1000 classes from model
        prob = self.model.predict(img)[0]
        end   = time.time()
        # Argsort, get prediction index from largest prob to lowest
        pred = np.argsort(prob)[::-1]
        # Get top5 label
        top5 = [self.synset[pred[i]] for i in range(5)]
        top5 = [top_str[top_str.find(' ')::].split(',')[0] for top_str in top5]
        top5_probs = ["%.2f" % pr for pr in prob[pred[0:5]]]
        top5 = zip(top5, top5_probs)
        print "WQETQWETQWERQWERQWERQWER", prob[pred[0:5]]
        return (True, top5, '%.3f' % (end - start))


class Clothes21MXNet(object):
    """
    Interface for clothes model backed by MxNet trained by Pinterest images. 
    It supports 21 clothing types
    """
    def __init__(self, gpu_mode):
        """
        Parameters
        ----------
        gpu_mode : bool
            If True model runs on GPU
        crop_center : bool, optional
            if True, model crops the image center by resizing the image regarding 
            shortest side.
        """
        ROOT_PATH = config.NN_MODELS_ROOT_PATH+'Models/MxNet/AlexNet/'
        
        # Load the pre-trained model
        prefix = ROOT_PATH+"inception_AlexNet"
        num_round = 24
        if gpu_mode:
            self.model = mx.model.FeedForward.load(prefix, num_round, ctx=mx.gpu(), numpy_batch_size=1)
        else:
            self.model = mx.model.FeedForward.load(prefix, num_round, ctx=mx.cpu(), numpy_batch_size=1)
        
        # Load mean file
        self.mean_img = np.ones([3,224,224])*127.0
        
        # Load synset (text label)
        self.categories = [l.split()[1] for l in open(ROOT_PATH+'categories.txt').readlines()]
        
        
    def preprocess_image(self, img, show_img=False):
        # load image
        if type(img) == str:
            img = io.imread(img)
        # we crop image from center
        if self.crop_img:
            short_egde = min(img.shape[:2])
            yy = int((img.shape[0] - short_egde) / 2)
            xx = int((img.shape[1] - short_egde) / 2)
            crop_img = img[yy : yy + short_egde, xx : xx + short_egde]
        else:
            crop_img = img
        # resize to 224, 224
        resized_img = transform.resize(crop_img, (224, 224))
        if show_img:
            io.imshow(resized_img)
        # convert to numpy.ndarray
        sample = np.asarray(resized_img) * 256
        # swap axes to make image from (224, 224, 4) to (3, 224, 224)
        sample = np.swapaxes(sample, 0, 2)
        sample = np.swapaxes(sample, 1, 2)
        # sub mean 
        normed_img = sample - self.mean_img
        normed_img.resize(1, 3, 224, 224)
        print img.mean()
        return normed_img
    
    def classify_image(self, img):
        start = time.time()
        img = self.preprocess_image(img)
        # Get prediction probability of 1000 classes from model
        prob = self.model.predict(img)[0]
        end   = time.time()
        # Argsort, get prediction index from largest prob to lowest
        pred = np.argsort(prob)[::-1]
        # Get top5 label
        top5 = [self.categories[pred[i]] for i in range(5)]
        top5_probs = ["%.2f" % pr for pr in prob[pred[0:5]]]
        top5 = zip(top5, top5_probs)
        print "WQETQWETQWERQWERQWERQWER", prob[pred[0:5]]
        return (True, top5, '%.3f' % (end - start))
    
class Places2Caffe(object):
    """
    Interface for the Places2 model backed by Caffe. It supports 408 scene classes.
    """
    def __init__(self,gpu_mode):
        """
        Parameters
        ----------
        gpu_mode : bool
            If True model runs on GPU
        """
        MODEL_DEPLOY  = config.NN_MODELS_ROOT_PATH+'Models/Caffe/Places2/deploy.prototxt'
        MODEL_BINARY  = config.NN_MODELS_ROOT_PATH+'Models/Caffe/Places2/inception_bn_aug_iter_60000.caffemodel'
        CATEGORY_FILE = config.NN_MODELS_ROOT_PATH+'Models/Caffe/Places2/categories.txt'
        
        logging.info('Loading net and associated files...')
        if gpu_mode:
            logging.info('Model in GPU mode !!') 
            caffe.set_mode_gpu()
        else:
            logging.info('Model in CPU mode !!') 
            caffe.set_mode_cpu()
        
        self.net = caffe.Net(MODEL_DEPLOY,
                        MODEL_BINARY,
                        caffe.TEST)

        # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2,0,1))
        self.transformer.set_mean('data', np.array([127.0,127.0,127.0])) # mean pixel
        self.transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
        self.transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
        
        # set net to batch size of 1
        self.net.blobs['data'].reshape(1,3,224,224)

        with open(CATEGORY_FILE) as f:
            labels_df = pd.DataFrame([
                {
                    'synset_id': l.strip().split(' ')[1],
                    'name':  l.strip().split(' ')[0].split('/')[-1]
                }
                for l in f.readlines()
            ])
        self.labels = labels_df['name'].values

    def classify_image(self, image):
            starttime = time.time()
            self.net.blobs['data'].data[...] = self.transformer.preprocess('data', image/255.0)
            scores = self.net.forward()
            scores = self.net.blobs['prob'].data[0].flatten()
            endtime = time.time()

            indices = self.net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
            print "INDICES: ",indices
            predictions = self.labels[indices]

            bet_result = zip(predictions, scores[indices])
            print bet_result
            return (True, bet_result, '%.3f' % (endtime - starttime))

        
