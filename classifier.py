import os
import sys
import pandas as pd
import cPickle
import time

# for torch models
import lutorpy as lua
import numpy as np
require('nn')
require('cunn')
require('cudnn')
lua.eval("torch.setdefaulttensortype('torch.FloatTensor')")

import logging
from skimage import io, transform
from eight_bit.utils.img_utils import thumbnail, crop_center
from matplotlib import pylab as plt
from config import Config
from utils import img_utils
import mxnet as mx

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

REPO_DIRNAME = os.path.abspath(os.path.dirname(__file__) + '/../..')
config = Config()

class ImageNet1KInceptionMXNet(object):
    """
    Interface for pretrained mxnet inception model on 1000 concepts used by ImageNet
    challenge. It is able to extract features and classify the given image
    """
    def __init__(self, gpu_mode, crop_center=False, is_retrieval=False, batch_size=1):
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
            self.model = mx.model.FeedForward.load(prefix, num_round, ctx=mx.gpu(), numpy_batch_size=batch_size)
        else:
            self.model = mx.model.FeedForward.load(prefix, num_round, ctx=mx.cpu(), numpy_batch_size=batch_size)

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

    def classify_image(self, img, N=5):
        start = time.time()
        img = self.preprocess_image(img)
        # Get prediction probability of 1000 classes from model
        prob = self.model.predict(img)[0]
        end   = time.time()
        # Argsort, get prediction index from largest prob to lowest
        pred = np.argsort(prob)[::-1]
        # Get topN label
        topN = [self.synset[pred[i]] for i in range(N)]
        topN = [top_str[top_str.find(' ')::].split(',')[0] for top_str in topN]
        topN_probs = prob[pred[0:N]]
        topN = zip(topN, pred[0:N])
        topN = [ topN[c] + (topN_prob,) for c,topN_prob in enumerate(topN_probs)]
        #print "WQETQWETQWERQWERQWERQWER", prob[pred[0:5]]
        return '%.3f' % (end - start), topN


    def feature_extraction(self, img):
        query_img = self.preprocess_image(img, show_img=False)
        query_feat = self.feature_extractor.predict(query_img)
        return query_feat

    def produce_cam(self, img, class_id=None, top=-1):

        # Create CAM model outputs Global Average Pooling layer
        internals = self.model.symbol.get_internals()
        fea_symbol = internals['ch_concat_5b_chconcat_output']
        CAM = mx.model.FeedForward(ctx=mx.gpu(), symbol=fea_symbol, numpy_batch_size=1,
                                             arg_params=self.model.arg_params, aux_params=self.model.aux_params,
                                             allow_extra_params=True)
        if type(img) is str or type(img) is unicode:
            img = io.imread(img)

        # Give image, preprocess and get GAP activations
        batch = self.preprocess_image(img)
        GAP = CAM.predict(batch).squeeze()
        # Get fc layer weights
        W = self.model.arg_params['fc_weight'].asnumpy()
        # Get class specific weights
        class_W = W[class_id]
        # Abs of class specific weights
        class_W_abs = np.abs(class_W)
        # Create empty GAP
        class_GAP = np.zeros(GAP.shape)
        if top > 0 :
            top_idxs = (-class_W_abs).argsort()[0:top]
            for count,idx in enumerate(top_idxs):
                class_GAP[idx] = GAP[idx] * class_W_abs[idx]
        else:
            for count,w in enumerate(class_W_abs):
                class_GAP[count] = GAP[count] * w

        # Create CAM
        # Find average GAP*W and normalize to 0,1 scale
        CAM = class_GAP.sum(axis=0)
        CAM = CAM + -1*CAM.min()
        CAM = CAM / CAM.max()
        assert CAM.min() == 0
        assert CAM.max() == 1

        # Resize to image size
        CAM_resized = transform.resize(CAM, (img.shape[0], img.shape[1]), )

        # plt.figure()
        # plt.title(synset[class_id])
        # plt.imshow(CAM_resized,cmap='cubehelix')
        # plt.show()
        return CAM_resized


class ImageNet1KInceptionV3MXNet(object):
    """
    Interface for pretrained mxnet inception model on 1000 concepts used by ImageNet
    challenge. It is able to extract features and classify the given image
    """
    def __init__(self, gpu_mode, crop_center=False, is_retrieval=False, batch_size=1):
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
            self.model = mx.model.FeedForward.load(prefix, num_round, ctx=mx.gpu(), numpy_batch_size=batch_size)
        else:
            self.model = mx.model.FeedForward.load(prefix, num_round, ctx=mx.cpu(), numpy_batch_size=batch_size)

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


    def preprocess_image(self,img):
        "first resize image to 384 x 384"
        # load image
        if type(img) == str:
            img = io.imread(img)
        if self.crop_center:
            short_egde = min(img.shape[:2])
            yy = int((img.shape[0] - short_egde) / 2)
            xx = int((img.shape[1] - short_egde) / 2)
            crop_img = img[yy : yy + short_egde, xx : xx + short_egde]
        else:
            crop_img = img
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

    def classify_image(self, img, preprocess=True, N=5):
        start = time.time()
        if preprocess:
            img = self.preprocess_image(img)
        # Get prediction probability of 1000 classes from model
        prob = self.model.predict(img)[0]
        end   = time.time()
        # Argsort, get prediction index from largest prob to lowest
        pred = np.argsort(prob)[::-1]
        # Get topN label
        topN = [self.synset[pred[i]] for i in range(N)]
        topN = [top_str[top_str.find(' ')::].split(',')[0] for top_str in topN]
        topN_probs = prob[pred[0:N]]
        topN = zip(topN, pred[0:N])
        topN = [ topN[c] + (topN_prob,) for c,topN_prob in enumerate(topN_probs)]
        #print "WQETQWETQWERQWERQWERQWER", prob[pred[0:5]]
        return '%.3f' % (end - start), topN

    def feature_extraction(self, img):
        query_img = self.preprocess_image(img)
        query_feat = self.feature_extractor.predict(query_img)
        return query_feat

    def produce_cam(self, img, class_id=None, top=-1):

        # Create CAM model outputs Global Average Pooling layer
        internals = self.model.symbol.get_internals()
        fea_symbol = internals['ch_concat_mixed_10_chconcat_output']
        CAM = mx.model.FeedForward(ctx=mx.gpu(), symbol=fea_symbol, numpy_batch_size=1,
                                             arg_params=self.model.arg_params, aux_params=self.model.aux_params,
                                             allow_extra_params=True)
        if type(img) is str or type(img) is unicode:
            img = io.imread(img)

        # Give image, preprocess and get GAP activations
        batch = self.preprocess_image(img)
        GAP = CAM.predict(batch).squeeze()
        # Get fc layer weights
        W = self.model.arg_params['fc1_weight'].asnumpy()
        # Get class specific weights
        class_W = W[class_id]
        # Abs of class specific weights
        class_W_abs = np.abs(class_W)
        # Create empty GAP
        class_GAP = np.zeros(GAP.shape)
        if top > 0 :
            top_idxs = (-class_W_abs).argsort()[0:top]
            for count,idx in enumerate(top_idxs):
                class_GAP[idx] = GAP[idx] * class_W_abs[idx]
        else:
            for count,w in enumerate(class_W_abs):
                class_GAP[count] = GAP[count] * w

        # Create CAM
        # Find average GAP*W and normalize to 0,1 scale
        CAM = class_GAP.sum(axis=0)
        CAM = CAM + -1*CAM.min()
        CAM = CAM / CAM.max()
        assert CAM.min() == 0
        assert CAM.max() == 1

        # Resize to image size
        CAM_resized = transform.resize(CAM, (img.shape[0], img.shape[1]), )

        # plt.figure()
        # plt.title(synset[class_id])
        # plt.imshow(CAM_resized,cmap='cubehelix')
        # plt.show()
        return CAM_resized

class ImageNet21KInceptionMXNet(object):
    def __init__(self, gpu_mode, crop_center=False, is_retrieval=0, batch_size=1):
        """
        Parameters
        ----------
        gpu_mode : bool
            If True model runs on GPU
        crop_center : bool, optional
            if True, model crops the image center by resizing the image regarding
            shortest side.
        """



        ROOT_PATH = config.NN_MODELS_ROOT_PATH+'Models/MxNet/imagenet-21k-inception/'

        # Load the pre-trained model
        prefix = ROOT_PATH+"Inception"
        num_round = 9
        if gpu_mode:
            self.model = mx.model.FeedForward.load(prefix, num_round, ctx=mx.gpu(), numpy_batch_size=batch_size)
        else:
            self.model = mx.model.FeedForward.load(prefix, num_round, ctx=mx.cpu(), numpy_batch_size=batch_size)

        self.mean_img = self.mean_img = np.ones([3,224,224])*117.0

        self.crop_center = crop_center

        # Load synset (text label)
        self.synset = [l.strip() for l in open(ROOT_PATH+'synset.txt').readlines()]

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
        # we crop image from center
        if self.crop_center:
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
        return normed_img

    def classify_image(self, img, preprocess=True, N=5):
        start = time.time()
        if preprocess:
            img = self.preprocess_image(img)
        # Get prediction probability of 1000 classes from model
        prob = self.model.predict(img)[0]
        end   = time.time()
        # Argsort, get prediction index from largest prob to lowest
        pred = np.argsort(prob)[::-1]
        # Get topN label
        topN = [self.synset[pred[i]] for i in range(N)]
        topN = [top_str[top_str.find(' ')::].split(',')[0] for top_str in topN]
        topN_probs = prob[pred[0:N]]
        topN = zip(topN, pred[0:N])
        topN = [ topN[c] + (topN_prob,) for c,topN_prob in enumerate(topN_probs)]
        #print "WQETQWETQWERQWERQWERQWER", prob[pred[0:5]]
        return '%.3f' % (end - start), topN

    def feature_extraction(self, img):
        query_img = self.preprocess_image(img)
        query_feat = self.feature_extractor.predict(query_img)
        return query_feat

    def produce_cam(self, img, class_id=None, top=-1):

        # Create CAM model outputs Global Average Pooling layer
        internals = self.model.symbol.get_internals()
        fea_symbol = internals['ch_concat_5b_chconcat_output']
        CAM = mx.model.FeedForward(ctx=mx.gpu(), symbol=fea_symbol, numpy_batch_size=1,
                                             arg_params=self.model.arg_params, aux_params=self.model.aux_params,
                                             allow_extra_params=True)
        if type(img) is str or type(img) is unicode:
            img = io.imread(img)

        # Give image, preprocess and get GAP activations
        batch = self.preprocess_image(img)
        GAP = CAM.predict(batch).squeeze()
        # Get fc layer weights
        W = self.model.arg_params['fc1_weight'].asnumpy()
        # Get class specific weights
        class_W = W[class_id]
        # Abs of class specific weights
        class_W_abs = np.abs(class_W)
        # Create empty GAP
        class_GAP = np.zeros(GAP.shape)
        if top > 0 :
            top_idxs = (-class_W_abs).argsort()[0:top]
            for count,idx in enumerate(top_idxs):
                class_GAP[idx] = GAP[idx] * class_W_abs[idx]
        else:
            for count,w in enumerate(class_W_abs):
                class_GAP[count] = GAP[count] * w

        # Create CAM
        # Find average GAP*W and normalize to 0,1 scale
        CAM = class_GAP.sum(axis=0)
        CAM = CAM + -1*CAM.min()
        CAM = CAM / CAM.max()
        assert CAM.min() == 0
        assert CAM.max() == 1

        # Resize to image size
        CAM_resized = transform.resize(CAM, (img.shape[0], img.shape[1]), )

        # plt.figure()
        # plt.title(synset[class_id])
        # plt.imshow(CAM_resized,cmap='cubehelix')
        # plt.show()
        return CAM_resized



class LystInception(object):
    """
    Interface for clothes model backed by MxNet trained by Pinterest images.
    It supports 21 clothing types
    """
    def __init__(self, gpu_mode, crop_center=False, batch_size = 1):
        """
        Parameters
        ----------
        gpu_mode : bool
            If True model runs on GPU
        crop_center : bool, optional
            if True, model crops the image center by resizing the image regarding
            shortest side.
        """
        sys.path.insert(0, '/media/eightbit/data_hdd/Libs/mxnet/python')


        ROOT_PATH = config.NN_MODELS_ROOT_PATH+'Models/MxNet/LystInception/'

        # Load the pre-trained model
        prefix = ROOT_PATH+"inception"
        num_round = 52
        if gpu_mode:
            self.model = mx.model.FeedForward.load(prefix, num_round, ctx=mx.gpu(), numpy_batch_size=batch_size)
        else:
            self.model = mx.model.FeedForward.load(prefix, num_round, ctx=mx.cpu(), numpy_batch_size=batch_size)

        # Load mean file
        self.mean_img = np.ones([3,224,224])*127.0

        self.crop_center = crop_center

        # Load synset (text label)
        self.synset = [l.split()[1] for l in open(ROOT_PATH+'synset.txt').readlines()]


    def preprocess_image(self, img, show_img=False):
        # load image
        if type(img) == str:
            img = io.imread(img)

        if self.crop_center:
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
        # print img.mean()
        return normed_img

    def classify_image(self, img, N=5):
        start = time.time()
        img = self.preprocess_image(img)
        # Get prediction probability of 1000 classes from model
        prob = self.model.predict(img)[0]
        end   = time.time()
        # Argsort, get prediction index from largest prob to lowest
        pred = np.argsort(prob)[::-1]
        # Get topN label
        topN = [self.synset[pred[i]] for i in range(N)]
        topN_probs = prob[pred[0:N]]
        topN = zip(topN, pred[0:N])
        topN = [ topN[c] + (topN_prob,) for c,topN_prob in enumerate(topN_probs)]
        return '%.3f' % (end - start), topN


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
        sys.path.insert(0,'/media/eightbit/data_hdd/Libs/caffe_bundle/caffe/python')
        import caffe

        MODEL_DEPLOY  = config.NN_MODELS_ROOT_PATH+'Models/Caffe/Places2/deploy.prototxt'
        MODEL_BINARY  = config.NN_MODELS_ROOT_PATH+'Models/Caffe/Places2/inception_bn_aug_iter_60000.caffemodel'
        CATEGORY_FILE = config.NN_MODELS_ROOT_PATH+'Models/Caffe/Places2/synset.txt'

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

    def classify_image(self, image, N=5):
            starttime = time.time()
            self.net.blobs['data'].data[...] = self.transformer.preprocess('data', image/255.0)
            scores = self.net.forward()
            scores = self.net.blobs['prob'].data[0].flatten()
            endtime = time.time()

            indices = (-self.net.blobs['prob'].data[0].flatten()).argsort()
            predictions = self.labels[indices[0:N]]
            scores = ["%.2f" % score for score in scores[indices[0:N]]]
            bet_result = zip(predictions, indices[:N])
            bet_result = [ bet + (scores[c],) for c,bet in enumerate(bet_result)]
            return '%.3f' % (endtime - starttime), bet_result


class Color43InceptionCam(object):
    def __init__(self, gpu_mode, crop_center=False, batch_size=1):
        """
        Parameters
        ----------
        gpu_mode : bool
            If True model runs on GPU
        crop_center : bool, optional
            if True, model crops the image center by resizing the image regarding
            shortest side.
        """

        ROOT_PATH = config.NN_MODELS_ROOT_PATH+'Models/MxNet/WateringColorInceptionCam/'

        # Load the pre-trained model
        prefix = ROOT_PATH+"inception-cam"
        num_round = 17
        if gpu_mode:
            self.model = mx.model.FeedForward.load(prefix, num_round, ctx=mx.gpu(), numpy_batch_size=batch_size)
        else:
            self.model = mx.model.FeedForward.load(prefix, num_round, ctx=mx.cpu(), numpy_batch_size=batch_size)

        self.mean_img = self.mean_img = np.ones([3,224,224])*127.0

        self.crop_center = crop_center

        # Load synset (text label)
        self.synset = [l.strip().replace('+', ' ') for l in open(ROOT_PATH+'synset.txt').readlines()]


    def preprocess_image(self, img, show_img=False):
        # load image
        if type(img) == str:
            img = io.imread(img)
        # we crop image from center
        if self.crop_center:
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
        return normed_img

    def classify_image(self, img, preprocess=True, N=5):
        start = time.time()
        if preprocess:
            img = self.preprocess_image(img)
        # Get prediction probability of 1000 classes from model
        prob = self.model.predict(img)[0]
        end   = time.time()
        # Argsort, get prediction index from largest prob to lowest
        pred = np.argsort(prob)[::-1]
        # Get topN label
        topN = [self.synset[pred[i]] for i in range(N)]
        topN_probs = prob[pred[0:N]]
        topN = zip(topN, pred[0:N])
        topN = [ topN[c] + (topN_prob,) for c,topN_prob in enumerate(topN_probs)]
        return '%.3f' % (end - start), topN

    def produce_cam(self, img, class_id=None, top=-1):

        # Create CAM model outputs Global Average Pooling layer
        internals = self.model.symbol.get_internals()
        fea_symbol = internals['relu_conv_cam_output']
        CAM = mx.model.FeedForward(ctx=mx.gpu(), symbol=fea_symbol, numpy_batch_size=1,
                                             arg_params=self.model.arg_params, aux_params=self.model.aux_params,
                                             allow_extra_params=True)
        if type(img) is str or type(img) is unicode:
            img = io.imread(img)

        # Give image, preprocess and get GAP activations
        batch = self.preprocess_image(img)
        GAP = CAM.predict(batch).squeeze()
        # Get fc layer weights
        W = self.model.arg_params['fc1_weight'].asnumpy()
        # Get class specific weights
        class_W = W[class_id]
        # Create empty GAP
        class_GAP = np.zeros(GAP.shape)
        if top > 0 :
            top_idxs = (-class_W).argsort()[0:top]
            for count,idx in enumerate(top_idxs):
                class_GAP[idx] = GAP[idx] * class_W[idx]
        else:
            for count,w in enumerate(class_W):
                class_GAP[count] = GAP[count] * w

        # Create CAM
        # Find average GAP*W and normalize to 0,1 scale
        CAM = class_GAP.sum(axis=0)
        CAM = CAM + -1*CAM.min() # or CAM = CAM[CAM<0] for more sparse mask inly centric to interest regions

        CAM = CAM / CAM.max()
        print CAM.min()
        assert CAM.min() >= 0
        assert CAM.max() <= 1

        # Resize to image size
        CAM_resized = transform.resize(CAM, (img.shape[0], img.shape[1]), )

        # plt.figure()
        # plt.title(synset[class_id])
        # plt.imshow(CAM_resized,cmap='cubehelix')
        # plt.show()
        return CAM_resized

class CarsInception(object):
    """
        car classifier for 52 different brand including NotCar and Others class for
        rarely shown car synset
    """
    def __init__(self, gpu_mode, crop_center=False, is_retrieval=False):
        ROOT_PATH = config.NN_MODELS_ROOT_PATH+'Models/MxNet/CarsInception/'

        # Load the pre-trained model
        prefix = ROOT_PATH+"inception-0"
        num_round = 80
        if gpu_mode:
            self.model = mx.model.FeedForward.load(prefix, num_round, ctx=mx.gpu(), numpy_batch_size=1)
        else:
            self.model = mx.model.FeedForward.load(prefix, num_round, ctx=mx.cpu(), numpy_batch_size=1)

        # Load mean file
        self.mean_img = np.ones([3,224,224])*127.0

        # Crop center flag
        self.crop_center = crop_center

        # Load synset (text label)
        self.synset = [l for l in open(ROOT_PATH+'synset.txt').readlines()]

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


    def preprocess_image(self, img, crop_center=False, show_img=False):
        # load image
        if type(img) == str:
            img = io.imread(img)
        #print("Original Image Shape: ", img.shape)
        # we crop image from center
        if crop_center:
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
        return normed_img

    def classify_image(self, img, N=5):
        start = time.time()
        img = self.preprocess_image(img, self.crop_center)
        # Get prediction probability of 1000 classes from model
        prob = self.model.predict(img)[0]
        end   = time.time()
        # Argsort, get prediction index from largest prob to lowest
        pred = np.argsort(prob)[::-1]
        # Get topN label
        topN = [self.synset[pred[i]] for i in range(N)]
        topN_probs = prob[pred[0:N]]
        topN = zip(topN, pred[0:N])
        topN = [ topN[c] + (topN_prob,) for c,topN_prob in enumerate(topN_probs)]
        return '%.3f' % (end - start), topN

    def feature_extraction(self, img):
        query_img = self.preprocess_image(img, self.crop_center)
        query_feat = self.feature_extractor.predict(query_img)
        return query_feat

    def produce_cam(self, img, class_id=None, top=-1):

        # Create CAM model outputs Global Average Pooling layer
        internals = self.model.symbol.get_internals()
        fea_symbol = internals['ch_concat_5b_chconcat_output']
        CAM = mx.model.FeedForward(ctx=mx.gpu(), symbol=fea_symbol, numpy_batch_size=1,
                                             arg_params=self.model.arg_params, aux_params=self.model.aux_params,
                                             allow_extra_params=True)
        if type(img) is str or type(img) is unicode:
            img = io.imread(img)

        # Give image, preprocess and get GAP activations
        batch = self.preprocess_image(img)
        GAP = CAM.predict(batch).squeeze()
        # Get fc layer weights
        W = self.model.arg_params['fc1_weight'].asnumpy()
        # Get class specific weights
        class_W = W[class_id]
        # Abs of class specific weights
        class_W_abs = np.abs(class_W)
        # Create empty GAP
        class_GAP = np.zeros(GAP.shape)
        if top > 0 :
            top_idxs = (-class_W_abs).argsort()[0:top]
            for count,idx in enumerate(top_idxs):
                class_GAP[idx] = GAP[idx] * class_W_abs[idx]
        else:
            for count,w in enumerate(class_W_abs):
                class_GAP[count] = GAP[count] * w

        # Create CAM
        # Find average GAP*W and normalize to 0,1 scale
        CAM = class_GAP.sum(axis=0)
        CAM = CAM + -1*CAM.min()
        CAM = CAM / CAM.max()
        assert CAM.min() == 0
        assert CAM.max() == 1

        # Resize to image size
        CAM_resized = transform.resize(CAM, (img.shape[0], img.shape[1]), )

        # plt.figure()
        # plt.title(synset[class_id])
        # plt.imshow(CAM_resized,cmap='cubehelix')
        # plt.show()
        return CAM_resized


class LocSegNetwork(object):
    '''
        This network borrowed from https://github.com/xiaolonw/nips14_loc_seg_testonly
        It first localizes the content then tries to segment it pixel-wise.
        Results are not perfect but still useful.
    '''
    def __init__(self, gpu_mode=1, is_loc=1, is_seg=1):
        sys.path.insert(0,'/media/eightbit/data_hdd/Libs/caffe_bundle/caffe/python')
        import caffe

        ROOT_PATH = config.NN_MODELS_ROOT_PATH + "Models/Caffe/LocSegModel/"
        # Load mean image
        mean_img = np.load('/media/eightbit/data_hdd/Libs/nips14_loc_seg_testonly/Caffe_Segmentation/segscripts/models/mean.npy')
        mean_img = mean_img[:, 14:241,14:241 ]
        self.mean_img = mean_img

        if gpu_mode:
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()

        if is_loc:
            # Localization model
            self.loc_net = caffe.Net(ROOT_PATH+'loc/imagenet_test_mem.prototxt',
                        ROOT_PATH+'loc.caffemodel',
                        caffe.TEST)

            # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
            self.loc_transformer = caffe.io.Transformer({'data': self.loc_net.blobs['data'].data.shape})
            self.loc_transformer.set_transpose('data', (2,0,1))
            self.loc_transformer.set_mean('data', self.mean_img) # mean pixel
            self.loc_transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
            self.loc_transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
            self.loc_net.blobs['data'].reshape(1,3,227,227)

        if is_seg:
            # Segmentation model
            self.seg_net = caffe.Net(ROOT_PATH+'seg/seg_test_mem.prototxt',
                            ROOT_PATH+'seg.caffemodel',
                            caffe.TEST)

            # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
            self.seg_transformer = caffe.io.Transformer({'data': self.seg_net.blobs['data'].data.shape})
            self.seg_transformer.set_transpose('data', (2,0,1))
            self.seg_transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
            self.seg_transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
            self.seg_net.blobs['data'].reshape(1,3,55,55)

    def localize(self,img, show_result=0):
        """
            returns bounding box for the object of interest
            with regard to 256x256 image size. If the target image
            has other size values, bbox coordinates should be checked
            regardingly.

            Inputs:
                img - target color image given in any size
                show_result - show the result bbox on the image for debug

            Outputs:
                bbox - bbox coordinates (x1,y1,x2,y2)
                loc_img - localized region
                img_256 - reference image for bbox coordinates
        """
        if type(img) is str or type(img) is unicode:
            img = io.imread(img)
        # preprocess img
        img_256 = transform.resize(img,[256,256,3])
        img = img_256[14:241,14:241, :]
        img = self.loc_transformer.preprocess('data', img)
        # give image to network
        self.loc_net.blobs['data'].data[...] = img
        # feedforward img
        out = self.loc_net.forward()
        # Set bbox predictions by the border limits
        bbox = out['fc8_loc'][0].astype(int)
        bbox[bbox<0] = 0
        bbox[bbox>=256] = 255
        # show results
        if show_result:
            print bbox
            plt.figure()
            plt.plot(np.array([bbox[0],bbox[0], bbox[2], bbox[2], bbox[0]]), np.array([bbox[1], bbox[3], bbox[3], bbox[1], bbox[1]]))
            plt.imshow(img_256)
            plt.show()
        # create final localized region
        loc_img = img_256[bbox[1]:bbox[3], bbox[0]:bbox[2], ]
        return bbox, loc_img, img_256


    def segment(self, img, show_result=1, bbox=None):
        """
            Segments the region of interest. Given image is resized to 55x55 and
            the resulting segmentation mask has size 50x50. So the segmentation
            mask should be resized for any third party use.

            Inputs:
                img - target color image given in any size
                show_result - show the result bbox on the image for debug
                bbox - bounding box coordinates to consider

            Outputs:
                seg_resized - segmenation mask in image size
                img_seg - segmented image
        """
        if type(img) is str or type(img) is unicode:
            img_org = io.imread(img)
        else:
            img_org = img
        if bbox is not None:
            img_crop = img_org[bbox[1]:bbox[3], bbox[0]:bbox[2], ]
        else:
            img_crop = img_org
        img_crop = transform.resize(img_crop, [55,55])
        # set the image mean and resize it to 55x55
        if bbox is not None:
            mean_crop = self.mean_img[:, bbox[1]:bbox[3], bbox[0]:bbox[2]]
        else:
            mean_crop = self.mean_img
        mean_crop = transform.resize(mean_crop, [3,55,55])
        self.seg_transformer.set_mean('data', mean_crop) # mean pixel
        # Feedforward network
        self.seg_net.blobs['data'].data[...] = self.seg_transformer.preprocess('data', img_crop)
        out = self.seg_net.forward()
        # Fetch result
        seg = out['fc8_seg']
        seg = np.fliplr(seg.reshape([50,50]))
        # create segmented img
        seg_resized = np.zeros(img_org.shape[:2])
        if bbox is not None:
            seg_resized[bbox[1]:bbox[3], bbox[0]:bbox[2]] += transform.resize(seg,[bbox[3]-bbox[1], bbox[2]-bbox[0]])
        else:
            seg_resized += transform.resize(seg,img_org.shape[:2])
        # seg_resized[seg_resized > 0] = 1
        img_seg = img_org.copy()
        img_seg[:,:,0] *= seg_resized
        img_seg[:,:,1] *= seg_resized
        img_seg[:,:,2] *= seg_resized
        # show result
        if show_result:
            plt.figure()
            plt.imshow(img_seg)
            plt.show()
        return seg_resized, img_seg

    def localize_and_segment(self, img, crop=0, show_result=1):
        """
            Apply localization and segmentation networks in oreder.

            Inputs:
                img - target color image given in any size
                crop - crop segmentated region
                show_result - show the result bbox on the image for debug

            Outputs:
                seg_mask - segmenation mask in image size
                img_seg - segmented image
        """
        bbox, loc_img, img_256 = self.localize(img, show_result)
        seg_mask, img_seg = self.segment(img_256, show_result, bbox)
        if crop:
            # crop segment out regions from image
            seg_mask[seg_mask<seg_mask.mean()] = 0
            y_idxs,x_idxs = seg_mask.nonzero()
            left = x_idxs.min()
            right = x_idxs.max()
            top = y_idxs.min()
            bot = y_idxs.max()
            img_seg = img_seg[top:bot, left:right, :]
        return seg_mask, img_seg

class TorchModel(object):
    '''
    Parent class for all torch Models
    '''

    def __init__(self, gpu_mode=0, model_path=None, model_file_name=None):
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

        ROOT_PATH = config.NN_MODELS_ROOT_PATH + model_path
        self.gpu_mode = gpu_mode

        # Load the pre-trained model
        if model_file_name ==  None:
            model_path = os.path.join(ROOT_PATH,"model_cpu.t7")
        else:
            model_path = os.path.join(ROOT_PATH, model_file_name)
        self.model = torch.load(model_path)
        self.model._add(nn.SoftMax())

        if gpu_mode:
            self.model._cuda()

        self.model._evaluate()

        # Load mean file
        self.mean = np.array([ 0.485,  0.456,  0.406])
        self.std  = np.array([ 0.229,  0.224,  0.225])

        # Load synset (text label)
        self.synset = [l.split(',')[0].strip() for l in open(ROOT_PATH+'synset.txt').readlines()]

    def preprocess_image(self, img):
        if type(img) is str or type(img) is unicode:
            img = io.imread(img)
        else:
            img = img

        # resize image by shortest edge
        img_resized = thumbnail(img, 224)/ float(255)

        # color normalization
        img_norm = img_resized - self.mean
        img_norm /= self.std

        # center cropping
        img_crop = crop_center(img_norm)

        # format img dimensions
        img_crop = img_crop.transpose([2,0,1])[None,:]

        assert img_crop.ndim == 4

        # pass data to torch and convert douple to float
        x = torch.fromNumpyArray(img_crop)
        x = x._float()
        return x

    def classify_image(self, img, N=2):
        start = time.time()
        img = self.preprocess_image(img)
        # Get prediction probability of 1000 classes from model
        if self.gpu_mode:
            prob = self.model._forward(img._cuda())
        else:
            prob = self.model._forward(img)
        prob = prob.asNumpyArray()[0]
        end   = time.time()
        # Argsort, get prediction index from largest prob to lowest
        pred = np.argsort(prob)[::-1]
        # Get topN label
        topN = [self.synset[pred[i]] for i in range(N)]
        topN_probs = prob[pred[0:N]]
        topN = zip(topN, pred[0:N])
        topN = [ topN[c] + (topN_prob,) for c,topN_prob in enumerate(topN_probs)]
        #print "WQETQWETQWERQWERQWERQWER", prob[pred[0:5]]
        return '%.3f' % (end - start), topN

class NsfwResnetTorch(TorchModel):
    def __init__(self, gpu_mode=0):
        super(NsfwResnetTorch, self).__init__(gpu_mode, model_path='Models/Torch/NSFW_resnet/')

class ImageNetResnetTorch(TorchModel):
    def __init__(self, gpu_mode=0):
        super(ImageNetResnetTorch, self).__init__(gpu_mode, model_path='Models/Torch/ImageNetResNet/', model_file_name='resnet-101_cpu.t7')

class FoodResnetTorch(TorchModel):
    def __init__(self, gpu_mode=0):
        super(FoodResnetTorch, self).__init__(gpu_mode, model_path='Models/Torch/Food/')
