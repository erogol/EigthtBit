from ..utils.img_utils import create_img_grids
from ..utils.io_utils import read_img
import cv2
import numpy as np
import sys
from matplotlib import pylab as plt

class ColorHist(object):
    def __init__(self, color_space = 'RGB', nbins = [6,6,6], ngrids = [2,2], center= True, mask = None, pre_pipeline=None):
        """
            color_space = HSV or RGB
            nbins       = number of bins for each channel. Final hist will be bin1*bin2*bin3
            ngrids      = number of spatial grids
            center      = Take an additional center grid
            mask        = Image mask to compute values only from active regions. Keep it None is no mask
            pre_pipeline = image preprocessing pipeline
        """
        self.color_space    = color_space
        self.nbins          = nbins
        self.mask           = mask
        self.ngrids         = ngrids
        self.center         = center
        self.pre_pipeline   = pre_pipeline

    def compute_color_hist(self, img):
        """
            Compute 1D RGB value histogram of the given image.
            Final histogram is the concatanation of the all color
            channel histograms
            
            nbins : number of bins for each color channel
        """

        if self.color_space == 'RGB':
            #hist_B = cv2.calcHist(img,[0], None, [self.nbins[0]], [0, 256]).flatten()
            #hist_G = cv2.calcHist(img,[1], None, [self.nbins[1]], [0, 256]).flatten()
            #hist_R = cv2.calcHist(img,[2], None, [self.nbins[2]], [0, 256]).flatten()
            #colorVec = np.hstack([hist_B, hist_G, hist_R])
            #colorvec = cv2.normalize(colorVec)
            
            colorVec = cv2.calcHist([img], [0, 1, 2], None, self.nbins, [0, 256, 0, 256, 0, 256])
            colorVec = cv2.normalize(colorVec).flatten()
        elif self.color_space == 'HSV':
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            hs_hist= cv2.calcHist(img_hsv, [0,1], None, [self.nbins[0],self.nbins[1]], [0, 180, 0, 256])
            hs_hist = hs_hist.flatten()
            v_hist = cv2.calcHist(img_hsv, [2], None, [self.nbins[2]], [0,256])
            v_hist = v_hist.flatten()
            colorVec = np.hstack([hs_hist, v_hist])
            
            #colorVec = cv2.calcHist(img_hsv, [0,1,2], None, self.nbins, [0, 180, 0, 256, 0, 256])
            colorVec = cv2.normalize(colorVec).flatten()
        else:
            print 'Wrond Color Space Parameter! Should be RGB or HSV'
            sys.exit(0)
        return colorVec

    def transform(self, img_path):
        # Preprocessing expects to have a final img or img and a mask
        if type(img_path) == str:
            img = read_img(img_path)
        else:
            img = img_path
        if self.pre_pipeline != None:
            img = self.pre_pipeline.transform(img)
            
            if type(img) is list or type(img) is tuple:
                #self.mask = img[1][0][1][0]
                img = img[0]

        img_feat = []
        #plt.figure()
        #plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        img_grids = create_img_grids(img, num_grids=self.ngrids, center=self.center)
        #if self.mask is not None:
        #    mask_grids = create_img_grids(self.mask , num_grids=self.ngrids, center=self.center)
        count = 0
        for img_grid in img_grids:
            #if self.mask is not None:
             #   self.mask = mask_grids[count]
             #   img_feat.append(self.compute_color_hist(img_grid))
            #else:
            #plt.figure()
            #plt.imshow(cv2.cvtColor(img_grid, cv2.COLOR_BGR2RGB))
            img_feat.append(self.compute_color_hist(img_grid))
            count += 1
        feat_vec = np.array(img_feat).flatten().astype(np.float32)
        feat_vec = feat_vec / feat_vec.sum()
        return feat_vec

    def save_obj(self, out_path):
        import dill
        f = open(out_path, 'wb')
        dill.dump(self.ngrids, f)
        f.close()


    def load_obj(self, in_path):
        import dill
        self = dill.loads(in_path)