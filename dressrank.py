# DressRank request handler by Sermetcan Baysal
# dressrank.py All rights reserved. 2014.

"""
Created on Fri Sep 12 14:38:14 2014
@author: DressRank - Sermetcan Baysal
@desc: Script to read images and their metadata from local path and insert them into DressRankDB
"""

import os
from flask import Flask, request, redirect, url_for, jsonify
from werkzeug import secure_filename
from flask.ext.mongoengine import MongoEngine
from Dress import Dress
import numpy as np
from scipy import misc, spatial
import time

import Image
import scipy.cluster

from bow.Bow import Bow
from feature.ColorHist import ColorHist
from img_processing.Saliency import *
from utils.img_utils import img_resize
from retrieval.DistFuncs import *


UPLOAD_FOLDER = '/var/www/dressrank/uploads/'
#UPLOAD_FOLDER = '/Users/sermetcan/Dropbox/Documents/dressrank_code/uploads'
ALLOWED_EXTENSIONS = set([ 'png', 'jpg', 'jpeg' ])

# Applicaton
dressrank = Flask(__name__)
dressrank.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
dressrank.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024
dressrank.config['MONGODB_DB'] = 'DressRankDB'
dressrank.config['MONGODB_HOST'] = 'localhost'
dressrank.config['MONGODB_PORT'] = 27017
db = MongoEngine(dressrank)

# method to check for file type
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

# method to handle query (file upload)
@dressrank.route('/query/', methods=['GET', 'POST'])
def query():
    if request.method == 'POST':
        
        start_time = time.time()            
            
        # get file         
        file = request.files['file']
        if file and allowed_file(file.filename):
            
            # save file            
            filename = secure_filename(file.filename)
            file_path = os.path.join(dressrank.config['UPLOAD_FOLDER'], filename);
            file.save(file_path)
            
            ########### FEATURE EXTRACTION ############

            img_path = file_path
           
            # COLOR FATURE EXTRACTION
            sm = SaliencyMask()
            operations = [('img_resize', img_resize), ('sal_mask', sm.transform)]
            args_list = [{'h_size':258}, {'cropped':True}]
            pre_pipeline = Pipeline(ops=operations, arg_list=args_list)
            
            ch = ColorHist('RGB', [6,6,6], [2,2], center=True, pre_pipeline = pre_pipeline)
            colorVec = ch.transform(img_path)


            # SURF BAG of WORDS
            sm = SaliencyMask()
            operations = [('img_resize', img_resize), ('sal_mask', sm.transform)]
            args_list = [{'h_size':500}, {'cropped':True}]
            pre_pipeline = Pipeline(ops=operations, arg_list=args_list)
            
            bow = Bow(pre_pipeline = pre_pipeline)
            bow.load_data(PATH_TO_BOW_VOCAB)
            surfVec = bow.transform(img_path)

            ########### EDN OF FEATURE EXTRACTION #############

            ########### COMPARE ##########
            DIST_FUNC = bhattacharyya
            color_distance = []
            surf_distance = []

            # compare by color with db
            allDresses = Dress.objects( dcategory = request.form['category'] )
            for d in allDresses:
                color_distance.append( spatial.distance.pdist( d.dColorFeat, colorVec, DIST_FUNC ) )
                surf_distance.append( spatial.distance.pdist( d.dSurfFeat, surfVec, DIST_FUNC ) )
            # rank clothes wrt to color
            whole_distance = np.multiply(color_distance, surf_distance)
            sorted_idx = sorted( range( len( whole_distance ) ), key=lambda k: whole_distance[k] )
            # return results
            ranklist = []
            for i in range( 0, 10 ):
                ranklist.append( { 'id' : str( allDresses[sorted_idx[i]].id ),
                                    'brand' : allDresses[sorted_idx[i]].dbrand,
                                    'price' : allDresses[sorted_idx[i]].dprice } )
            print 'request served in ' + str( time.time() - start_time )
            return jsonify(results=ranklist)
        else:
            return '412 Precondition Failed. Only png, jpg, jpeg files of size at most 2MB accepted'
            
    return '''<!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
	<br>
<input type="text" name="category">
         <input type=submit value=Upload>
    </form>
    '''		

# Execute DressRank server
if __name__ == '__main__':
    dressrank.run(port=8000, debug=True)
