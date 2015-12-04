"""
Created on Fri Sep 12 14:38:14 2014
@author: DressRank - Sermetcan Baysal
@desc: Script to read images and their metadata from local path and insert them into DressRankDB
"""

import os
import Image
import numpy as np
from scipy import misc
from Dress import Dress
from mongoengine import *
from random import randint
from bson.objectid import ObjectId
from bow.Bow import Bow
from feature.ColorHist import ColorHist
from img_processing.Saliency import *
from utils.img_utils import img_resize
from utils.data_utils import get_data_paths
from pipeline import Pipeline
from progressbar import ProgressBar
from pathos.multiprocessing import ProcessingPool


def insert_category_to_DB_helper(img_path, category, gender, imageRootPath):
    PATH_TO_BOW_OBJ = 'DressrankRetrieval/data/bow_vocab.pkl'

    connect('DressRankDB', host='54.149.5.208', port=27017)  
    
    # Full paths
    path = os.path.join(os.path.join(imageRootPath, gender), category)
    fileNameBase = os.path.basename(img_path).split('.')[0]  
    print fileNameBase           

   ########### FEATURE EXTRACTION ############
   
    # COLOR FATURE EXTRACTION
    sm = SaliencyMask()
    operations = [('img_resize', img_resize), ('sal_mask', sm.transform)]
    args_list = [{'h_size':258}, {'cropped':True}]
    pre_pipeline = Pipeline(ops=operations, arg_list=args_list)
    
    ch = ColorHist('RGB', [8,8,8], [0,0], center=True, pre_pipeline = pre_pipeline)
    colorVec = ch.transform(img_path)
    
    print 'ColorVec shape:', colorVec.shape
    print 'ColorVec sum :', colorVec.sum()

    # SURF FEATS
    bow = load_obj(PATH_TO_BOW_OBJ)    
    surfVec = bow.transform(img_path)
    
    print 'surfVec shape:', surfVec.shape
    print 'surfVec sum :', surfVec.sum()

   ########### EDN OF FEATURE EXTRACTION #############

    colorVec = list( colorVec )
    surfVec = list(surfVec)

    # 4. Check if object already exists in DB
    try:
        if ObjectId.is_valid( fileNameBase ) == True and len(Dress.objects( dcategory = category, id = fileNameBase )) > 0:   
            
            # 4.1 Update metadata and features of existing object in DB
            #Dress.objects( dcategory = category, id = fileNameBase ).update( set__dbrand = metadata[0], set__dgender = metadata[1], set__dcategory = metadata[2], set__durl = metadata[3], set__dvFeat = featVec, set__dvAutoEncFeat = featVec, set__dvColorFeat = colorVec )
            Dress.objects( dcategory = category, id = fileNameBase ).update( set__dbrand = 'My Brand', set__dgender = gender, set__dprice = randint(50,150), set__dcategory = category, set__dColorFeat = colorVec, set__dSurfFeat = surfVec )
            print img_path + ' successfully updated in DressRankDB with id ' + fileNameBase    
            return
    except ValidationError:
        print 'Validation Error:', img_path
    # Insert new image metadata and features to DB
    #dress = Dress( dbrand = metadata[0], dgender = metadata[1], dcategory = metadata[2], durl = metadata[3], dvFeat = featVec, dvAutoEncFeat = featVec, dvColorFeat = colorVec )
    dress = Dress( dbrand = 'My Brand', dgender = gender, dprice = randint(50,150), dcategory = category, dColorFeat = colorVec, dSurfFeat = surfVec )   
    dress.save()
    
    # 4.2.2 Rename file and its metadata in local directory
    #os.rename( path + fileMetaData, path + dress.id.__str__() + '.txt' )  
    path_prefix = os.path.join(os.path.join(imageRootPath, gender),category)           
    os.rename( path_prefix+'/'+fileNameBase + '.jpg', path_prefix +'/'+ dress.id.__str__() + '.jpg' )
    #os.rename( path + fileNameBase + '_feat.txt', path + dress.id.__str__() + '_feat.txt' )
    
    print img_path + ' successfully inserted to DressRankDB with id ' + dress.id.__str__()   
         
def insert_category_to_DB(category, gender, imageRootPath):
    # Connect to MongoDB
    data_path = os.path.join(os.path.join(imageRootPath,gender), category)
    print data_path
    img_paths = get_data_paths(data_path, ext='*.jpg')[0]
    # Traverse images and their metedata in given directory
    ProcessingPool().map(insert_category_to_DB_helper, img_paths, [category]*len(img_paths), 
                                                                  [gender]*len(img_paths), 
                                                                 [imageRootPath]*len(img_paths))
    #insert_category_to_DB_helper(img_paths[0], category, gender, imageRootPath)
        