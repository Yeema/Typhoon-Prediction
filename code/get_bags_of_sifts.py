from PIL import Image
import numpy as np
from scipy.spatial import distance
import pickle
import scipy.spatial.distance as distance
from cyvlfeat.sift.dsift import dsift
from time import time
import pdb

def get_bags_of_sifts(image_paths):
    ############################################################################
    # TODO:                                                                    #
    # This function assumes that 'vocab.pkl' exists and contains an N x 128    #
    # matrix 'vocab' where each row is a kmeans centroid or visual word. This  #
    # matrix is saved to disk rather than passed in a parameter to avoid       #
    # recomputing the vocabulary every time at significant expense.            #
                                                                    
    # image_feats is an N x d matrix, where d is the dimensionality of the     #
    # feature representation. In this case, d will equal the number of clusters#
    # or equivalently the number of entries in each image's histogram.         #
    
    # You will want to construct SIFT features here in the same way you        #
    # did in build_vocabulary.m (except for possibly changing the sampling     #
    # rate) and then assign each local feature to its nearest cluster center   #
    # and build a histogram indicating how many times each cluster was used.   #
    # Don't forget to normalize the histogram, or else a larger image with more#
    # SIFT features will look very different from a smaller version of the same#
    # image.                                                                   #
    ############################################################################
    '''
    Input : 
        image_paths : a list(N) of training images
    Output : 
        image_feats : (N, d) feature, each row represent a feature of an image
    '''
    # with open("vocab.pkl",'r') as file:
    #     vocab = pickle.load(file)
    #     image_feats = np.z
    # image_feats = []

    # for img_file in image_paths:
        
    #     image_feats.append(np.asarray(Image.open(img_file),dtype='float32').flatten().tolist())
    
    # image_feats = np.array(tiny_images_ori)
    # #將feature做mean unit length normalization
    # tiny_images_T = np.transpose(image_feats)

    # for idx, feature in enumerate(tiny_images_T):
    #     feature_scaled = preprocessing.scale(feature,with_std = False)
    #     feature_scaled = feature_scaled/np.max(abs(feature_scaled))
    #     image_feats[:,idx] = feature_scaled
    with open("vocab.pkl", "rb") as vocab_file:
        voc = pickle.load(vocab_file)
        image_feats = np.zeros((len(image_paths),len(voc)))    
    distance_array=[]
    for i in range(len(image_paths)):
        img = np.asarray(Image.open(image_paths[i]) , dtype='float32')
        _ , descriptors = dsift(img, step=[5,5], fast=True)
        dist = distance.cdist(voc, descriptors, 'euclidean')
        choose = np.argmin(dist, axis=0)
        # for j in voc:
        #     for k in descriptors:
        #         distance_array.append(np.sum((j-k)**2)**(1/2))
        #     choose = np.argmin(distance_array)
        #     distance_array=[]
        for vote in choose:
            image_feats[i,vote] += 1   



    #############################################################################
    #                                END OF YOUR CODE                           #
    #############################################################################
    return image_feats
