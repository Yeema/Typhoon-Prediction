from __future__ import print_function

import numpy as np
import scipy.spatial.distance as distance

def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats):
    ###########################################################################
    # TODO:                                                                   #
    # This function will predict the category for every test image by finding #
    # the training image with most similar features. Instead of 1 nearest     #
    # neighbor, you can vote based on k nearest neighbors which will increase #
    # performance (although you need to pick a reasonable value for k).       #
    ###########################################################################
    ###########################################################################
    # NOTE: Some useful functions                                             #
    # distance.cdist :                                                        #
    #   This function will calculate the distance between two list of features#
    #       e.g. distance.cdist(? ?)                                          #
    ###########################################################################
    '''
    Input : 
        train_image_feats : 
            image_feats is an (N, d) matrix, where d is the 
            dimensionality of the feature representation.

        train_labels : 
            image_feats is a list of string, each string
            indicate the ground truth category for each training image. 

        test_image_feats : 
            image_feats is an (M, d) matrix, where d is the 
            dimensionality of the feature representation.
    Output :
        test_predicts : 
            a list(M) of string, each string indicate the predict
            category for each testing image.
    '''
    k_num = 50
    disc_of_dist={}
    predicts = []
    distance_array = []

    test_predicts = []
    train_image_feats = np.array(train_image_feats)
    test_image_feats = np.array(test_image_feats)
    for i in train_image_feats:
        for j in test_image_feats:
            distance_array.append(np.sum((i-j)**2)**(1/2))
        for k in range(k_num):
            min_index = np.argmin(distance_array)
            predicts.append(train_labels[min_index])
            distance_array[min_index] = 1000000000
        labels_uni,counts = np.unique(predicts,return_counts = True)
        test_predicts.append(labels_uni[np.argmax(counts)])
        predicts = []
        distance_array = []


    #############################################################################
    #                                END OF YOUR CODE                           #
    #############################################################################
    return test_predicts
