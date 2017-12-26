from __future__ import print_function
from random import shuffle
import os
import argparse
import pickle

from get_image_paths import get_image_paths
from get_tiny_images import get_tiny_images
from build_vocabulary import build_vocabulary
from get_bags_of_sifts import get_bags_of_sifts
from visualize import visualize

from nearest_neighbor_classify import nearest_neighbor_classify
from svm_classify import svm_classify
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import sys
import random
import bisect
import math

def choose_training_id(p):
    ids=[]
    for i in range(110):
        rand=random.uniform(0, 1)
        if rand in p:
            ids.append(p.index(rand))
        else:
            ids.append(bisect.bisect_left(p, rand))
    return ids
def update_possibility(p,pred_labels,tested_labels):
    wrong=[]
    correct=[]
    # test 90 examples by classifier trained by 10 examples
    for id,x in enumerate(zip(tested_labels,pred_labels)):
        if x[0]==x[1]:
            correct.append(id)
        else:
            wrong.append(id)
    # record the possibility of the wrongly classified examples 
    error=[p[e] for e in wrong]
    # count total error rate
    error=sum(error)
    # calculate beta
    beta=math.sqrt( abs(error/(1-error)))
    # correct examples will be multiplied by beta
    # incorrect examples will be divided by beta
    if error>1e-10:
        for e in wrong:
            p[e]=p[e]/beta
        for c in correct:
            p[e]=p[e]*beta
    # normalize sum of the possibility to be 1
    total=sum(p)
    p=[x/total for x in p]
    sum_p=[]
    # accumulate the normalized possibility
    for i in range(len(p)):
        sum_p.append(sum(p[:i+1]))
    return p,sum_p
def cal_result(testset,gt):
    # use 9 classifiers to estimate examples' type
    # if more than half of classifiers agree with 'some type'
    # then that is the final result
    acc=[]
    sum_result=0
    # print(len(testset),len(testset[0]),len(gt),len(gt[0])) 3 300 300 3
    for i in range(num_trial):
        temp=[]
        for x in zip(testset[i],gt):
            # print(x)
            if x[0]==x[1]:
                temp.append(1)
            else:
                temp.append(0)
        acc.append(temp)
    # print('acc:',len(acc),len(acc[0]),len(testset),len(testset[0]))
    for i in range(len(testset[0])):#300
        temp=0
        for j in range(len(acc)):#3
            temp+=acc[j][i]
        # print(temp,acc[0][j],acc[1][j],acc[2][j],len(acc))
        if temp>int(len(acc)/2):
            sum_result+=1
    return float(sum_result/len(gt))
# Step 0: Set up parameters, category list, and image paths.

#For this project, you will need to report performance for three
#combinations of features / classifiers. It is suggested you code them in
#this order, as well:
# 1) Tiny image features and nearest neighbor classifier
# 2) Bag of sift features and nearest neighbor classifier
# 3) Bag of sift features and linear SVM classifier
#The starter code is initialized to 'placeholder' just so that the starter
#code does not crash when run unmodified and you can get a preview of how
#results are presented.

parser = argparse.ArgumentParser()
parser.add_argument('--feature', help='feature', type=str, default='dumy_feature')
parser.add_argument('--classifier', help='classifier', type=str, default='dumy_classifier')
args = parser.parse_args()

DATA_PATH = '../data'

#This is the list of categories / directories to use. The categories are
#somewhat sorted by similarity so that the confusion matrix looks more
#structured (indoor and then urban and then rural).

CATEGORIES = ['low', 'mid', 'high']

CATE2ID = {v: k for k, v in enumerate(CATEGORIES)}

ABBR_CATEGORIES =  ['low', 'mid', 'high']


# FEATURE = args.feature
# FEATURE  = 'bag_of_sift'
FEATURE = 'tiny_image'
CLASSIFIER = 'support_vector_machine'
# CLASSIFIER = 'support_vector_machine'
#python proj3.py --feature=tiny_image --classifier=support_vector_machine
#number of training examples per category to use. Max is 100. For
#simplicity, we assume this is the number of test cases per category, as
#well.

NUM_TRAIN_PER_CAT = 100
num_all=300
num_trial=11
def main():
    #This function returns arrays containing the file path for each train
    #and test image, as well as arrays with the label of each train and
    #test image. By default all four of these arrays will be 1500 where each
    #entry is a string.
    print("Getting paths and labels for all train and test data")
    train_image_paths, test_image_paths, train_labels, test_labels = \
        get_image_paths(DATA_PATH, CATEGORIES, NUM_TRAIN_PER_CAT)

    # TODO Step 1:
    # Represent each image with the appropriate feature
    # Each function to construct features should return an N x d matrix, where
    # N is the number of paths passed to the function and d is the 
    # dimensionality of each image representation. See the starter code for
    # each function for more details.

    if FEATURE == 'tiny_image':
        # YOU CODE get_tiny_images.py
        # print('in tiny image')
        if os.path.isfile('tiny_test.pkl') is False:
            print('false QQ')
            train_image_feats = get_tiny_images(train_image_paths)
            with open('test_image_feats.pkl', 'wb') as handle:
                    pickle.dump(train_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
            test_image_feats = get_tiny_images(test_image_paths)
            with open('test_image_feats.pkl', 'wb') as handle:
                    pickle.dump(test_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open('tiny_train.pkl', 'rb') as handle:
                # print('tiny_train exists')
                train_image_feats = pickle.load(handle)
                # print(type(train_image_feats),len(train_image_feats))
            with open('tiny_test.pkl', 'rb') as handle:
                # print('test_image_feats exists')
                test_image_feats = pickle.load(handle)
                # print(type(test_image_feats),len(test_image_feats))

    elif FEATURE == 'bag_of_sift':
        # YOU CODE build_vocabulary.py
        if os.path.isfile('vocab.pkl') is False:
            print('No existing visual word vocabulary found. Computing one from training images\n')
            vocab_size = 400   ### Vocab_size is up to you. Larger values will work better (to a point) but be slower to comput.
            vocab = build_vocabulary(train_image_paths, vocab_size)
            with open('vocab.pkl', 'wb') as handle:
                pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if os.path.isfile('train_image_feats.pkl') is False:
            # YOU CODE get_bags_of_sifts.py
            train_image_feats = get_bags_of_sifts(train_image_paths);
            with open('train_image_feats.pkl', 'wb') as handle:
                pickle.dump(train_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open('train_image_feats.pkl', 'rb') as handle:
                train_image_feats = pickle.load(handle)

        if os.path.isfile('test_image_feats.pkl') is False:
            test_image_feats  = get_bags_of_sifts(test_image_paths);
            with open('test_image_feats.pkl', 'wb') as handle:
                pickle.dump(test_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open('test_image_feats.pkl', 'rb') as handle:
                test_image_feats = pickle.load(handle)
    elif FEATURE == 'dumy_feature':
        train_image_feats = []
        test_image_feats = []
    else:
        raise NameError('Unknown feature type')
    #####################initialize the possibility
    possibility=[float(1/num_all)]*num_all
    sum_p=[]
    for i in range(1,num_all):
        sum_p.append(float(i/num_all))
    testing=[]
    for i in range(num_trial):
        training_id=[]
        #choose 100 examples 
        training_id=choose_training_id(sum_p)
        #training
        predicted_categories = svm_classify([train_image_feats[idx] for idx in training_id], [train_labels[idx] for idx in training_id],train_image_feats)
        temp=[]
        temp=svm_classify([train_image_feats[idx] for idx in training_id], [train_labels[idx] for idx in training_id],test_image_feats)
        testing.append(temp)
        #update possibility
        sum_p=[]
        possibility,sum_p=update_possibility(possibility,predicted_categories,train_labels) 
    accuracy=cal_result(testing,test_labels)
    # TODO Step 2: 
    # Classify each test image by training and using the appropriate classifier
    # Each function to classify test features will return an N x 1 array,
    # where N is the number of test cases and each entry is a string indicating
    # the predicted category for each test image. Each entry in
    # 'predicted_categories' must be one of the 15 strings in 'categories',
    # 'train_labels', and 'test_labels.

    # if CLASSIFIER == 'nearest_neighbor':
    #     # YOU CODE nearest_neighbor_classify.py
    #     predicted_categories = nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats)

    # elif CLASSIFIER == 'support_vector_machine':
    #     # YOU CODE svm_classify.py
    #     print('training')
    #     predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats)

    # elif CLASSIFIER == 'dumy_classifier':
    #     # The dummy classifier simply predicts a random category for
    #     # every test case
    #     predicted_categories = test_labels[:]
    #     shuffle(predicted_categories)
    # else:
    #     raise NameError('Unknown classifier type')

    # accuracy = float(len([x for x in zip(test_labels,predicted_categories) if x[0]== x[1]]))/float(len(test_labels))
    print("Accuracy = ", accuracy)
    '''
    test_labels_ids = [CATE2ID[x] for x in test_labels]
    predicted_categories_ids = [CATE2ID[x] for x in predicted_categories]
    train_labels_ids = [CATE2ID[x] for x in train_labels]
    
    # Step 3: Build a confusion matrix and score the recognition system
    # You do not need to code anything in this section. 
   
    build_confusion_mtx(test_labels_ids, predicted_categories_ids, ABBR_CATEGORIES)
    visualize(CATEGORIES, test_image_paths, test_labels_ids, predicted_categories_ids, train_image_paths, train_labels_ids)
    '''
def build_confusion_mtx(test_labels_ids, predicted_categories, abbr_categories):
    # Compute confusion matrix
    cm = confusion_matrix(test_labels_ids, predicted_categories)
    np.set_printoptions(precision=2)
    '''
    print('Confusion matrix, without normalization')
    print(cm)
    plt.figure()
    plot_confusion_matrix(cm, CATEGORIES)
    '''
    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #print('Normalized confusion matrix')
    #print(cm_normalized)
    plt.figure()
    plot_confusion_matrix(cm_normalized, abbr_categories, title='Normalized confusion matrix')

    plt.show()
     
def plot_confusion_matrix(cm, category, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(category))
    plt.xticks(tick_marks, category, rotation=45)
    plt.yticks(tick_marks, category)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
 

if __name__ == '__main__':
    main()

