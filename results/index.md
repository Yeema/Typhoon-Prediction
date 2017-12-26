<center>
<h1>Project 3 results visualization</h1>
<img src="confusion_matrix.png">

## Overview
The project is related to 
> KNN classification and linear SVMs for multi-class classification


## Implementation
1. Build a tiny image feature
	* Cut the images to square, and resize to the smaller ones. 
	* Zero mean and unit length normalization to the features.
	
```python
    #Zero mean unit length normalization
    tiny_images_T = tiny_images

    for idx, feature in enumerate(tiny_images_T):
        feature_scaled = preprocessing.scale(feature,with_std = True)
        # feature_scaled = feature_scaled/np.max(abs(feature_scaled))
        tiny_images[idx,:] = feature_scaled
```
2. Bags-of-sifts feature
	* Build up the vocabulary for training and testing dataset.
	* Sample the descriptors and the cluster to the center with kmeans.
3. Get bags of sifts
	* Extract the descriptors for specific step number.
	* Calculate the distances between the vocabulary and the descriptors.
	* Choose the minimum distance for the each descriptor and build up the histogram.
4. KNN
	* Calculate the euclidean distance between the training image features and the testing image features
	* Count the first K small distances' label, the most one is the prediction.
5. Linear SVM
	Using LinearSVC function to predict the output labels. We can modify the C value to get the best result with different step when we build the vocab.


## Installation
* Other required packages.
	* Install Anaconda and get the packages inside.
	* conda install -c menpo cyvlfeat

* How to compile from source?
	* activate the environment which include the packages used in this project
	* Use command line type "python proj3.py --feature=? --classifier=?", and it can compile the source.
	* Feature can choose "tiny_image" or "bag_of_sift"
	* Classifier can choose "nearest_neighbor" or "support_vector_machine"
	
### Results
1. **tiny-images**  x  **KNN**(K=1) :  0.2307
2. **tiny-images**  x  **linear SVM**(C=0.0004) : 0.2174
3. **bags-of-sifts**  x  **KNN**
	* Step=[5,5], K=1: 0.5020
	* Step=[5,5], K=5: 0.5374
	* Step=[3,3], K=1: 0.5027
	* Step=[3,3], K=5: 0.5514
4. **bags-of-sifts**  x  **linear SVM**
	* Step=[5,5], C=0.00003 : 0.7047
	* Step=[3,3], C=0.000004 : 0.7194

### Confusion matrix
<img src="matrix.png">

<br>Use **bags-of-sifts**(Step=[3,3])  x  **linear SVM**(C=0.000004)</br>
<br>Accuracy (mean of diagonal of confusion matrix) is 0.7194</br>

<p>

## Visualization
| Category name | Sample training images | Sample true positives | False positives with true label | False negatives with wrong predicted label |
| :-----------: | :--------------------: | :-------------------: | :-----------------------------: | :----------------------------------------: |
| Kitchen | ![](thumbnails/Kitchen_train_image_0001.jpg) | ![](thumbnails/Kitchen_TP_image_0192.jpg) | ![](thumbnails/Kitchen_FP_image_0024.jpg) | ![](thumbnails/Kitchen_FN_image_0190.jpg) |
| Store | ![](thumbnails/Store_train_image_0001.jpg) | ![](thumbnails/Store_TP_image_0151.jpg) | ![](thumbnails/Store_FP_image_0026.jpg) | ![](thumbnails/Store_FN_image_0143.jpg) |
| Bedroom | ![](thumbnails/Bedroom_train_image_0001.jpg) | ![](thumbnails/Bedroom_TP_image_0175.jpg) | ![](thumbnails/Bedroom_FP_image_0008.jpg) | ![](thumbnails/Bedroom_FN_image_0180.jpg) |
| LivingRoom | ![](thumbnails/LivingRoom_train_image_0001.jpg) | ![](thumbnails/LivingRoom_TP_image_0147.jpg) | ![](thumbnails/LivingRoom_FP_image_0047.jpg) | ![](thumbnails/LivingRoom_FN_image_0145.jpg) |
| Office | ![](thumbnails/Office_train_image_0002.jpg) | ![](thumbnails/Office_TP_image_0185.jpg) | ![](thumbnails/Office_FP_image_0005.jpg) | ![](thumbnails/Office_FN_image_0144.jpg) |
| Industrial | ![](thumbnails/Industrial_train_image_0002.jpg) | ![](thumbnails/Industrial_TP_image_0152.jpg) | ![](thumbnails/Industrial_FP_image_0032.jpg) | ![](thumbnails/Industrial_FN_image_0148.jpg) |
| Suburb | ![](thumbnails/Suburb_train_image_0002.jpg) | ![](thumbnails/Suburb_TP_image_0176.jpg) | ![](thumbnails/Suburb_FP_image_0076.jpg) | ![](thumbnails/Suburb_FN_image_0013.jpg) |
| InsideCity | ![](thumbnails/InsideCity_train_image_0005.jpg) | ![](thumbnails/InsideCity_TP_image_0137.jpg) | ![](thumbnails/InsideCity_FP_image_0047.jpg) | ![](thumbnails/InsideCity_FN_image_0140.jpg) |
| TallBuilding | ![](thumbnails/TallBuilding_train_image_0010.jpg) | ![](thumbnails/TallBuilding_TP_image_0129.jpg) | ![](thumbnails/TallBuilding_FP_image_0059.jpg) | ![](thumbnails/TallBuilding_FN_image_0131.jpg) |
| Street | ![](thumbnails/Street_train_image_0001.jpg) | ![](thumbnails/Street_TP_image_0147.jpg) | ![](thumbnails/Street_FP_image_0128.jpg) | ![](thumbnails/Street_FN_image_0149.jpg) |
| Highway | ![](thumbnails/Highway_train_image_0009.jpg) | ![](thumbnails/Highway_TP_image_0162.jpg) | ![](thumbnails/Highway_FP_image_0079.jpg) | ![](thumbnails/Highway_FN_image_0144.jpg) |
| OpenCountry | ![](thumbnails/OpenCountry_train_image_0003.jpg) | ![](thumbnails/OpenCountry_TP_image_0125.jpg) | ![](thumbnails/OpenCountry_FP_image_0082.jpg) | ![](thumbnails/OpenCountry_FN_image_0123.jpg) |
| Coast | ![](thumbnails/Coast_train_image_0006.jpg) | ![](thumbnails/Coast_TP_image_0130.jpg) | ![](thumbnails/Coast_FP_image_0030.jpg) | ![](thumbnails/Coast_FN_image_0122.jpg) |
| Mountain | ![](thumbnails/Mountain_train_image_0002.jpg) | ![](thumbnails/Mountain_TP_image_0123.jpg) | ![](thumbnails/Mountain_FP_image_0124.jpg) | ![](thumbnails/Mountain_FN_image_0101.jpg) |
| Forest | ![](thumbnails/Forest_train_image_0003.jpg) | ![](thumbnails/Forest_TP_image_0142.jpg) | ![](thumbnails/Forest_FP_image_0101.jpg) | ![](thumbnails/Forest_FN_image_0128.jpg) |


