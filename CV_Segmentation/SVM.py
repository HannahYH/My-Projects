# Writted by Hanxin Chen, Xueyan Lu and Han Yang 
# This is the code for TwoStep (SVM + Post-processing) Method
# https://github.com/dgriffiths3/ml_segmentation
# above link is the reference of our code

import os
import cv2
import time
import math
import random
import numpy as np
import pickle as pkl
import mahotas as mt
from glob import glob
from matplotlib import pyplot as plt
from numpy.lib import stride_tricks
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import train_test_split
from skimage import measure
from skimage.morphology import reconstruction, dilation
from scipy import ndimage as ndimage

# ============ Utils ==========================================
def read_data(image_dir, label_dir):
    print('[INFO] Loading image data.')
    # extract all pathes of images and labels, contains augmented images and labels
    trainlist = glob(os.path.join(image_dir, '*.jpg'))
    labellist = glob(os.path.join(label_dir, '*.jpg'))
    image_list = []
    label_list = []
    image_list_test = []
    label_list_test = []
    # read all first 30 raw images and labels, use it as test dataset for cross validation
    for file in trainlist[:30]:
        image_list_test.append(cv2.imread(file, 0))
    for file in labellist[:30]:
        label_list_test.append(cv2.imread(file, 0))
    # shuffle all images, use it as whole dataset for cross validation
    random.shuffle(trainlist)
    for i in range(len(trainlist[:30])):
        num = trainlist[i].split('volume')[1].split('.')[0]
        # extract the corresponding label file name
        name = labellist[0].split('label')[0] + 'labels/train-labels' + num + '.jpg'
        image_list.append(cv2.imread(trainlist[i], 0))
        label_list.append(cv2.imread(name, 0))
    return image_list, label_list, image_list_test, label_list_test
  
def subsample_idx(low, high, sample_size): 
    # create random index
    return np.random.randint(low,high,sample_size) 
# ============ Utils ==========================================

# ========== Extract Features Part =============================
def create_dark_blob(img):
    print('[INFO] Detecting dark blobs.')
    erosion = cv2.erode(img,None,iterations = 1)
    dilation = cv2.dilate(erosion, None, iterations=1)
    labels = measure.label(dilation, neighbors=8, background=255)
    mask = np.zeros(dilation.shape, dtype="uint8")
    # loop over the unique components
    for label in np.unique(labels):
        # if this is the background label, ignore it
        if label == 0:
            continue
        # otherwise, construct the label mask and count the number of pixels 
        labelMask = 255 * np.zeros(dilation.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = len(labels[labels == label])
        # if the number of pixels in the component is sufficiently
        # large, then add it to our mask of "large blobs"
        if numPixels > 5000:
            mask = cv2.add(mask, labelMask)
    return 255-mask
  
def create_LCHF_features(img):
    print('[INFO] Computing LCHF features.')
    median = cv2.medianBlur(img,3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(15,15))
    clahe = clahe.apply(median)
    ret, im_th = cv2.threshold(clahe,100,255,cv2.THRESH_BINARY)
    seed = np.copy(im_th)
    seed[1:-1, 1:-1] = im_th.max()
    mask = im_th
    filled = reconstruction(seed, mask, method='erosion').astype(int)
    filled[filled>255] = 255
    filled = filled.astype(np.uint8)
    out = cv2.medianBlur(filled, 5)
    return out
  
def create_features(img, label, train=True):
    # number of examples per image to use for training model
    num_examples = 1000 
    # reduce the dimention of image by 1 dim
    img_reshape = img.reshape(img.shape[0]*img.shape[1], 1)
    features = np.zeros((img.shape[0]*img.shape[1], 1))
    # the first dimention is the pixel of image
    features[:,:1] = img_reshape
    
    ss_idx = []
    # get random features' index
    if train == True:
        non = []
        mem = []
        labels = label.reshape(label.shape[0]*label.shape[1], 1)
        for i in range(len(labels)):
            if labels[i] > 128:
                non.append(i)
            else:
                mem.append(i)
        # randomly select 2000 pixels, half membrane and half non-membrane
        idx_non = subsample_idx(0, len(non), num_examples)
        idx_mem = subsample_idx(0, len(mem), num_examples)
        ss_idx = [*idx_non, *idx_mem]
        features = features[ss_idx]
        # create labels
        labels = labels[ss_idx]
    else:
        labels = None
    
    # ---------- start creating features ----------
    # Laplace
    Laplace_feature = cv2.Laplacian(img, cv2.CV_64F)
    # LCHF
    LCHF_feature = create_LCHF_features(img)
    # Sobel
    sobel_feature = cv2.Sobel(img, cv2.CV_8U, 1, 1, ksize=7)
    # Bilteral & global thresholding
    bil = cv2.bilateralFilter(img,9,75,75)
    ret, bg_feature = cv2.threshold(bil,116,255,cv2.THRESH_BINARY)
    # Attribute Opening
    ao = ndimage.grey_opening(img, structure=np.ones((3,3))).astype(np.int)
    ao = np.array(ao, np.uint8)
    ao = cv2.bilateralFilter(ao,9,75,75)
    ret, ao_feature = cv2.threshold(ao,116,255,cv2.THRESH_BINARY)
    # ---------- integrating features ----------
    # Laplace
    Laplace_features = Laplace_feature.reshape(Laplace_feature.shape[0]*Laplace_feature.shape[1], 1)
    if train == True:
        Laplace_features = Laplace_features[ss_idx]
    features = np.hstack((features, Laplace_features))
    # LCHF
    LCHF_features = LCHF_feature.reshape(LCHF_feature.shape[0]*LCHF_feature.shape[1], 1)
    if train == True:
        LCHF_features = LCHF_features[ss_idx]
    features = np.hstack((features, LCHF_features))
    # Sobel
    sobel_features = sobel_feature.reshape(sobel_feature.shape[0]*sobel_feature.shape[1], 1)
    if train == True:
        sobel_features = sobel_features[ss_idx]
    features = np.hstack((features, sobel_features))
    # Bilteral & global thresholding
    bg_features = bg_feature.reshape(bg_feature.shape[0]*bg_feature.shape[1], 1)
    if train == True:
        bg_features = bg_features[ss_idx]
    features = np.hstack((features, bg_features))
    # Attribute Opening
    ao_features = ao_feature.reshape(ao_feature.shape[0]*ao_feature.shape[1], 1)
    if train == True:
        ao_features = ao_features[ss_idx]
    features = np.hstack((features, ao_features))
    # ---------- end creating features ----------
    return features, labels

def create_dataset(image_list, label_list):
    print('[INFO] Creating dataset on %d image(s).' %len(image_list))
    X = []
    y = []
    for i, img in enumerate(image_list):
        features, labels = create_features(img, label_list[i])
        X.append(features)
        y.append(labels)
    X = np.array(X)
    print('[INFO] Features shape in ', X.shape)
    X = X.reshape(X.shape[0]*X.shape[1], X.shape[2])
    y = np.array(y)
    print('[INFO] Labels shape in ', y.shape)
    y = y.reshape(y.shape[0]*y.shape[1], y.shape[2]).ravel()
    return X, y
# ========== Extract Features Part ======================
  
# ========== Training Part ==============================
def train_model(X, y):
    print('[INFO] Training Support Vector Machine model.')
    model = SVC(gamma='scale', verbose=1)
    model.fit(X, y)
    print('[INFO] Model training complete.')
    print('[INFO] Training Accuracy: %.2f' %model.score(X, y))
    return model
# ========== Training Part =============================

# ========== Evaluation Part ===========================
def test_model(X, y, model):
    print('[INFO] Evaluating Support Vector Machine model.')
    pred = model.predict(X)
    accuracy = metrics.accuracy_score(y, pred)
    precision = metrics.precision_score(y, pred, average='weighted', labels=np.unique(pred))
    recall = metrics.recall_score(y, pred, average='weighted', labels=np.unique(pred))
    f1 = metrics.f1_score(y, pred, average='weighted', labels=np.unique(pred))
    
    print('--------------------------------')
    print('[RESULTS] Accuracy: %.2f' %accuracy)
    print('[RESULTS] Precision: %.2f' %precision)
    print('[RESULTS] Recall: %.2f' %recall)
    print('[RESULTS] F1: %.2f' %f1)
    print('--------------------------------')
    return accuracy
# ========== Evaluation Part =============================
    
# ========== Prediction Part =============================
def create_features_for_test(img):
    features, _ = create_features(img, label=None, train=False)
    return features

def compute_prediction(img, model):
    features = create_features_for_test(img)
    print('[INFO] Predicting:')
    predictions = model.predict(features.reshape(-1, features.shape[1]))
    print('[INFO] Finish Prediction:')
    pred_size = int(math.sqrt(features.shape[0]))
    inference_img = predictions.reshape(pred_size, pred_size)
    return inference_img
# ========== Prediction Part =============================
    
def main(image_dir, label_dir, output_model, test_dir, output_dir):
    start = time.time()
    print('[INFO] Start:')
    # training 
    image_list, label_list, image_list_test, label_list_test = read_data(image_dir, label_dir)
    print('[INFO] Creating the whole dataset:')
    X, y = create_dataset(image_list, label_list)
    print('[INFO] Creating the Testing dataset:')
    X_test, y_test = create_dataset(image_list_test, label_list_test)
    
    # cross-validation 
    fold = 5
    fold_len = int(len(X)/fold)
    idx = [i for i in range(len(X))]
    for i in range(fold):
        # for each fold, the size of train dataset is 120
        X_train = X[[*idx[:i*fold_len], *idx[(i+1)*fold_len:fold*fold_len]]]
        y_train = y[[*idx[:i*fold_len], *idx[(i+1)*fold_len:fold*fold_len]]]
        print('[INFO] Fold ' + str(i+1) + ', Training dataset shape in ', X_train.shape, y_train.shape)
        print('[INFO] Fold ' + str(i+1) + ', Testing dataset shape in ', X_test.shape, y_test.shape)
        model = train_model(X_train, y_train)
        pkl.dump(model, open(output_model+'/aug_data_0810_'+str(i)+'.pkl', "wb"))
        # evaluation
        test_model(X_test, y_test, model) 
      
    # predicting 
    print('[INFO] Using Support Vector Machine model to do prediction.')
    loaded_model = pkl.load(open(output_model+'/aug_data_0810_'+str(0)+'.pkl', "rb"))
    filelist = glob(os.path.join(test_dir,'*.jpg'))
    print('[INFO] Running prediction on %s test images' %len(filelist))
    for file in filelist:
        print('[INFO] Processing images:', os.path.basename(file))
        inference_img = compute_prediction(cv2.imread(file, 0), loaded_model)
        # post-processing: remove dark blobs
        print('[INFO] Post-Processing images:')
        mask = create_dark_blob(inference_img)
        cv2.imwrite(os.path.join(output_dir, 'predict_' + os.path.basename(file)), mask)
    
    print('[INFO] Processing time:',time.time()-start)

if __name__ == "__main__":
    image_dir = "/content/drive/My Drive/COMP9517/data/images"
    label_dir = "/content/drive/My Drive/COMP9517/data/labels"
    output_model = "/content/drive/My Drive/COMP9517/data"
    predict_dir = "/content/drive/My Drive/COMP9517/data/tests"
    output_dir = "/content/drive/My Drive/COMP9517/data/results"
    main(image_dir, label_dir, output_model, predict_dir, output_dir)
