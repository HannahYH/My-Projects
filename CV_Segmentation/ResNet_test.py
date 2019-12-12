# Writted by Hanxin Chen, Xueyan Lu and Han Yang 
# Test should be done with one folder
# the model and test data set need to be change every time manualy

import time
import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import scipy.ndimage
from skimage import measure
from skimage import measure
from keras.models import load_model
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_test_list(test_list):
  test = []
  test_dir = "/content/drive/My Drive/COMP9517/data/images/train-volume"
  for i in test_list:
    if i < 10:
      file_num = '0'+str(i)
    else:
      file_num = str(i)
    test.append(test_dir+file_num+'.jpg')
  return test


def create_dark_blob(img):
    erosion = cv2.erode(img,None,iterations = 1)
    dilation = cv2.dilate(erosion, None, iterations=1)
    labels = measure.label(dilation, neighbors=8, background=255)
    mask = np.zeros(dilation.shape, dtype="uint8")
    # loop over the unique components
    for label in np.unique(labels):
        # if this is the background label, ignore it
        if label == 0:
            continue
        # otherwise, construct the label mask and count the
        # number of pixels 
        labelMask = 255 * np.zeros(dilation.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = len(labels[labels == label])
        # if the number of pixels in the component is sufficiently
        # large, then add it to our mask of "large blobs"
        if numPixels > 5000:
            mask = cv2.add(mask, labelMask)
    #print('mask: ', mask)
    return 255-mask

class RESNET_TEST(object):
    def __init__(self):
        # choose the model be test 
        self.model_name='/content/drive/My Drive/residual_epoch30_valacc0.92_valloss0.19.hdf5'
        self.input_shape=(512,512)
        self.model=load_model(self.model_name)

    def pre_process_image(self,image_name):
        image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, self.input_shape)
        image = image / 255
        image = image.reshape(1,self.input_shape[0], self.input_shape[1],1)
        return image

    def infer(self,input_image):
        output_mask = self.model.predict(input_image, verbose=1)
        output_mask=output_mask.reshape(self.input_shape[0],self.input_shape[1],1)
        return output_mask


test_imgs = get_test_list()# this function need a parameter to get a list of intger that shown in train
resnet=RESNET_TEST()
test_datas = []
test_labels = []
for test_img in test_imgs:

    # load the test data
  img = resnet.pre_process_image(test_img)
  output_mask=resnet.infer(img)
  
  save_dir = "/content/drive/My Drive/COMP9517/Residual/model/batch_2/train-volume"
  label_dir = "/content/drive/My Drive/COMP9517/data/labels/train-labels"
  if test_img[-7] in ['1','2']:
    file_num = test_img[-7:]
  else:
    file_num = test_img[-6:]
  
  # save the data into a file
  output_mask.resize(512,512)
  plt.imsave(save_dir+file_num, output_mask, cmap='gray',format='jpg')
  predict_img = cv2.imread(save_dir+file_num,0)
  #cv2.imwrite(save_dir+file_num, predict_img)
  
  # do post-processing of the prediction result
  # and write the processed result into a file
  predict_img = cv2.imread(save_dir+file_num,0)
  img = scipy.ndimage.grey_opening(predict_img, structure=np.ones((3,3))).astype(np.int)
  img = np.array(img, np.uint8)
  img = cv2.bilateralFilter(img,9,75,75)
  ret,img = cv2.threshold(img,116,255,cv2.THRESH_BINARY)
  #img = create_dark_blob(img)
  plt.imshow(img, cmap='gray')
  plt.show()
  cv2.imwrite(save_dir+file_num, img)
  
  # load test data and normalise it
  test_data = cv2.imread(test_img, cv2.IMREAD_GRAYSCALE)
  test_data = test_data.astype('float32') / 255.0
  test_datas.append(test_data)
  
  # load test label and normalise it
  # also need to set the value binary
  test_label = cv2.imread(label_dir+file_num, cv2.IMREAD_GRAYSCALE)
  test_label = test_label.astype('float32') / 255.0
  test_label[test_label > 0.5] = 1
  test_label[test_label <= 0.5] = 0
  test_labels.append(test_label)

print(len(test_imgs))
test_datas = np.asarray(test_datas).reshape(len(test_imgs),512,512,1)
test_labels = np.asarray(test_labels).reshape(len(test_imgs),512,512,1)
# input the test data and test ground truth use Keras function 
# to test the loss and accuracy of the model
e = resnet.model.evaluate(test_datas, test_labels)
print(e)
