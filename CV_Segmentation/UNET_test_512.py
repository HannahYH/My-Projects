# Writted by Hanxin Chen, Xueyan Lu and Han Yang 
# this is testing part of code for one fold
# since the hdf5 filename might sometimes vary for some reasons
# we decide to do the test for each fold one by one
# when we change to evaluate on another fold,
# the code might vary a bit

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

# retrieve file path from test list
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
    for label in np.unique(labels):
        if label == 0:
            continue
        labelMask = 255 * np.zeros(dilation.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = len(labels[labels == label])
        if numPixels > 500:
            mask = cv2.add(mask, labelMask)
    return 255-mask
  

class UNET_TEST(object):
    def __init__(self):
        self.model_name='/content/drive/My Drive/COMP9517/model/21_3_2_unet_epoch221_valacc0.94_valloss0.15.hdf5'
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

# use pre-separated test list
# get filepath from test list
test_imgs = get_test_list([21, 3, 2])

unet=UNET_TEST()

test_datas = []
test_labels = []
for test_img in test_imgs:
    # predict and save images in order to do Vrand Vinfo offline
    img = unet.pre_process_image(test_img)
    output_mask=unet.infer(img)
    save_dir = "/content/drive/My Drive/COMP9517/model/21_3_2/train-volume"
    label_dir = "/content/drive/My Drive/COMP9517/data/labels/train-labels"
    if test_img[-7] in ['1','2']:
        file_num = test_img[-7:]
    else:
        file_num = test_img[-6:]
    output_mask.resize(512,512)
    plt.imsave(save_dir+file_num, output_mask, cmap='gray',format='jpg')
    predict_img = cv2.imread(save_dir+file_num,0)
    cv2.imwrite(save_dir+file_num, predict_img)
    predict_img = cv2.imread(save_dir+file_num,0)

    # further post-processing
    img = scipy.ndimage.grey_opening(predict_img, structure=np.ones((3,3))).astype(np.int)
    img = np.array(img, np.uint8)
    img = cv2.bilateralFilter(img,9,75,75)
    ret,img = cv2.threshold(img,116,255,cv2.THRESH_BINARY)
    #img = create_dark_blob(img)
    plt.imshow(img, cmap='gray')
    plt.show()

    # get the intial raw image and label
    # pre-processing
    test_data = cv2.imread(test_img, cv2.IMREAD_GRAYSCALE)
    test_data = test_data.astype('float32') / 255.0
    test_datas.append(test_data)
  
    test_label = cv2.imread(label_dir+file_num, cv2.IMREAD_GRAYSCALE)
    test_label = test_label.astype('float32') / 255.0
    test_label[test_label > 0.5] = 1
    test_label[test_label <= 0.5] = 0
    test_labels.append(test_label)
  
    cv2.imwrite(save_dir+file_num, img)

print(len(test_imgs))
test_datas = np.asarray(test_datas).reshape(len(test_imgs),512,512,1)
test_labels = np.asarray(test_labels).reshape(len(test_imgs),512,512,1)
# use in-built function do evaluation
# this step will give accuracy and loss of test data
e = unet.model.evaluate(test_datas, test_labels)
print(e)
