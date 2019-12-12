# Writted by Hanxin Chen, Xueyan Lu and Han Yang 
# This the code for Deep contextual residual network
# We use the RESNET calss, in the class it has two function
# Network() for set the layer to build the network for training
# train_network() to train data utilize the network
# generate_data() used to read data from file, reshape it into an array for train_network

# In main function, we split the raw data into k folder to do the cross validation
# befor train data for each folder, the current folder will be print
# test should be done for only one folder

import keras
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from keras.models import Model
from keras.models import load_model
from keras.layers import Input, Concatenate, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D,Reshape, BatchNormalization, add, Conv2DTranspose, Dense

from keras.optimizers import Adam


class RESNET(object):
    def __init__(self, train_list):
        self.img_size = (512,512)
        self.train_image_path = './COMP9517/data/images/'
        self.train_label_path = './COMP9517/data/labels/'
        self.train_batch_size = 4
        self.val_batch_size = 1
        self.val_size = 0.2
        self.learningrate = 0.0001
        self.epoch = 10
        self.train_list = train_list

        self.model = self.Network()
        
        
    def Network(self):
        input_shape = Input(shape=(512, 512, 1))
        #conv1
        conv1_1 = Conv2D(16, 3, activation='elu',padding = 'same', name='conv1_1')(input_shape)
        #pool1
        poo1_1 = MaxPooling2D(pool_size=(2, 2),name='poo1_1')(conv1_1)#--->
        #dconv1
        dconv1 = Conv2D(32, 3, activation='elu',padding = 'same', name='dconv1')(UpSampling2D(size=(2, 2))(poo1_1))
        #conv2
        conv2_1 = Conv2D(32, 3, activation='elu',padding = 'same',subsample =(2,2), name='conv2_1')(poo1_1)
        conv2_2 = Conv2D(32, 3, activation='elu',padding = 'same', name='conv2_2')(conv2_1)
        #conv3
        conv3_1 = Conv2D(32, 3, activation='elu',padding = 'same', name='conv3_1')(conv2_2)
        conv3_2 = Conv2D(32, 3, activation='elu',padding = 'same', name='conv3_2')(conv3_1)
        conv3_3 = Conv2D(32, 3, activation='elu',padding = 'same', name='conv3_3')(conv3_2)
        conv3_4 = Conv2D(32, 3, activation='elu',padding = 'same', name='conv3_4')(conv3_3)#--->
        #dconv2
        dconv2 = Conv2D(32, 3, activation='elu',padding = 'same', name='dconv2')(UpSampling2D(size=(4, 4))(conv3_4))
        #conv4
        conv4_1 = Conv2D(64, 3, activation='elu',padding = 'same',subsample =(2,2), name='conv4_1')(conv3_4)
        conv4_2 = Conv2D(64, 3, activation='elu',padding = 'same', name='conv4_2')(conv4_1)
        #conv5
        conv5_1 = Conv2D(64, 3, activation='elu',padding = 'same', name='conv5_1')(conv4_2)
        conv5_2 = Conv2D(64, 3, activation='elu',padding = 'same', name='conv5_2')(conv5_1)
        conv5_3 = Conv2D(64, 3, activation='elu',padding = 'same', name='conv5_3')(conv5_2)
        conv5_4 = Conv2D(64, 3, activation='elu',padding = 'same', name='conv5_4')(conv5_3)
        #conv6
        conv6_1 = Conv2D(128, 3, activation='elu',padding = 'same', name='conv6_1')(conv5_4)
        conv6_2 = Conv2D(128, 3, activation='elu',padding = 'same', name='conv6_2')(conv6_1)
        conv6_3 = Conv2D(128, 3, activation='elu',padding = 'same', name='conv6_3')(conv6_2)
        conv6_4 = Conv2D(128, 3, activation='elu',padding = 'same', name='conv6_4')(conv6_3)
        conv6_5 = Conv2D(128, 3, activation='elu',padding = 'same', name='conv6_5')(conv6_4)
        conv6_6 = Conv2D(128, 3, activation='elu',padding = 'same', name='conv6_6')(conv6_5)
        conv6_7 = Conv2D(128, 3, activation='elu',padding = 'same', name='conv6_7')(conv6_6)
        conv6_8 = Conv2D(128, 3, activation='elu',padding = 'same', name='conv6_8')(conv6_7)
        conv6_9 = Conv2D(128, 3, activation='elu',padding = 'same', name='conv6_9')(conv6_8)
        conv6_10 = Conv2D(128, 3, activation='elu',padding = 'same', name='conv6_10')(conv6_9)
        conv6_11 = Conv2D(128, 3, activation='elu',padding = 'same', name='conv6_11')(conv6_10)
        conv6_12 = Conv2D(128, 3, activation='elu',padding = 'same', name='conv6_12')(conv6_11)
        #conv7
        conv7_1 = Conv2D(128, 3, activation='elu',padding = 'same', name='conv7_1')(conv6_12)
        conv7_2 = Conv2D(128, 3, activation='elu',padding = 'same', name='conv7_2')(conv7_1)
        conv7_3 = Conv2D(128, 3, activation='elu',padding = 'same', name='conv7_3')(conv7_2)
        conv7_4 = Conv2D(128, 3, activation='elu',padding = 'same', name='conv7_4')(conv7_3)
        conv7_5 = Conv2D(128, 3, activation='elu',padding = 'same', name='conv7_5')(conv7_4)
        conv7_6 = Conv2D(128, 3, activation='elu',padding = 'same', name='conv7_6')(conv7_5)
        #conv8
        conv8_1 = Conv2D(128, 3, activation='elu',padding = 'same', name='conv8_1')(conv7_6)
        conv8_2 = Conv2D(256, 3, activation='elu',padding = 'same', name='conv8_2')(conv8_1)
        conv8_3 = Conv2D(256, 3, activation='elu',padding = 'same', name='conv8_3')(conv8_2)
        #conv9
        conv9_1 = Conv2D(256, 3, activation='elu',padding = 'same', name='conv9_1')(conv8_3)
        conv9_2 = Conv2D(512, 3, activation='elu',padding = 'same', name='conv9_2')(conv9_1)
        conv9_3 = Conv2D(512, 3, activation='elu',padding = 'same', name='conv9_3')(conv9_2)
        #dconv3
        dconv3 = Conv2D(32, 3, activation='elu',padding = 'same', name='dconv3')(UpSampling2D(size=(8, 8))(conv9_3))
        #merge
        merge1 = add([dconv1,dconv2])
        merge2 = add([merge1,dconv3])

        #merge2 = concatenate([dconv1,dconv2,dconv3],axis = 3)
        #dropout
        drop1 = Dropout(0.5)(merge2)
        #conv10
        conv10_1 = Conv2D(16, 3, activation='relu',padding = 'same', name='conv10_1')(drop1)
        #dropout
        drop2 = Dropout(0.5)(conv10_1)
        #conv11
        conv11_1 = Conv2D(16, 3, activation='relu',padding = 'same', name='conv11_1')(drop2)
        #outpt
        outpt = Conv2D(1, 1, activation='sigmoid',padding = 'same', name='outpt')(conv11_1)

        model = Model(input = input_shape, output = outpt)
        adam = Adam(1e-4)
        model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

        return model


    
    def train_network(self):
        images, labels = self.generate_data()
        print("begin")
        checkpointer = keras.callbacks.ModelCheckpoint(
            filepath='residual_epoch{epoch:02d}_valacc{val_acc:.2f}_valloss{val_loss:.2f}.hdf5', monitor='val_acc',
            verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        self.model.fit(images, labels, batch_size = 4, validation_split = 0.1, epochs = 80, callbacks = [checkpointer])
    

    def generate_data(self):
      X_image = []
      Y_label = []
      
      imgs = sorted(os.listdir(self.train_image_path))
      labs = sorted(os.listdir(self.train_label_path))
      print("image: ", imgs)
      print("label: ", labs)
      
      for i in range(len(imgs)):
        # For corss validation, to skip read the files that will be test later
        if imgs[i][-7] not in ['2','1'] and int(imgs[i][-6:-4]) in self.train_list:
          continue
        img = cv2.imread(self.train_image_path+imgs[i], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, self.img_size)
        img = img.astype('float32') / 255.0
        X_image.append(img)
        
        label = cv2.imread(self.train_label_path+labs[i], cv2.IMREAD_GRAYSCALE)
        label = cv2.resize(label, self.img_size)
        label = label.astype('float32') / 255.0
        label[label > 0.5] = 1
        label[label <= 0.5] = 0
        Y_label.append(label)
      
      X_image = np.asarray(X_image).reshape(len(imgs)-len(self.train_list),512,512,1)
      Y_label = np.asarray(Y_label).reshape(len(imgs)-len(self.train_list),512,512,1)
      
      return X_image, Y_label


# Split the data into K folders    
def cross_validation_k_fold(k):
  arr = np.arange(30)
  np.random.shuffle(arr)
  arr = arr.reshape(k, int(30/k))
  return arr


if __name__ == '__main__':
  cross_val_folders = cross_validation_k_fold(10)
  for fold in cross_val_folders:
    print(fold)
    resnet = RESNET(fold)
    resnet.train_network()
