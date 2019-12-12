# Writted by Hanxin Chen, Xueyan Lu and Han Yang 
# Network 1
# 572 * 572 -> 388 * 388
# After we try on Network 1,
# we found output of this network does not match with labels(even after perform a center crop)
# it looks like the predicted image zoom in a little bit than the 388*388 center cropped label
# so later we focus on Network 2 which also implements cross validation

import keras
import random
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Model
from keras.models import load_model
from keras.layers import Input, Concatenate, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D,Reshape
from keras.optimizers import Adam

class UNET(object):
    def __init__(self):
        self.img_size = (572,572)
        self.train_image_path = '/content/drive/My Drive/COMP9517/data/images/'
        self.train_label_path = '/content/drive/My Drive/COMP9517/data/labels/'
        self.train_batch_size = 10
        self.val_batch_size = 1
        self.val_size =0.2
        self.learningrate = 1e-4
        self.epoch = 100

        self.model = self.Network()
    
    def Network(self):
        # 572,572
        inputs = Input(shape=(self.img_size[0], self.img_size[1], 1), batch_shape=None)
        
        conv1 = Conv2D(64, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        
        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(drop5))
        # Center Crop and Skip Connection
        merge6 = Concatenate(axis=3)([Cropping2D(cropping=((4,4),(4,4)))(drop4), up6])
        conv6 = Conv2D(512, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(conv6)
        
        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))
        # Center Crop and Skip Connection
        merge7 = Concatenate(axis=3)([Cropping2D(cropping=((16,16),(16,16)))(conv3), up7])
        conv7 = Conv2D(256, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv7))
        # Center Crop and Skip Connection
        merge8 = Concatenate(axis=3)([Cropping2D(cropping=((40,40),(40,40)))(conv2), up8])
        conv8 = Conv2D(128, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv8))
        # Center Crop and Skip Connection
        merge9 = Concatenate(axis=3)([Cropping2D(cropping=((88,88),(88,88)))(conv1), up9])
        conv9 = Conv2D(64, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
        print('9-10:', conv9.shape, merge9.shape, conv10.shape)
        print(conv10)

        model = Model(input=inputs, output=conv10)

        return model
      
    def train_network(self):
        images_label_list_train, images_label_list_val = self.load_images()
        self.model.compile(optimizer=Adam(lr=self.learningrate), loss='binary_crossentropy', metrics=['accuracy'])

        print("begin")
        checkpointer = keras.callbacks.ModelCheckpoint(
            filepath='unet_epoch{epoch:02d}_valacc{val_acc:.2f}_valloss{val_loss:.2f}.hdf5', monitor='val_acc',
            verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        reducelr=keras.callbacks.ReduceLROnPlateau(monitor='train_loss', factor=0.1, patience=4, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
        earlystopping=keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=30, verbose=1, mode='auto')
        self.model.fit_generator(self.generate_data(images_label_list_train), \
                                 samples_per_epoch=self.train_batch_size, \
                                 nb_epoch=self.epoch, \
                                 initial_epoch=0, \
                                 validation_data=self.generate_data(images_label_list_val), \
                                 nb_val_samples=self.val_batch_size, \
                                 verbose=1, \
                                 nb_worker=1, \
                                 callbacks=[checkpointer,reducelr,earlystopping])

    
    # return a filepath list according to train_image_path and train_label_path
    def load_images(self):
        images_label_list_train = []
        images_label_list_val = []
        picnames = sorted(os.listdir(self.train_image_path))
        print(picnames)
        labelnames = sorted(os.listdir(self.train_label_path))
        print(labelnames)
        for i in range(0, len(picnames)):
                if i % int(len(picnames) / (len(picnames) * self.val_size)) != 0:
                    images_label_list_train.append([self.train_image_path + picnames[i], self.train_label_path + labelnames[i]])
                else:
                    images_label_list_val.append([self.train_image_path + picnames[i], self.train_label_path + labelnames[i]])
        return (images_label_list_train, images_label_list_val)

    # retrieve image and pre-processing
    def generate_data(self, images_label_list_train):
        while True:
            random.shuffle(images_label_list_train)
            X_image = []
            Y_label = []
            count = 0
            for image_label in images_label_list_train:
                raw = cv2.imread(image_label[0], cv2.IMREAD_GRAYSCALE)
                img = cv2.copyMakeBorder(raw,30,30,30,30,cv2.BORDER_REFLECT)
                img = img.astype('float32') / 255.0
                X_image.append(img)

                label_raw = cv2.imread(image_label[1], cv2.IMREAD_GRAYSCALE)
                label = label_raw[62:-62,62:-62]
                label = label.astype('float32') / 255.0
                label[label > 0.5] = 1
                label[label <= 0.5] = 0
                Y_label.append(label)
                count += 1
                if count == self.train_batch_size:
                    count = 0
                    out_image=np.asarray(X_image).reshape(self.train_batch_size,img.shape[0],img.shape[1],1)
                    out_label=np.asarray(Y_label).reshape(self.train_batch_size,label.shape[0],label.shape[1],1)
                    yield (out_image,out_label)
                    X_image = []
                    Y_label = []


if __name__ == '__main__':
    unet = UNET()
    unet.train_network()
