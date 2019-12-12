# Writted by Hanxin Chen, Xueyan Lu and Han Yang 
# Data Augmentation is useful in learning based methods
# we use elastic transformation to enlarge our dataset to 300 images
# by applying different parameters to the function

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
def elastic_transform(image, mask, alpha, sigma, alpha_affine=None, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    res_x = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    res_y = map_coordinates(mask, indices, order=1, mode='reflect').reshape(shape)
    return res_x, res_y

if __name__ == '__main__':
    img_list = []
    label_list = []
    aug_img_list = []
    aug_label_list = []
    alpha_list = [180,200,180,150,220,180,190,240,160]
    sigma_list = [8,  9,  10,  10, 10, 9,  9,  9,  9]
    i = 30
    for t in range(9):
        for file in sorted(os.listdir('/content/drive/My Drive/Colab Notebooks/9517 colab/data/images')):
            image = cv2.imread('/content/drive/My Drive/Colab Notebooks/9517 colab/data/images/'+file,0)
            label = cv2.imread('/content/drive/My Drive/Colab Notebooks/9517 colab/data/labels/train-labels'+file[-6:],0)
            image = np.array(image)
            label = np.array(label)
            img_list.append(image)
            label_list.append(label)
            
            aug_img, aug_label = elastic_transform(image, label, alpha=alpha_list[t], sigma=sigma_list[t], \
                                                   alpha_affine=None, random_state=None)

            aug_img_list.append(aug_img)
            aug_label_list.append(aug_label)
            cv2.imwrite("/content/drive/My Drive/COMP9517/data/images/train-volume"+str(i)+".jpg", aug_img)
            cv2.imwrite("/content/drive/My Drive/COMP9517/data/labels/train-labels"+str(i)+".jpg", aug_label)
            i += 1
            plt.subplots(4, sharex=True, figsize=(15,15))
            plt.subplot(1,4,1);plt.imshow(image,cmap='gray');
            plt.subplot(1,4,2);plt.imshow(label,cmap='gray');
            plt.subplot(1,4,3);plt.imshow(aug_img,cmap='gray')
            plt.subplot(1,4,4);plt.imshow(aug_label,cmap='gray')
