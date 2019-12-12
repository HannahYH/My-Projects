
"""
Written by 
Student Name: Chengbin Feng 
Student Id: z5109259 
Student Email: chengbin.feng@unsw.edu.au 
"""

# coding: utf-8

# In[478]:


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from PIL import Image
from scipy.misc import toimage

size = 100, 100
img_dir = "data/images"
label_dir = "data/labels"
#load images 
imagelist = os.listdir(img_dir)
labellist = os.listdir(label_dir)


# In[91]:


img_names = [img_dir + "/" + img for img in imagelist]
label_names = [label_dir + "/" + label for label in labellist]


# In[244]:


img = cv2.imread(img_names[0], 0)
plt.imshow(gray, cmap="gray")


# ### Original 

# In[200]:


#
ret, th1 = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
ret, th2 = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY)
ret, th3 = cv2.threshold(img, 90, 255, cv2.THRESH_BINARY)
ret, th4 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
ret, th5 = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
ret, th6 = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)
ret, th7 = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
ret, th8= cv2.threshold(img, 220, 255, cv2.THRESH_BINARY)


# In[201]:


O_IMAGS = [th1, th2, th3, th4, th5, th6, th7, th8]
O_TITS = ['30', '60', '90', '127', '150', '180', '200', '220']
for i in range(8):
    plt.subplot(2, 4,i+1), plt.imshow(O_IMAGS[i])
    plt.title(O_TITS[i])
    plt.xticks([]), plt.yticks([])
plt.gray()


# In[222]:


th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 11, 2)
th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 2)
ret, th4 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)


# In[212]:


# AdaptiveThreshold Mean 
plt.imshow(th2)


# In[224]:


# AdaptiveThreshold GAUSSIAN
plt.imshow(th3)


# In[223]:


#OTSU
plt.imshow(th4)


# ---

# ### Median Blur is implemented in this part 

# In[249]:


median = cv2.medianBlur(img, 35)
plt.imshow(median)


# In[246]:


# Apply media 

ret, th1media = cv2.threshold(median, 90, 255, cv2.THRESH_BINARY)
plt.imshow(th1media)


# In[266]:


th2median = cv2.adaptiveThreshold(median, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 11, 2)
plt.imshow(th2median)


# In[268]:


th3median = cv2.adaptiveThreshold(median, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 3)
plt.imshow(th3median)


# In[280]:


ret, th4median = cv2.threshold(median, 60, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
plt.imshow(th4median)


# #### inisght: not so good 

# ---

# ### Gaussian Blur is implemended here

# In[281]:


gussian = cv2.GaussianBlur(img, (35,35), 0)
plt.imshow(gussian)


# In[298]:


ret, th1gussian = cv2.threshold(gussian, 120, 255, cv2.THRESH_BINARY)
plt.imshow(th1gussian)


# In[308]:


th2gussian = cv2.adaptiveThreshold(gussian, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 11, 3)
plt.imshow(th2gussian)


# In[314]:


th3gussian = cv2.adaptiveThreshold(gussian, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 1)
plt.imshow(th3gussian)


# In[321]:


ret, th4gussian = cv2.threshold(gussian, 80, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

plt.imshow(th4gussian)


# #### insight: 

# ----

# ### Watershed

# In[325]:


waterimg = cv2.imread(img_names[0], 0)
plt.imshow(waterimg)


# We applt the Otsu's binarization in finiding an approximate estimate of the VNC

# In[461]:


ret, threshwater = cv2.threshold(waterimg, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

plt.imshow(threshwater)


# In[380]:


def do_watershed(img, threshold = None):
    img_grey = img.convert('L')
    #convert to binary image 
    if threshold is not None: 
        img_grey = img_grey.point(lambda x: 0 if x < threshold else 255, '1')
    waterimg = np.array(img_grey)
    distance = ndi.distance_transform_edt(waterimg)
    local_maxi = peak_local_max(distance, indices=False,
                                    footprint=np.ones((3, 3)),
                                    labels=waterimg)
    markers = ndi.label(local_maxi)[0]
    ws_labels = watershed(-distance, markers, mask=waterimg)
    return ws_labels, distance 


# In[456]:


ws_labels_list = []
for img_path in img_names:
    img = Image.open(img_path)
    img.thumbnail(size)
    img_mat = np.array(img)
    ws_labels, distance = do_watershed(img, threshold=130)
    ws_labels_list.append(ws_labels)
print(len(ws_labels_list))
plt.imshow(ws_labels_list[20])


# In[462]:


ret, threshwaterorigin = cv2.threshold(waterimg, 90, 255, cv2.THRESH_BINARY)
plt.imshow(threshwaterorigin)


# In[483]:


# convert the nparray back to an image 
threshwaterorigin_water, distance_origin_water = do_watershed(toimage(threshwaterorigin, 'RGB'), 70)
plt.imshow(threshwaterorigin_water)


# In[490]:


threshwater_ad_median = cv2.adaptiveThreshold(waterimg, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 11, 2)
plt.imshow(threshwaterorigin)


# In[498]:


threshwaterorigin_water_median, distance = do_watershed(toimage(threshwater_ad_median,'RGB'), 200)
plt.imshow(threshwaterorigin_water_median)


# In[ ]:


image_median = Image.open(img_names[0])
threshwaterorigin_water, distance_origin_water = do_watershed(image_origin, 100)
plt.imshow(threshwaterorigin_water)


# ### Meanshit 

# In[459]:


def do_meanshift(img):
    img_mat = np.array(img)[:, :, :3]

    # Extract the three colour channels
    red = img_mat[:, :, 0]
    green = img_mat[:, :, 1]
    blue = img_mat[:, :, 2]

    # Store the shape so we can reshape later
    original_shape = red.shape
    colour_samples = np.column_stack([red.flatten(), green.flatten(), blue.flatten()])

    ms_clf = MeanShift(bin_seeding=True)
    ms_labels = ms_clf.fit_predict(colour_samples).reshape(original_shape)

    return ms_labels

