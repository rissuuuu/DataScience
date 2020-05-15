# -*- coding: utf-8 -*-
"""Convert_Color_Image_to_Sketch.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vcd0PcYBJ-RtS_iYr262c4yVfofzkuAY
"""

#Install imageio package
!pip install imageio

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import imageio
import requests
import matplotlib.pyplot as plt
import IPython.display as dp

#Display the image from the web using an URI.
img="https://www.lovethisimages.com/wp-content/uploads/2018/04/sorry-images-download-1.jpg"
dp.Image(requests.get(img).content)

#Read the image using imgaeio package's imread method and URI from web using 
source_img = imageio.imread(img)

# Make the image Black and White using the formula Y= 0.299*R + 0.587*G + 0.114*B i.e. applying greyscale
import numpy as np

def grayscaleimg(rgb): 
  return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

gryscl_img = grayscaleimg(source_img)

#Invert the image by subtracting it from 255
inv_img = (255 - gryscl_img)
plt.imshow(inv_img)

#Blur the image using gaussian filter
import scipy.ndimage
blurred_img = scipy.ndimage.filters.gaussian_filter(inv_img, sigma=5)
plt.imshow(blurred_img)

#Create function to perform dogding(belnding together greyscale and blurred image)
def dodging(blur_img, gryscl_img):
    resultant_dodge=blur_img*255/(255-gryscl_img) 
    resultant_dodge[resultant_dodge>255]=255
    resultant_dodge[gryscl_img==255]=255
    return resultant_dodge.astype('uint8')

#Generate the target image by applying the dodge
target_img= dodging(blurred_img, gryscl_img)

# Commented out IPython magic to ensure Python compatibility.
#Display the target image
# %matplotlib inline
import matplotlib.pyplot as plt
plt.imshow(target_img, cmap="gray")

#Save the image
plt.imsave('target_image.png', target_img, cmap='gray', vmin=0, vmax=255)

