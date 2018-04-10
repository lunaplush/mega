# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 12:16:44 2018

@author: Luna
"""

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np

from skimage import data, color, feature 
import skimage.data

image = color.rgb2gray(data.chelsea())
hog_vec, hog_vis = feature.hog(image, visualise = True)

fig, ax = plt.subplots(1, 2, figsize=(12,6), subplot_kw =  dict(x_ticks = [], y_ticks = []))
ax[0].imgshow(image,cmap='gray')
ax[0].set_title('input image')
ax[1].imgshow(hog_vis)
ax[1].set_title('visualisation of HOG features')