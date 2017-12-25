#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import cv2
import argparse 
import numpy as np
import cPickle as pickle
import pdb
import os
import sys
import glob
#from os import listdir #pour lire d'un dossier
#from os.path import isfile, join #pour lire d'un dossier

#pdb.set_trace()

#descriptors_f.close ()



#pour mettre toutes les images
imagesI = [] 
# pour les images en gris et redimentionnées 512 x 288 
gray_images = [] 

def laodimages_f ( folder ):
	for img in glob.glob(folder):
	    	n= cv2.imread(img)
		#if n is not None:
		imagesI.append(n)
		image = cv2.resize(n, (256, 256))
		gray_n = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		gray_images.append(gray_n)
	
	return gray_images

folder ='./images/*.jpg'
gray_images = laodimages_f ( folder )
print len(imagesI)
print len (gray_images)
np.save('gray256_images.npy',gray_images)

