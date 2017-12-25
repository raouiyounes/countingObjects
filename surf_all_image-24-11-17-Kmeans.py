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


#pour mettre toutes les images
imagesI = [] 
# pour les images en gris sont redimentionnées en (64 x 64) 
gray_images = [] 
alldiscriptors = []

"""
#cv2.imshow("gray_imag", gray_im)

#Detect keypoints and descriptors in greyscale image

surf = cv2.xfeatures2d.SURF_create(400)
#keypoints= surf.detect(gray_im, None)
print surf.getUpright()
#Pour fixe l'orientation de surf
surf.setUpright(True)
for i in range(0 , len(gray_images)):
	keypoints= surf.detect(gray_images[i], None)
	(keypoints,descriptors)=surf.detectAndCompute(gray_images[i],None)
	#print('image %d: %d keypoints, et : %d features' % (i, len(keypoints), len(descriptors)))
	#descriptors_f=open('desc(%d).txt' %i,"w")
	alldiscriptors.append(descriptors)
	#for line_desc in descriptors:
	#	for entry in line_desc:
	#		descriptors_f.write(str(entry))
	#		descriptors_f.write("\n")


#descriptors_f.close ()

#np.savetxt('alldescriptors256.txt', alldiscriptors, delimiter=" ", fmt="%s")
	
#print alldiscriptors

concatenated = np.concatenate(alldiscriptors)
np.save('alldescriptors256_orient.npy', concatenated)	
print('Number of descriptors: {}'.format(len(concatenated)))
#concatenated = concatenated[::64]
#print('Number of descriptors: {}'.format(len(concatenated)))
"""
#np.savetxt('save256_concatenated.txt', concatenated, delimiter=" ", fmt="%s")
# Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
concatenated = np.load('alldescriptors256_orient.npy')
#changement de 10 par 1.0e-4
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0e-4)

# Set flags
flags = cv2.KMEANS_RANDOM_CENTERS
KMeans = []
# Apply KMeans whith k=20
#compactness,labels,centers = cv2.kmeans(concatenated,20,None,criteria,10,flags)

# Apply KMeans whith k=5
compactness,labels,centers = cv2.kmeans(concatenated,15,None,criteria,100,flags)


#Save output centers of kmeans
#np.savetxt('Kmeans256_centers5.txt', centers, delimiter=" ", fmt="%s")
np.save('Kmeans256_--centers15.npy', centers)
#print('centers: %s' %centers)   

print('compactness: %s' %compactness)
print ('leng of centers: %d' %len(centers))
#print centers[0][63]
#Display colour image with detected features
#cv2.imshow("features", image)

#cv2.waitKey(0)

# close all open windows 
cv2 . destroyAllWindows ( )

