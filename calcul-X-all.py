#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import pdb
import sys
import math 
km=15
l= 256*256*19
X = np.zeros((l,km))
#k = np.load('descr_1image_pix.npy')
k = np.load('descr1pixel-allimages256_orient.npy')
kmen = np.load ('Kmeans256_centers15.npy')

d = []

def indic (vec,m):
	for x in range (0, km):
		if m==vec[x] :
    			return x

def distance(v1,v2): 
    return sum([np.power((x-y), 2) for (x,y) in zip(v1,v2)])**(0.5)

#distance = np.sqrt(np.sum((a-b)**2))
for i in range (0, l):
	d = []
	for j in range (0, km):
		d1 = distance(k[i], kmen[j])
		d.append(d1)
	
	mn = np.min(d)
	y=indic (d,mn)
	X[i][y]=1		 
		
		
print mn

	
"""
m = np.min(d)
for j in range (0, 3):
	if m==d[j]:
		print j
		X[0][j]=1
print len(d)
"""
#np.savetxt('X-allimages-k=5.txt',X , delimiter=" ", fmt="%s")
np.save('X_all_orient-k=15.np', X)

cv2 . destroyAllWindows ( )
