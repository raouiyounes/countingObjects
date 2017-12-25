#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import pdb
import sys
import math 
l= 256*256
km = 15
X = np.zeros((l,km))
#k = np.load('descr_1image_pix.npy')
#k = np.load('descr1imagepix-test.npy')
#k = np.load('descr1imagepix-flos.npy')
k = np.load('b.npy')
kmen = np.load ('Kmeans256_centers15.npy')
v = np.zeros(km)
d = []
V = []
def indic (vec,m):
	for x in range (0, km):
		if m==vec[x] :
    			return x

def distance(v1,v2): 
    return sum([np.power((x-y), 2) for (x,y) in zip(v1,v2)])**(0.5)

#distance = np.sqrt(np.sum((a-b)**2))
for i in range (0, l):
	d = []
	v = np.zeros(km)
	for j in range (0, km):
		d1 = distance(k[i], kmen[j])
		d.append(d1)
	
	mn = np.min(d)
	y=indic (d,mn)
	X[i][y]=1
	#v[y] = 1		 
	#V.append(v)
		
		
print mn
		
"""
m = np.min(d)
for j in range (0, 3):
	if m==d[j]:
		print j
		X[0][j]=1
print len(d)
"""
#print V[2]
#print len(V)
#np.savetxt('X1imageDeFLOS.txt',X , delimiter=" ", fmt="%s")
#np.save('Xtest-k=15', X)
#np.save('Vflos15' , V)
#np.save('Xflos-k=15', X)

np.save('Xb=15', X)
#np.save('Vim-k=15' , V)
cv2 . destroyAllWindows ( )
