#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import pdb
import sys
import math 
k = []
feat_object=cv2.xfeatures2d.SURF_create(400)
def compute_descr_of_1_point(img):
	img = cv2.resize(img, (256, 256))
	height, width = img.shape[:2]
	feat_object=cv2.xfeatures2d.SURF_create(400)
	for x in range (0, width):
		for y in range (0, height):

			pt=[cv2.KeyPoint(x,y,10)]
			out=feat_object.compute(img,pt)
			k.append(out[1][0])
	#print out[1][0]
	return k
	
feat_object.setUpright(True)
#img=cv2.imread("objects_to_cout.jpg")
#img=cv2.imread("test.jpg")
img=cv2.imread("b.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
k = compute_descr_of_1_point(img)

#np.save('descr1imagepix-test.npy', k)
#np.save('descr1imagepix-flos.npy', k)
#np.save('descr1px-im.npy', k)
np.save('b.npy', k)

# close all open windows 
cv2 . destroyAllWindows ( )
