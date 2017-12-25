#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import pdb
import cPickle as pickle
import sys
import numpy as np
from numpy.linalg import eig, inv
"""
img=cv2.imread("objects_to_cout.jpg")
#img=cv2.imread("im5.jpg")
### when i resize image to 256x256 and i tack sig=2 give a good result
image = cv2.resize(img, (256, 256))
imag = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

height, width = imag.shape[:2]


clickcord   =   [ ]  
pixlcord = [ ]
i = 0
mat = [ ] 
"""#
"""
def   on_mouse ( event ,   x ,   y ,   flags ,   param ) : 
	 # grab references to the global variables 
		
	 global   clickcord 
	 
	 # if the left mouse button was clicked, record the starting 
	 # (x, y) coordinates 
	 # performed 
	 if   event   ==   cv2 . EVENT_LBUTTONDOWN : 
		 #cordclick   =   [ ( x ,   y ) ] 
		 
		 cv2.circle(imag, (x, y), 2, (255, 255, 255), 2)  # draw center of circle
		 clickcord . append ([x , y ])
		 #anoxx.append(x)
"""
#for x in range (0,width):
		
#	for y in range (0,height):
#		pixlcord.append([x , y ])
"""
while   True : 
	 cv2 . imshow ( "image" ,   imag )
	  
		
	 key   =   cv2 . waitKey ( 1 )   &   0xFF 
 
	 # if the 'q' key is pressed, break from the loop
	 if   key   ==   ord ( "q" ) : 

		 break 

 
	 # if the 'c' key is pressed for mor annotations
	 elif  key == ord ("c") : 
		#cv2 . namedWindow ( "image" )
		
		cv2 . setMouseCallback ( "image" ,   on_mouse )
"""
height, width = 256, 256		
click= np.load('annotations256_globale.npy')		
print len(click)	



#cov =np.array([[2,0],[0,2]])
#covT =cov.T

#covinv =inv(np.power(cov,2))
#covdet = np.linalg.det(cov)
 
F0 = []
F00 = []
F000 = [] #density of object for every pixls in image
#print (covinv**2)
"""
def sigma2fwhm(sigma):
     return sigma * np.sqrt(8 * np.log(2))



def fwhm2sigma(fwhm):
    return fwhm / np.sqrt(8 * np.log(2))

"""
fwhm = 4

print len(click)
	#F00 = []
	#g = 0.0
	#f1 = np.zeros((2,2))
for i in range (len(click)):
	clickcord = click[i]
	F0 = []
	for x in range (0,width):
			#
		for y in range (0,height):
			f1 = 0.0
			for c in range (len(clickcord)):
					#a = 0.0
					#b = 0.0
				#Gaussian = np.zeros((2,2))		
				G = 0.0
				a = x - clickcord[c][0]
				b = y - clickcord[c][1]
					#XT = np.transpose(X)
				#vexp = -0.5*(np.power(a,2) + np.power(b,2))*covinv #la valeur du exp
				#coeffexp = np.exp(vexp)
				#Gaussian =(1/(2*np.pi))*covinv *coeffexp
				G = (8*np.log(2)/(2*(fwhm**2)*np.pi))*(np.exp(-4*np.log(2) * ((a)**2 + (b)**2) / fwhm**2))
				#G = np.random.normal(np.sqrt(np.power(a,2) + np.power(b,2)), 1.69, 1)
			   	#f1 =f1 + Gaussian[0][0]
				f1 = f1 + G
					#F00.append(Gaussian[0][0])
		
			F0.append(f1)
			
			#np.save('y-img[i].npy', F0)		
	F00.append(F0)
	print np.sum(F0)		
YY = np.concatenate(F00)
np.save('Y-sigma1-6_globale.npy' , F00)	
np.save('YY-sigma1-6.npy', YY)
	#F00.append(f1)
	#g= np.sum(F00)/4096
	#print (np.sum(F00))
	
	#print ("\n\n 1cercle")
	#Gt = Gt + g
	#print (Gt)


	
print (len(F00))
print ("\n\n")
print ('longeur total de YY: %s',len(YY))
	
print (np.sum(F00))
print (np.sum(YY))	
#np.save('Yim5.npy', F0)
#np.save('F000-1im5.npy', F000)
#4.02920569639

#print k
#def Gaussian 
