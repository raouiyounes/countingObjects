import cv2
import numpy as np
import math
import pdb
from scipy.spatial import distance
from matplotlib import cbook


# create an object partition that has size, data which are descriptors, the coordinates of the pixels and the indexes
class Partition:
    def __init__(self,elements_dd):
        
        self.size=len(elements_dd.desco)
        self.data=elements_dd.desco
        self.pixel_whole=elements_dd.pixel
        self.index_of_all=elements_dd.im_idx

# CReates a codebook object
class CODEBOOK:
    
    # creates an instance of a partition
    def create_partition(self,element_p):
        
        p=Partition(element_p)
           
        return p
        
    # the constructor that receives tha annotatons and the elements
    def __init__(self,annot,elements):
        self.N=16
        self.median=0
        self.annot_whole=annot
        self.descriptors=elements.desco
        self.elements=elements
        self.index_of_max_variance=0
        
        self.partition=Partition(elements)
        self.set_of_partitions=[]
        self.set_of_partitions.append(self.partition)
        self.median=[]
         # find the partitions of more that N annotated descriptors
    def count_nb_annotation_part(self,index_of_partition):
        part=self.set_of_partitions[index_of_partition]
        count=0
        s1=0
        for i in range(part.size):
            for j in range(len(self.annot_whole.idx_whole)):
                if self.annot_whole.idx_whole[j]==part.index_of_all[i] and self.annot_whole.pixel_whole[j][0]==part.pixel_whole[i].x and self.annot_whole.pixel_whole[j][1]==part.pixel_whole[i].y:
                    count+=1
        return count
    # you have to use the data of the partition not the whole data
    def get_feat_dim_max(self,partition_i):
        #pdb.set_trace()
        
        descriptors=partition_i.data
        pixel_w_part=partition_i.pixel_whole
        index_of_part=partition_i.index_of_all
    
        variance=[0]*len(descriptors)
        desc_std_i=[0]*len(descriptors)
        for i in range(len(descriptors[0])):
            #print i
            temp=self.descriptors[:][i]
            desc_std_i[i]=np.std(temp)
        variance=max(desc_std_i)
        index_of_max_variance=desc_std_i.index(variance)
        median=0
        for i in range(len(self.annot_whole.pixel_whole)):
            
            x_ann=self.annot_whole.pixel_whole[i][0]
            y_ann=self.annot_whole.pixel_whole[i][1]
            idx_ann=self.annot_whole.idx_whole[i]
            chosen_index=0
            for index_of_desc in range(len(pixel_w_part)):
                if pixel_w_part[index_of_desc].x == x_ann and pixel_w_part[index_of_desc].y==y_ann and index_of_part[index_of_desc]==idx_ann:
                    chosen_index=index_of_desc
            median+=descriptors[chosen_index][index_of_max_variance]
        median/=len(self.annot_whole.pixel_whole)
        return median
    # process that verify that the number of the annotations in a partition is higher that N
    def process(self):
        count=0
        for i in range(len(self.set_of_partitions)):
            if self.count_nb_annotation_part(i)>self.N:
                count+=1
        if count!=0:
            return True
        else:
            return False
    # removes a partition from the set of partitions when it is divided into two ones
    def remove_partition(self,i):
        del self.set_of_partitions[i]
    
    #computes the mean of the descriptor's partition
    def compute_mean_of_desc_parti(self):
        for i in range(len(self.set_of_partitions)):
            desc_of_part=cbook.set_of_partitions[i].data
            self.set_of_partitions[i].mean_desc=np.mean(desc_of_part,axis=0)
    
    # this loops overs all the patitions and split them using the algorithm of Arteta et al

    def split_partition(self):
        proc=True
        while proc==True:
            for j in range(len(self.set_of_partitions)):
                
                if self.count_nb_annotation_part(j)>self.N:
                    part=self.set_of_partitions[j]
                    self.remove_partition(j)
                
                    med=self.get_feat_dim_max(part)
                    k_1=0
                    k_2=0
                    pixels_part_1=[]
                    index_part_1=[]
                    x_i_1=[]
                    y_i_1=[]
                    x_i_2=[]
                    y_i_2=[]
                    index_part_2=[]
                    data_i_1_pile=[]
                    data_i_2_pile=[]
                    for i in range(len(part.data)):
                        if (part.data[i][self.index_of_max_variance]<med):
                            data_i_1_pile.append(part.data[i][:])
                            x_i_1.append(part.pixel_whole[i].x)
                            y_i_1.append(part.pixel_whole[i].y)
                            index_part_1.append(part.index_of_all[i])                            
                            k_1+=1
                        else:
                            data_i_2_pile.append(part.data[i][:])
                            x_i_2.append(part.pixel_whole[i].x)
                            y_i_2.append(part.pixel_whole[i].y)
                            index_part_2.append(part.index_of_all[i])
                            k_2+=1
                    
                     
                    
                    data_i_1=list(data_i_1_pile)
                    data_i_2=list(data_i_2_pile)
                    p=[]
                    if len(data_i_1)!=0:
                        e=Element(index_part_1,x_i_1,y_i_1,data_i_1)
                        p=self.create_partition(e)
                        self.set_of_partitions.append(p)
                        
                        
                        #self.set_of_partitions.append(p)
                    p=[]
                    if len(data_i_2)!=0:
                        e=Element(index_part_2,x_i_2,y_i_2,data_i_2)
                        p=self.create_partition(e)
                        self.set_of_partitions.append(p)
                    
            s=0
            for i in range(len(self.set_of_partitions)):
                s+=len(self.set_of_partitions[i].data)
            
            print "size is : ",s
            proc=self.process()
        print "fin du partitionnement"
# 400 is the threshold
# data is i*j*index of the image
    
    # looks for the partition that is the nearest to the descriptor of each pixel of the database
    
    def find_partition_4(self,desc_i):
        distance_between_descriptors=[0]*len(self.set_of_partitions)
        voted_dim=0
        for i in range(len(self.set_of_partitions)):
            distance_between_descriptors[i]=np.linalg.norm(desc_i- self.set_of_partitions[i].mean_desc)
        min= distance_between_descriptors[0]
        for j in range(len(distance_between_descriptors)):
            if distance_between_descriptors[j]<min:
                min=distance_between_descriptors[j]
                voted_dim=j
        return voted_dim
        
""" 
    
    def find_partition_3(self,desc_i):
        #pdb.set_trace()
        desco=[0]*len(desc_i)
        
        for i in range(len(desc_i)):
            desco[i]=desc_i[i]
        
   
      
        partitions=self.set_of_partitions
        part_i=self.set_of_partitions[0].data
        dist_i=distance.cdist([desc_i],part_i,'cityblock')
        min_i=min(np.transpose(dist_i))
        index_i=0
    
        
        for i in range(1,len(partitions)):
            part_i=self.set_of_partitions[i].data
            dist_i=distance.cdist([desc_i],part_i,'cityblock')
            min_of_dist_desc_part=min(np.transpose(dist_i))
            if min_i>min_of_dist_desc_part[0]:
                index_i=i
        #print "index in the dico",index_i
        return index_i
    def find_partition(self,desc_i,partitions):
        #print "longueur du desc",len(desc_i)
        desco=[0]*len(desc_i)
        for i in range(len(desc_i)):
            desco[i]=desc_i[i]
        #print desc_i
        #partitions=self.set_of_partitions
        difference_of_desc=[d1-d2 for d1,d2 in zip(partitions[0].data[0][:],desco)]
        euclidean_distance_data=np.std(difference_of_desc)
        min_index_partition=0
        for i in range(1,len(partitions)):
            for k in range(1,len(partitions[i].data)):
                difference_of_desc=[d1-d2 for d1,d2 in zip(partitions[i].data[k][:],desco)]
                if np.std(difference_of_desc)<euclidean_distance_data:
                    euclidean_distance_data=np.std(difference_of_desc)
                    min_index_partition=i
        
        print min_index_partition
    def find_partition_2(self,desc_i):
        print "jkjkjkjkj"
        desco=[0]*len(desc_i)
        for i in range(len(desc_i)):
            desco[i]=desc_i[i]
        partitions=self.set_of_partitions
        difference_of_desc=[d1-d2 for d1,d2 in zip(partitions[0].data[0][:],desco)]
        euclidean_distance_data=np.std(difference_of_desc)
        min_index_partition=0
        for i in range(1,len(partitions)):
            for k in range(1,len(partitions[i].data)):
                difference_of_desc=[d1-d2 for d1,d2 in zip(partitions[i].data[k][:],desco)]
                if np.std(difference_of_desc)<euclidean_distance_data:
                    euclidean_distance_data=np.std(difference_of_desc)
                    min_index_partition=i
        print min_index_partition
        return min_index_partition
"""

class Pixel:
    def __init__(self,x_pix,y_pix):
        self.x=x_pix
        self.y=y_pix


# creates an element that contains index of the image of the database, the corresponding pixel and its descriptor
class Element:
    def __init__(self,im_idx,x,y,descp):
        self.pixel=[]
        self.desco=[]
        self.im_idx=[]
        for i in range(len(descp)):
            self.pixel.append(Pixel(x[i],y[i]))                  
            self.desco.append(descp[i])
            self.im_idx.append(im_idx[i])

# class of the annotations of created in the entire database 
class Annot:
    def __init__(self,idx,pixels):
        self.idx_whole=[]
        self.pixel_whole=[]
        for i in range(len(idx)):
            self.idx_whole.append(idx[i])
            self.pixel_whole.append(pixels[i])



# load the database gray_imagesBR.npy 
image_db=np.load('gray_imagesBR.npy')
feat_object=cv2.xfeatures2d.SURF_create(400)
data=np.zeros([256,256,len(image_db)])

el=[]
desc=[]
x=[]
y=[]
index=[]

#creates the poses of each pixel and its descriptor

for i in range(3):
    img=cv2.resize(image_db[i],(256,256))
    height,width=img.shape[:2]
    for x_i in range(0,width):
        for y_i in range(0,height):
            pt=[cv2.KeyPoint(x_i,y_i,10)]
            desc_i=feat_object.compute(img,pt)
            desc.append(desc_i[1][0])
            x.append(x_i)
            y.append(y_i)
            index.append(i)
            
            
el=Element(index,x,y,desc)


# loading of the annotation file

anot_1=np.load('annotationGlobalBR.npy')

index_of_annot=0
pixels=[]
index_i=[]
for i in range(len(anot_1)):
    anot_of_i_image=anot_1[i]
    for j in range(len(anot_of_i_image)):
        pix=[0]*2
        pix[0]=anot_of_i_image[j][0]
        pix[1]=anot_of_i_image[j][1]
        pixels.append(pix)
        index_of_annot+=1
        index_i.append(i)
ann=Annot(index_i,pixels)
                
# constructs an object of CODEBOOK

cbook=CODEBOOK(ann,el)

# Create the partition as in Arteta et al 
cbook.split_partition()


X=np.zeros([(len(image_db)*256*256),len(cbook.set_of_partitions)])
    

for i in range(len(cbook.set_of_partitions)):
    data=cbook.set_of_partitions[i].data
    cbook.set_of_partitions[i].mean_desc=np.mean(data,axis=0)
# construct the new object cbook by adding another attribute mean_desc
cbook_with_mean=cbook 
# Compute the X matrix   
index_of_partition=[]
index_of_pixel=0
for i in range(len(image_db)):
    img=cv2.resize(image_db[i],(256,256))
    height,width=img.shape[:2]
    for x_i in range(0,width):
        for y_i in range(0,height):
            print "pixel x,pixel y, image db index",x_i,y_i,i
            pt=[cv2.KeyPoint(x_i,y_i,10)]
            desc_i=feat_object.compute(img,pt)
            index_of_partition.append(cbook_with_mean.find_partition_4(desc_i[1][0]))
            voted_word=cbook_with_mean.find_partition_4(desc_i[1][0])
            X[index_of_pixel][voted_word]=1
            index_of_pixel+=1
print X

# size of the image 256*256 in grayscale levels
# F(p)= x_requete*w

def calcul_x_requete(cbook):
    x_requete=np.zeros([256*256],len(cbook.set_of_partitions))
    img=cv2.imread("image_requete.jpg")
    img=cv2.resize(img,(256,256))
    height,width=img.shape[:2]
    index_of_image_pixel_i=0
    for x_i in range(0,width):
        for y_i in range(0,height):
            pt=[cv2.KeyPoint(x_i,y_i,10)]
            desc_i=feat_object.compute(img,pt)
            voted_word=cbook_with_mean.find_partition_4(desc_i[1][0])
            X_requete[index_of_image_pixel_i][voted_word]=1
            index_of_image_pixel_i+=1
    return x_requete
            
