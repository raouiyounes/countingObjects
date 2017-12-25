import cv2
import numpy as np
import math
import pdb


class Partition:
    def __init__(self,elements_dd):
        
        self.size=len(elements_dd.desco)
        self.data=elements_dd.desco
        self.pixel_whole=elements_dd.pixel
        self.index_of_all=elements_dd.im_idx

class CODEBOOK:
    
    
    def create_partition(self,element_p):
        self.partition.size=len(element_p.desco)
        self.partition.data=element_p.desco
        self.partition.pixel_whole=element_p.pixel
        self.partition.index_of_all=element_p.im_idx
    
        
        return self.partition
        
        
    def __init__(self,annot,elements):
        self.N=2
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
    def processing_for_partition(self):
        for i in self.set_of_partitions:
            self.median.append(self.get_feat_dim_max(self.set_of_partitions[i].data))
        
    def get_feat_dim_max(self,descriptors):
        #pdb.set_trace()
        variance=[0]*len(descriptors)
        desc_std_i=[0]*len(descriptors)
        for i in range(len(descriptors[0])):
            #print i
            temp=self.descriptors[:][i]
            desc_std_i[i]=np.std(temp)
        variance=max(desc_std_i)
        index_of_max_variance=desc_std_i.index(variance)
        self.median=0
        for i in range(len(self.annot_whole.pixel_whole)):
            
            x_ann=self.annot_whole.pixel_whole[i][0]
            y_ann=self.annot_whole.pixel_whole[i][1]
            idx_ann=self.annot_whole.idx_whole[i]
            index_of_desc=0
            while self.elements.pixel[index_of_desc].x != x_ann and self.elements.pixel[index_of_desc].y!=y_ann and self.elements.im_idx[index_of_desc]!=idx_ann:
               index_of_desc+=1     
                    
            
            self.median+=descriptors[index_of_desc][index_of_max_variance]
        self.median/=len(self.annot_whole.pixel_whole)
        return self.median/len(self.annot_whole.pixel_whole)


    def process(self):
        count=0
        for i in range(len(self.set_of_partitions)):
            if self.count_nb_annotation_part(i)>self.N:
                count+=1
        if count!=0:
            return True
        else:
            return False
            
    def remove_partition(self,i):
        del self.set_of_partitions[i]
    def split_partition(self):
        proc=True
        while proc==True:
            for j in range(len(self.set_of_partitions)):
                part=self.set_of_partitions[j]
                
                med=self.get_feat_dim_max(part.data)
                
                if self.count_nb_annotation_part(j)>0:
                    part_i_1=[]
                    part_i_2=[]
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
                    print med
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
                    
                    #data_i_1=[[0]*len(data_i_1_pile[0])]*len(data_i_1_pile)
                    #data_i_2=[[0]*len(data_i_2_pile[0])]*len(data_i_2_pile)
                    
                    data_i_1=list(data_i_1_pile)
                    data_i_2=list(data_i_2_pile)
                    e_i_1=Element(index_part_1,x_i_1,y_i_1,data_i_1)
                    e_i_2=Element(index_part_2,x_i_2,y_i_2,data_i_2)
                    
                    
                    p1=self.create_partition(e_i_1)
                    p2=self.create_partition(e_i_2)
                    self.remove_partition(j)
                    self.set_of_partitions.append(p1)
                    self.set_of_partitions.append(p2)
                    print "size of partition 0",self.set_of_partitions[0].size
            proc=self.process()        
# 400 is the threshold
# data is i*j*index of the image

class Pixel:
    def __init__(self,x_pix,y_pix):
        self.x=x_pix
        self.y=y_pix



class Element:
    def __init__(self,im_idx,x,y,descp):
        self.pixel=[]
        self.desco=[]
        self.im_idx=[]
        for i in range(len(descp)):
            self.pixel.append(Pixel(x[i],y[i]))                  
            self.desco.append(descp[i])
            self.im_idx.append(im_idx[i])

class Annot:
    def __init__(self,idx,pixels):
        self.idx_whole=[]
        self.pixel_whole=[]
        for i in range(len(idx)):
            self.idx_whole.append(idx[i])
            self.pixel_whole.append(pixels[i])

image_db=np.load('gray_images.npy')
feat_object=cv2.xfeatures2d.SURF_create(400)
data=np.zeros([256,256,len(image_db)])

el=[]
desc=[]
x=[]
y=[]
index=[]

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

anot_1=np.load('annotations256_globale.npy')

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
                
    
cbook=CODEBOOK(ann,el)
cbook.split_partition()
