import cv2
from matplotlib import pyplot as plt
import numpy as np

im = cv2.imread('image1.jpg')

(b,g,r) =cv2.split(im)
r1 = im[:,:,2]
g1 = im[:,:,1]
b1 = im[:,:,0]

gray = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
# ori = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)

# cv2.imshow('ORIGIN',im)
# cv2.imshow('BLUE',b)
# cv2.imshow('GREEN',g)
# cv2.imshow('RED',r)
# cv2.imshow('GRAY',gray)
# cv2.waitKey(0)

color = ('b','g','r')

for i,col in enumerate(color) :
    hist = cv2.calcHist([im],[i],None,[256],[0,256])
    plt.plot(hist,color=col)
    plt.xlim([0,256])

# plt.hist(hist)
# plt.show()

(h,w) = im.shape[:2]
print(h,w)
center = (h/2,w/2)

M = cv2.getRotationMatrix2D(center,90,1.0)
rotated = cv2.warpAffine(im,M,(h,w))

# cv2.imshow('ROTATE',rotated)
# cv2.waitKey(0)

cropped = im[100:200,100:200]

# cv2.imshow('CROPPED',cropped)
# cv2.waitKey(0)

edge = np.array(([-1,-1,-1],[-1,8,-1],[-1,-1,-1]),dtype="int")
blur = np.array(([1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]))
emboss = np.array(([-2,1,0],[-1,1,1],[0,1,2]),dtype="int")

convolution = cv2.filter2D(im,-1,edge)
convolution1 = cv2.filter2D(im,-1,blur)
convolution2 = cv2.filter2D(im,-1,emboss)

plot_image = np.concatenate((im,convolution1),axis=1)
# plt.subplot(221),plt.imshow(im),plt.title('ORIGINAL')
# plt.subplot(222),plt.imshow(convolution1),plt.title('BLUR')
cv2.imshow("COMPARE",plot_image)
cv2.waitKey(0)
