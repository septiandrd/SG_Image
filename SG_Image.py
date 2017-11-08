import cv2
from matplotlib import pyplot as plt

im = cv2.imread('image1.jpg')

print(im.shape)

cv2.imshow('ORIGIN',im)

(b,g,r) =cv2.split(im)
r1 = im[:,:,2]
g1 = im[:,:,1]
b1 = im[:,:,0]

gray = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
ori = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)

# cv2.imshow('BLUE',b)
# cv2.imshow('GREEN',g)
# cv2.imshow('RED',r)
# cv2.imshow('GRAY',gray)
cv2.imshow('ORIfromGRAY',ori)
cv2.waitKey(0)
