import cv2
import numpy as np
from scipy.spatial import Delaunay
image=cv2.imread('test3.png')
image=cv2.resize((image), (800, 600))
image1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image12=image1.copy()
image14=image1.copy()
_,image1=cv2.threshold(image1,127,255,cv2.THRESH_BINARY)
image= cv2.ximgproc.thinning(image1,thinningType =cv2.ximgproc.THINNING_ZHANGSUEN)
image2=(image)
image1[image2 == 255] = [0]
cv2.imshow('image', image1)
cv2.imshow('thinning', image2)

count=np.sum(image2==255)
print("Number of white pixels in the thinned image:", count)


image10=(image2 > 0).astype(np.uint8)
filtre= np.array([[1,1,1],[1,10,1],[1,1,1]])
conv = cv2.filter2D(image10, -1, filtre, borderType=cv2.BORDER_CONSTANT)
end=np.argwhere((image10==1) & (conv==11))
branch=np.argwhere((image10==1) & (conv>=13))
normal=np.argwhere((image10==1) & (conv==12))
image11=cv2.cvtColor((image10*255),cv2.COLOR_GRAY2BGR)
for y,x in end:
    cv2.circle(image11, (x,y), 10, (0,0,255), -1)   # red
for y,x in branch:
    cv2.circle(image11, (x,y), 1, (255,0,0), -1)   # green
for y,x in normal: 
    cv2.circle(image11, (x,y), 1, (255,0,0), -1)   # blue   
cv2.imshow('EndPoints', image11)
image1=cv2.cvtColor(image1,cv2.COLOR_GRAY2BGR)
mask=np.all(image11==[0,0,255],axis=-1)
image1[mask]=[0,0,255]
cv2.imshow('merge', image1)
# cv2.waitKey(20000)
_,image12=cv2.threshold(image12,127,255,cv2.THRESH_BINARY)

contours, _ = cv2.findContours(image12, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
image12=cv2.cvtColor(image12,cv2.COLOR_GRAY2BGR)
print("Number of contours found:", len(contours))
contour=max(contours, key=cv2.contourArea)
cv2.drawContours(image12, [contour], 0, (0,255,0), 3)
# cv2.imshow('contours', image12)
cv2.waitKey(0)
# pts = contour.reshape(-1, 2)
# print(pts.shape)
# tri = Delaunay(pts)

# vis = image14.copy()
# for t in tri.simplices:
#     p1, p2, p3= pts[t[0]], pts[t[1]], pts[t[2]]
#     cv2.polylines(vis, [np.array([p1,p2,p3])], True, (0,255,0), 1)

# cv2.imshow('mesh', vis)
# cv2.waitKey(0)
# cv2.imshow('image',image14)
