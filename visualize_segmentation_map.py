import tifffile
import os
import cv2
import numpy as np



filepath = "/media/love/Love Extern/OUTPUT/BOMI2_TIL_1_Core[1,1,G]_[15997,34963]_binary_seg_maps.tif"

image = tifffile.imread(filepath)
print(image.shape)
print(image.dtype)
for i in range(4):
    imagei = image[i,:,:]
    imagei = imagei/np.amax(imagei)
    imagei = cv2.resize(imagei, (0, 0), fx = 0.1, fy = 0.1)
    cv2.imshow("image "+ str(i), imagei)

cv2.waitKey(0)
