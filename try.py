# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 14:39:04 2019

@author: MEDHA
"""

import cv2
import os
cv2.resizeWindow("a", 200, 100) 
image = cv2.imread("abc.jpg")
cv2.imshow('a',image)
cv2.waitKey()
cv2.imwrite(os.path.join('xyz.jpg' ), image)
#cv2.destroyAllWindows()
