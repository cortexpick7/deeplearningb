import json
import os
import cv2



def processImage(img):
  ret,thresh2 = cv2.threshold(img,220,155,cv2.THRESH_BINARY_INV)
  edges = cv2.Canny(thresh2, 100, 50)
  return edges