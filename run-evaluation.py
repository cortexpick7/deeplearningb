import os
from keras.models import load_model
import cv2
import tensorflow as tf
import numpy as np

model = load_model('./models/mmm.h5')
aa = "./testBC/00239199.jpeg"
img = cv2.imread(os.path.join(aa))
ret,thresh2 = cv2.threshold(img,220,155,cv2.THRESH_BINARY_INV)
edges = cv2.Canny(thresh2, 100, 50)
resize = tf.image.resize(edges, (540, 960))

evalResult = model.predict(np.expand_dims(resize/255, 0))

print(evalResult)

if evalResult > 0.5:
  print('Lodged wheat')
else:
  print('Good wheat')