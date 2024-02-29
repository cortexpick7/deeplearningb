import json
from keras.models import load_model
import cv2
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import os


currPath = os.getcwd()
pathToSettings = os.path.join(currPath, "settings.json")
settingsJsonString = open(pathToSettings)
pathToDataFolder = json.load(settingsJsonString)["testFolder"]

model = load_model('./models/mmm.h5')
picturesFolderDir = os.path.join(pathToDataFolder, "Good")

results = []

lodged = 0
good = 0
count = 0

for fileName in os.listdir(picturesFolderDir):
  img = cv2.imread(os.path.join(picturesFolderDir, fileName))
  resize = tf.image.resize(img, (540, 960))
  evalResult = model.predict(np.expand_dims(resize/255, 0))
  results.append(float(evalResult))
  count+=1
  if evalResult > 0.5:
    lodged+=1
  else:
    good+=1
resultOrder = [i for i in range(1, count+1)]

plt.plot(resultOrder, results)  # Plot numbers
plt.show()

print("Lodged: " + str(lodged)) 
print("Good: " + str(good))