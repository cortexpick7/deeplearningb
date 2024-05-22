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
picturesFolderDirGood = os.path.join(pathToDataFolder, "Good")
picturesFolderDirLodged = os.path.join(pathToDataFolder, "Lodged")
resultsGood = []
resultsLodged = []
lower = []
upper = []
lodged = 0
good = 0
count = 0



for fileName in os.listdir(picturesFolderDirGood):
  img = cv2.imread(os.path.join(picturesFolderDirGood, fileName))
  resize = tf.image.resize(img, (540, 960))
  evalResult = model.predict(np.expand_dims(resize/255, 0))
  if evalResult < 0.6:
   resultsGood.append(float(evalResult))
  count+=1



for fileName in os.listdir(picturesFolderDirLodged):
  img = cv2.imread(os.path.join(picturesFolderDirLodged, fileName))
  resize = tf.image.resize(img, (540, 960))
  evalResult = model.predict(np.expand_dims(resize/255, 0))
  if evalResult > 0.4:
    resultsLodged.append(float(evalResult))
  count+=1

for i in range (count):
  lower.append(0.43)
  upper.append(0.59)

plt.plot(resultsGood, 'g.', resultsLodged, "r.", upper, "b", lower, "b") 
plt.xlim(0, 61)
plt.title("Výsledky klasifikace obrázků ze dvou souborů dat")
plt.legend(["Dobrá pšenice", "Polehlá pšenice", "Oblast překryvu"], loc="upper right")
plt.show()
