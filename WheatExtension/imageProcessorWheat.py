import cv2
import json
import os

from imageProcessor import processImage

# This script performs operation Canny to detect edges on images
# and writes cannied images to new files in same folder hierarchy

currPath = os.getcwd()
pathToSettings = os.path.join(currPath, "settings.json")
settingsJsonString = open(pathToSettings)
pathToDataFolder = json.load(settingsJsonString)["dataFolder"]
cannyImagesDir = os.path.join(currPath, pathToDataFolder, 'cannyImages')

# Make folder for cannied images
if os.path.exists(cannyImagesDir) is False:
  os.makedirs(cannyImagesDir)

# Get subset dir Train/Test
for subdirname in os.listdir(pathToDataFolder):
  imgNumber = 0
  if (subdirname != "cannyImages"):
    if os.path.exists(os.path.join(cannyImagesDir, subdirname)) is False:
      os.makedirs(os.path.join(cannyImagesDir, subdirname))
    # Get quality dir Good/Lodged
    if subdirname == "Test" or subdirname == "Train":
      for dirf in os.listdir(os.path.join(pathToDataFolder, subdirname)):
        for fileName in os.listdir(os.path.join(pathToDataFolder, subdirname, dirf)):
          imgNumber += 1
          img = cv2.imread(os.path.join(pathToDataFolder, subdirname, dirf,  fileName))
          height, width = img.shape[:2]
          if width < height:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
          edges = processImage(img)
          if os.path.exists(os.path.join(cannyImagesDir, subdirname, dirf)) is False:
            os.makedirs(os.path.join(cannyImagesDir, subdirname, dirf))
          cv2.imwrite(os.path.join(cannyImagesDir, subdirname, dirf, fileName), edges)
    else:
      for fileName in os.listdir(os.path.join(pathToDataFolder, subdirname)):
        imgNumber += 1
        img = cv2.imread(os.path.join(pathToDataFolder, subdirname,  fileName))
        height, width = img.shape[:2]
        if width < height:
          img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        edges = processImage(img)
        cv2.imwrite(os.path.join(cannyImagesDir, subdirname, fileName), edges)

print("Change paths to cannied images folders in settings? (y/n)")
decission = input()
while (decission.lower() != 'y' and decission.lower() != 'n'):
  print("Change paths to cannied images folders in settings? (y/n)")
  decission = input()
if (decission.lower() == "y"):
  settings = json.load(settingsJsonString)
  settings["trainFolder"] = os.path.join(cannyImagesDir, 'Train')
  settings["testFolder"] = os.path.join(cannyImagesDir, 'Test')
  with open(pathToSettings, 'w') as json_file:
    json.dump(settings, json_file)





