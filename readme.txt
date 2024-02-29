Setup Guide:
1) setup directories in settings.json
  reltive paths only:
    dataFolder:
      a) testBC - testing data data folder
      b) Training data - Train - for training model / Test for testing
    trainFolder and testFolder:
      subdirs of dataFolder, must exist at the time of the training
    modelsFolder:
      folder for storing trained models
    logsFolder:
      folder for storing training logs
2)image preprocessing:
  for single image use imageProcessor.py method processImage()
  for bulk use imageProcessorWhear.py it will take all pictures from dataFolder and run processImage() over each also it will rotate picture if it is vertical
3) Training:
  for training run model.py
  it will take data from trainFolder and train model on them
4) Testing and validation
  for testing on a single file use run-evaluation.py - specify model path and picture path there and it will evaluate given image with help of given model
  for bulk testing use evaluate-bulk.py - it will take images from given folder (user must specify concrete folder with pictures, without subdirs), evaluate them and return plot with results and number of lodged/not lodged pictures. Model can also be specified
5) Testing on different files than given
  For testing on custom files run preprocess-custom.py over the file and then specify path to it in run-evaluation.py and run the script. It will return result for chosen picture