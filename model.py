
import json
from matplotlib import pyplot as plt
import tensorflow as tf


from keras.metrics import Precision, Recall, BinaryAccuracy
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
import os
#https://evileg.com/ru/post/619/

# Set random seed for purposes of reproducibility
seed = 12
pathToSettings = 'settings.json'

testImages = []
trainImages = []

settingsJsonString = open(pathToSettings)
settings = json.load(settingsJsonString)

# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#   tf.config.experimental.set_memory_growth(gpu, True)

# Load data as numpy array
data = tf.keras.utils.image_dataset_from_directory(settings["trainFolder"], image_size=(540, 960))

# Returns an iterator which converts all elements of the dataset to numpy
data_iterator = data.as_numpy_iterator()

batch = data_iterator.next()

# Check batch shape, in our case 32 pictures 540x960 HxW
print(batch[0].shape)

# By default pixels in numpy array represent number from 0 to 255.
# We want those numbers to be as small as possible, so we divide each pixel representing number by 255.
# We first translate number from int to float by casting it, then divide.
# As a result we will have floating numbers from 0 to 1.

# Here we map through all pictures and divide their x by 255, so now on we have number from 0 to 1 there
data = data.map(lambda x,y: (x/255, y))


# We separating batch for purpose of training model. In each epoch 70% of pictures will be used to train, 20% 
train_size = int(len(data) * .7)
val_size = int(len(data) * .3)


train = data.take(train_size)
val = data.skip(train_size).take(val_size)

model = Sequential()

model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(540, 960, 3)))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Dropout(seed=12312, rate=0.2))

model.add(Flatten())

model.add(Dense(1, activation='sigmoid'))

model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
model.summary()


precis = Precision()
rec = Recall()
acc = BinaryAccuracy()


logdir = settings["logsFolder"]
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

hist = model.fit(train, epochs=7, validation_data=val, callbacks=[tensorboard_callback])

fig = plt.figure()
plt.plot(hist.history['loss'], color='red', label='loss')
plt.plot(hist.history['val_loss'], color='black', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc='upper right')
plt.show()

fig = plt.figure()
plt.plot(hist.history['accuracy'], color='red', label='Accuracy')
plt.plot(hist.history['val_accuracy'], color='black', label='val_acc')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc='upper left')
plt.show()

print("Enter name for model: ")
model_name = input()
model.save(os.path.join(settings['modelsFolder'], model_name + ".h5"))