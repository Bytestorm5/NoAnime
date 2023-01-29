from cProfile import label
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense,Flatten,Conv2D,MaxPooling2D,Dropout
from keras.models import Sequential
from keras.optimizers import Adam,SGD
import tensorflow as tf
import os

img_height,img_width=256,256
batch_size=32
train_ds = tf.keras.utils.image_dataset_from_directory(
  "output", #set to "data\\Train" for preprocessed images
  validation_split=0.1,
  subset="training",
  seed=256,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  labels="inferred",
  label_mode="binary",
  shuffle=True)

val_ds = tf.keras.utils.image_dataset_from_directory(
  "output", #set to "data\\Test" for preprocessed images
  validation_split=0.1, #Issue: Images aren't split properly when not using a dedicated folder for validation images
  subset="validation",
  seed=256,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  labels="inferred",
  label_mode="binary",
  shuffle=True)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(6):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(str(labels[i]))
    plt.axis("off")
plt.show()

resnet_model = Sequential()
if os.path.isdir("final_model"):
    resnet_model = tf.keras.models.load_model("final_model")
    resnet_model.summary()
else:
    pretrained_model= tf.keras.applications.ResNet50(include_top=False,
                    input_shape=(256,256,3),
                    pooling='avg',classes=2048,
                    weights='imagenet')
    for layer in pretrained_model.layers:
             layer.trainable=True

    resnet_model.add(pretrained_model)
    resnet_model.add(Dense(512, activation='relu'))
    resnet_model.add(Dense(64, activation='relu'))
    resnet_model.add(Dense(1, activation='tanh'))
    resnet_model.add(Dense(1, activation='sigmoid'))

    # resnet_model.add(Conv2D(32, (3, 3), activation='relu',
    #                     input_shape=(256, 256, 3)))
    # resnet_model.add(MaxPooling2D((2, 2)))
    # resnet_model.add(Conv2D(64, (3, 3), activation='relu'))
    # resnet_model.add(MaxPooling2D((2, 2)))
    # resnet_model.add(Conv2D(128, (3, 3), activation='relu'))
    # resnet_model.add(MaxPooling2D((2, 2)))
    # resnet_model.add(Flatten())
    # resnet_model.add(Dense(512, activation='relu'))
    # resnet_model.add(Dense(1, activation='sigmoid'))

    resnet_model.summary()

    probability_thresholds = np.linspace(0, 1, num=1000).tolist()
    resnet_model.compile(optimizer=SGD(learning_rate=10**-3),loss='binary_crossentropy',metrics=['accuracy', tf.keras.metrics.Recall(name="recall", thresholds=probability_thresholds), tf.keras.metrics.Precision(name="precision", thresholds=probability_thresholds)])

history = resnet_model.fit(train_ds, validation_data=val_ds,epochs=20)

resnet_model.save("final_model")

fig1 = plt.gcf()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.axis(ymin=0.4,ymax=1)
plt.grid()
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'])
plt.show()

plt.plot(history.history['recall'][-1], history.history['precision'][-1])
plt.plot(history.history['val_recall'][-1], history.history['val_precision'][-1])
plt.title('model precision vs. recall')
plt.xlabel("recall")
plt.ylabel("precision")
plt.legend("train", "validation")
plt.show()