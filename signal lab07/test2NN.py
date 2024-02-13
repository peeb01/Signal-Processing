import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from numpy.linalg import inv
import tensorflow as tf
import pandas as pd

img1 = mpimg.imread('happyCK.png').flatten()
img2 = mpimg.imread('angerCK.png').flatten()
img3 = mpimg.imread('disgustCK.png').flatten()
img4 = mpimg.imread('fearCK.png').flatten()
img5 = mpimg.imread('neutralCK.png').flatten()
img6 = mpimg.imread('sadCK.png').flatten()
img7 = mpimg.imread('surpriseCK.png').flatten()

data = np.array([img1, img2, img3, img4, img5, img6, img7])

y = [1,2,3,4,5,6,7]
y_ = tf.constant(pd.get_dummies(y).values.astype(np.float32))

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, input_shape=(data.shape[1],), activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(128, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(32, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(48, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(7, activation=tf.keras.activations.softmax)
])

model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

history = model.fit(data, y_, epochs=500, batch_size=8, validation_split=0.2)

img7 = img7.reshape(1, -1)
print(img7.shape)
model.predict(img7)