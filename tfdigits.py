import data
import tensorflow as tf
from tensorflow import keras
import numpy as np

def plot_image(i, predictions_arr, true_label, img):
    return

(xTrain, yTrain), (xTest, yTest) = data.getData()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(units=32, activation='relu'),
    keras.layers.Dense(units=16, activation='relu'),
    keras.layers.Dense(10)
])

model.compile(optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

model.fit(xTrain, yTrain, epochs=10)

test_loss, test_acc = model.evaluate(xTest, yTest)
print('\nTest Accuracy: ', test_acc)

probability_model = tf.keras.Sequential([model, keras.layers.Softmax()])

predictions = probability_model.predict(xTest)
print(np.argmax(predictions[0]), ' vs ', yTest[0])