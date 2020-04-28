import tensorflow as tf
import matplotlib.pyplot as plt

def getData():
    return tf.keras.datasets.mnist.load_data()

def showImage(features, labels, index):
    print(labels[index])
    plt.imshow(features[index], cmap='Greys')
    
def showImages(features, labels, start, finish):
    for i in range(start, finish):
        print(labels[i])
        plt.imshow(features[i], cmap='Greys')