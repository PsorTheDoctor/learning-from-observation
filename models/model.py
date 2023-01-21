import tensorflow as tf


class TestModel(tf.keras.Model):

  def __init__(self, outputs):
    super().__init__()
    self.conv1 = tf.keras.layers.Conv2D(5, 4, input_shape=(224, 224, 3))
    # self.pool1 = tf.keras.layers.MaxPool2D(2)
    self.conv2 = tf.keras.layers.Conv2D(5, 4, activation='relu')
    self.pool2 = tf.keras.layers.MaxPool2D(2)
    self.conv3 = tf.keras.layers.Conv2D(5, 4, activation='relu')
    self.flatten = tf.keras.layers.Flatten()
    self.dense2 = tf.keras.layers.Dense(250, activation=tf.nn.relu)
    self.dense4 = tf.keras.layers.Dense(outputs)

  def call(self, inputs):
    x = self.conv1(inputs)
    # x = self.pool1(x)
    x = self.conv2(x)
    x = self.pool2(x)
    images = self.conv3(x)
    x = self.flatten(images)
    x = self.dense2(x)
    x = self.dense4(x)
    return x, images

class AlexNet(tf.keras.Model):

  def __init__(self, outputs):
    super().__init__()
    self.conv1 = tf.keras.layers.Conv2D(11, 96, input_shape=(224, 224, 2), strides=4, activation=tf.nn.leaky_relu)
    self.pool1 = tf.keras.layers.MaxPool2D(3, strides=2)
    self.conv2 = tf.keras.layers.Conv2D(5, 4, activation=tf.nn.leaky_relu)
    self.pool2 = tf.keras.layers.MaxPool2D(3)
    self.conv3 = tf.keras.layers.Conv2D(3, 1, activation=tf.nn.leaky_relu)
    self.conv4 = tf.keras.layers.Conv2D(3, 1, activation=tf.nn.leaky_relu)
    self.conv5 = tf.keras.layers.Conv2D(3, 1, activation=tf.nn.leaky_relu)
    self.pool3 = tf.keras.layers.MaxPool2D(3)
    self.flatten = tf.keras.layers.Flatten()
    self.dense2 = tf.keras.layers.Dense(500, activation=tf.nn.relu)
    self.dense4 = tf.keras.layers.Dense(outputs)

  def call(self, inputs):
    x = self.conv1(inputs)
    x = self.pool1(x)
    images = self.conv2(x)
    x = self.pool2(images)
    x = self.conv3(x)
    x = self.conv4(x)
    x = self.conv5(x)
    x = self.pool3(x)
    x = self.flatten(x)
    x = self.dense2(x)
    x = self.dense4(x)
    return x, images
