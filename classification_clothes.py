import tensorflow as tf
# Import TensorFlow Datasets
import tensorflow_datasets as tfds
# Helper libraries
import math
import numpy as np
import matplotlib.pyplot as plt
import logging

# Configure some settings for the libs
tfds.disable_progress_bar()
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# ---------------------------- GET, PREPROCESS AND EXPLORE THE DATA ----------------------------------

# Get the Fashion MNIST data from the Datasets API:
dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
# The Dataset includes training (60000) and testing (10000) items. Get the items:
train_dataset, test_dataset = dataset['train'], dataset['test']

# Each image is mapped to a single label. Since the class names are not included with the dataset, store them here to
# use later when plotting the images.
# 0	T-shirt/top
# 1	Trouser
# 2	Pullover
# 3	Dress
# 4	Coat
# 5	Sandal
# 6	Shirt
# 7	Sneaker
# 8	Bag
# 9	Ankle boot
class_names = metadata.features['label'].names
print("Class names: {}".format(class_names))

# Explore the data that we get
num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples
print("Number of training examples: {}".format(num_train_examples))
print("Number of test examples:     {}".format(num_test_examples))


# The value of each pixel in the image data is an integer in the range [0,255]. For the model to work properly, these
# values need to be normalized to the range [0,1]
def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels


# Apply the normalize function to each element in the train and test datasets
train_dataset = train_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)

# The first time you use the dataset, the images will be loaded from disk. Caching will keep them in memory, making
# training faster
train_dataset = train_dataset.cache()
test_dataset = test_dataset.cache()

# Plot 1 image to explore the processed data. Take a single image, and remove the color dimension by reshaping
for image, label in test_dataset.take(1):
    break

# Plot the image - This is a single item of fashion clothing
plt.figure()
plt.imshow(image, cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show()

# Plot several images from the dataset
plt.figure(figsize=(10, 10))
for i, (image, label) in enumerate(test_dataset.take(25)):
    image = image.numpy().reshape((28, 28))
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image, cmap=plt.cm.binary)
    plt.xlabel(class_names[label])
plt.show()

# ----------------------------- BUILD, COMPILE AND TRAIN THE MODEL ------------------------------------

# This network has three layers:
# input tf.keras.layers.Flatten — This layer transforms the images from a 2d-array of 28  ×  28 pixels, to a 1d-array
# of 784 pixels (28*28). Think of this layer as unstacking rows of pixels in the image and lining them up. This layer
# has no parameters to learn, as it only reformats the data.
#
# "hidden" tf.keras.layers.Dense— A densely connected layer of 128 neurons. Each neuron (or node) takes input from all
# 784 nodes in the previous layer, weighting that input according to hidden parameters which will be learned during
# training, and outputs a single value to the next layer.
#
# output tf.keras.layers.Dense — A 128-neuron, followed by 10-node softmax layer. Each node represents a class of
# clothing. As in the previous layer, the final layer takes input from the 128 nodes in the layer before it, and outputs
# a value in the range [0, 1], representing the probability that the image belongs to that class. The sum of all 10 node
# values is 1.
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Compile the model.
#
# - Loss function: An algorithm for measuring how far the model's outputs are from the desired output. The goal of
# training is this measures loss.
# - Optimizer: An algorithm for adjusting the inner parameters of the model in order to minimize loss.
# - Metrics: Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the
# images that are correctly classified.
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# First, we define the iteration behavior for the train dataset:
#
# - Repeat forever by specifying dataset.repeat() (the epochs parameter described below limits how long we perform
# training).
# - The dataset.shuffle(60000) randomizes the order so our model cannot learn anything from the order of the examples.
# - And dataset.batch(32) tells model.fit to use batches of 32 images and labels when updating the model variables.
BATCH_SIZE = 32
train_dataset = train_dataset.cache().repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
test_dataset = test_dataset.cache().batch(BATCH_SIZE)

# Training is performed by calling the model.fit method:
#
# - Feed the training data to the model using train_dataset.
# - The model learns to associate images and labels.
# - The epochs=5 parameter limits training to 5 full iterations of the training dataset, so a total of
# 5 * 60000 = 300000 examples.
model.fit(train_dataset, epochs=5, steps_per_epoch=math.ceil(num_train_examples / BATCH_SIZE))

# ------------------------------------ EVALUATE ACCURACY -------------------------------------------

# Compare how the model performs on the test dataset. Use all examples we have in the test dataset to assess accuracy.
test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(num_test_examples / 32))
print('Accuracy on test dataset:', test_accuracy)

# ------------------------------------ MAKE PREDICTIONS ---------------------------------------------

# Use the model to predict 1 single image
for test_images, test_labels in test_dataset.take(1):
    test_images = test_images.numpy()
    test_labels = test_labels.numpy()
    predictions = model.predict(test_images)

print('Predictions result: ', predictions.shape)
print('First prediction: ', predictions[0])
print('Label number that has the highest confidence value: ', np.argmax(predictions[0]))
print('Label that has the highest confidence value: ', test_labels[0])


# Define these helper functions to plot the data
def plot_image(i, predictions_array, true_labels, images):
    predictions_array, true_label, img = predictions_array[i], true_labels[i], images[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img[..., 0], cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# Plot the 0th image and the prediction
i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions, test_labels)
plt.show()

# Plot the first X test images, their predicted label, and the true label color.
# Correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions, test_labels)
plt.show()

# To make a prediction of a single image, grab an image from the test dataset
img = test_images[0]
print(img.shape)

# Add the image to a batch where it's the only member.
img = np.array([img])
print(img.shape)

predictions_single = model.predict(img)
print(predictions_single)

plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()

np.argmax(predictions_single[0])
