# import everything we need first
from keras import utils
import numpy as np
from sklearn.model_selection import train_test_split
import pylab as plt


from astroNN.models import Galaxy10CNN
from astroNN.datasets import load_galaxy10sdss
from astroNN.datasets.galaxy10sdss import galaxy10cls_lookup, galaxy10_confusion

# To load images and labels (will download automatically at the first time)
# First time downloading location will be ~/.astroNN/datasets/
images, labels = load_galaxy10sdss()

# To convert the labels to categorical 10 classes
labels = utils.to_categorical(labels, 10)

# Select 10 of the images to inspect
img = None
plt.ion()
print('===================Data Inspection===================')
for counter, i in enumerate(range(np.random.randint(0, labels.shape[0], size=10).shape[0])):
    img = plt.imshow(images[i])
    plt.title('Class {}: {} \n Random Demo images {} of 10'.format(np.argmax(labels[i]), galaxy10cls_lookup(labels[i]), counter+1))
    plt.draw()
    plt.pause(2.)
plt.close('all')
print('===============Data Inspection Finished===============')

# To convert to desirable type
labels = labels.astype(np.float32)
images = images.astype(np.float32)

# Split the dataset into training set and testing set
train_idx, test_idx = train_test_split(np.arange(labels.shape[0]), test_size=0.1)
train_images, train_labels, test_images, test_labels = images[train_idx], labels[train_idx], images[test_idx], labels[test_idx]

# To create a neural network instance
galaxy10net = Galaxy10CNN()
print(galaxy10net) # hux
# set maximium epochs the neural network can run, set 5 to get quick result
galaxy10net.max_epochs = 50

# To train the nerual net
# astroNN will normalize the data by default
print(r'To train the nerual net')
print('\n')
galaxy10net.train(train_images, train_labels)

# print model summary before training
print('print model summary before training')
print('\n')
galaxy10net.keras_model.summary()

# After the training, you can test the neural net performance
# Please notice predicted_labels are labels predicted from neural network. test_labels are ground truth from the dataset
print('After the training, you can test the neural net performance')
print('\n')
predicted_labels = galaxy10net.test(test_images)
# print('predicted_labels\n', predicted_labels)
# Convert predicted_labels to class
# print('Convert predicted_labels to class')
# print('\n')
prediction_class = np.argmax(predicted_labels, axis=1)
print('prediction_class:\n',prediction_class, len(prediction_class))
# Convert test_labels to class
print('Convert test_labels to class')
test_class = np.argmax(test_labels, axis=1)
print('test_class:',test_class, len(test_class))
# Prepare a confusion matrix
confusion_matrix = np.zeros((10,10))
# create the confusion matrix
for counter, i in enumerate(prediction_class):
    confusion_matrix[i, test_class[counter]] += 1

# print('confusion_matrix:\n')
# print(confusion_matrix)
# Plot the confusion matrix
galaxy10_confusion(confusion_matrix)