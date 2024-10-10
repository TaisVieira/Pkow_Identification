import matplotlib.pyplot as plt
import os
import PIL
import tensorflow as tf
import glob
import pathlib
import numpy as np
import random
import xgboost

from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

data_dir = pathlib.Path('C:/Users/taisb/Desktop/655_project/cnn/')
batch_size = 32
img_height = 128
img_width = 128

# Load the training dataset from a directory, split into training (80%) and validation (20%).
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",  # Specify that we are loading the training portion
  seed=123,  
  image_size=(img_height, img_width),  # Resize images for consistency
  batch_size=batch_size  
)

# Print the names of the classes (based on folder names in the data directory).
class_names = train_ds.class_names

# Load the validation dataset (20% split)
val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",  # Specify that we are loading the validation portion
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

# Visualize the first batch of training images (9 images in a 3x3 grid)
plt.figure(figsize=(10, 10))  
for images, labels in train_ds.take(1): 
    for i in range(9):  
        ax = plt.subplot(3, 3, i + 1)  
        plt.imshow(images[i].numpy().astype("uint8"))  
        plt.title(class_names[labels[i]])  
        plt.axis("off")  

# Define a Convolutional Neural Network (CNN) with 3 Conv2D layers followed by MaxPooling
model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),  # Normalize the pixel values to [0, 1]
  layers.Conv2D(16, 3, padding='same', activation='relu'),  # 1st convolutional layer
  layers.MaxPooling2D(),  # Max pooling layer to reduce the spatial dimensions
  layers.Conv2D(32, 3, padding='same', activation='relu'),  # 2nd convolutional layer with more filters
  layers.MaxPooling2D(),  # Another max pooling layer
  layers.Conv2D(64, 3, padding='same', activation='relu'),  # 3rd convolutional layer with more filters
  layers.MaxPooling2D(),  # Final max pooling layer
  layers.Flatten(),  # Flatten the output to pass into dense layers
  layers.Dense(128, activation='relu'),  # Fully connected layer with 128 units
  layers.Dense(num_classes)  # Output layer with one unit per class
])

# Compile the model with Adam optimizer and Sparse Categorical Crossentropy loss
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  
              metrics=['accuracy'])  

# Train the model for 100 epochs using the training and validation datasets
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=100,  
  verbose=0  
)

# Plot training and validation accuracy over the epochs
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)  
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# Plot training and validation loss over the epochs
plt.subplot(1, 2, 2)  
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

# Load a single test image and prepare it for prediction
img = tf.keras.utils.load_img(
    test_data_dir, target_size=(img_height, img_width)  
)
img_array = tf.keras.utils.img_to_array(img)  
img_array = tf.expand_dims(img_array, 0)  

# Make a prediction on the single test image
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])  # Apply softmax to get class probabilities

# Print the predicted class and confidence level
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

# Function to test multiple images from the test directory
def test_images(names):
    for img_name in names:
        img_path = 'C:/Users/taisb/Desktop/655_project/data/test/' + img_name  # Path to each test image
        img = tf.keras.utils.load_img(img_path, target_size=(img_height, img_width))
        img_array = tf.keras.utils.img_to_array(img)  # Convert the image to array format
        img_array = tf.expand_dims(img_array, 0)  # Add batch dimension for prediction

        predictions = model.predict(img_array)  # Make prediction
        score = tf.nn.softmax(predictions[0])  # Get class probabilities

        # Print actual vs predicted class and confidence score
        print(
            "This image is a {} and was classified as {} with a {:.2f} percent confidence."
            .format(img_name[:-5], class_names[np.argmax(score)], 100 * np.max(score))
        )

test_images(names)  # Test the entire batch of test images
