# Handwritten Character Recognition
# Overview
This repository contains a deep learning model for handwritten character recognition. The model is built using TensorFlow and Keras, and it processes images of handwritten characters to classify them into predefined labels. The dataset used for training includes a variety of handwritten images.

# Dataset
Images: The images used for training are located in the Img directory.
Labels: The corresponding labels for the images are stored in a CSV file named english.csv. Each label represents a character from the handwritten dataset.
Requirements
Python 3.x
TensorFlow
OpenCV
NumPy
Pandas
Matplotlib
Seaborn
scikit-learn
Pillow

# Data Preprocessing
Image Loading: The model loads images from the specified directory. Each image is read using OpenCV.
Image Resizing: Images are resized to 64x64 pixels for uniformity.
Normalization: The pixel values of the images are normalized by dividing by 255.0 to scale them between 0 and 1.
Label Encoding: The labels from the CSV file are transformed into numerical format using LabelEncoder.

# Model Architecture
The model is a convolutional neural network (CNN) structured as follows:

Input Layer: Accepts images of shape (64, 64, 3).

Convolutional Layers:
1. First layer: 512 filters with a kernel size of (5, 5), ReLU activation.
2. Second layer: 256 filters with a kernel size of (3, 3), ReLU activation.
3. Third layer: 256 filters with a kernel size of (3, 3), ReLU activation.
   
Max Pooling Layers: After each convolutional layer to reduce the spatial dimensions.

Flattening Layer: Converts the 3D output to 1D.

Dense Layers:
Fully connected layers with 512, 1024, 512, and 256 neurons, using ReLU and sigmoid activations.
Output layer: 62 neurons with softmax activation, corresponding to the number of character classes.

The model is compiled using the following parameters:
1.Loss Function: Sparse categorical cross-entropy
2. Optimizer: Adam
3. Metrics: Accuracy
# Training
The model is trained for 20 epochs, with a validation split of 20%. During training, the loss and accuracy are monitored.

# Model Evaluation
The model's performance is evaluated using accuracy, precision, recall, F1 score, and confusion matrix metrics.

# Visualization
The results are visualized using:

A confusion matrix heatmap.
Random sample predictions showing the predicted and true labels for selected images.
# Image Prediction
To predict the label of a new handwritten image:

Load and preprocess the image.
Use the trained model to predict the label.
Display the predicted label along with the original image.
# Saving and Loading the Model
The trained model can be saved to a file for later use. It can also be loaded back into memory for further predictions.

# File Text Prediction
The model can predict labels for images selected through a file dialog. It can match predicted labels with corresponding text from a CSV file, allowing users to see the text representation of their handwriting.

# Usage
Run the main script to load the model and perform predictions on selected images. The script will prompt for a CSV file containing text labels and an image file for which to predict the label.

# Conclusion
This project demonstrates the application of convolutional neural networks in image recognition tasks, specifically for handwritten character recognition. It serves as a foundation for further enhancements and adaptations for different datasets or more complex recognition tasks.
