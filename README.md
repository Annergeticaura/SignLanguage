# SignLanguage
Sign Language detection using custom CNN architecture

# Sign Language Detection
## Overview
This project aims to develop a machine learning model that predicts and recognizes some known sign language gestures from images or real-time video feeds. The system includes a GUI that allows users to either upload an image for detection or use a real-time video stream to recognize sign language gestures. The model operates during a specific time period, such as from 6 PM to 10 PM, as a simulated business logic constraint.

# Features
Image and Video Input: The system supports both image uploads and real-time video feed for gesture recognition.
Time Restriction: The model operates only within a set time period (e.g., 6 PM to 10 PM).
Recognition of Known Words: The system can recognize a set of known sign language gestures (e.g., 25 signs).
GUI Interface: Provides an intuitive graphical user interface (GUI) for interaction.
# Technologies Used
Python (v3.x)
TensorFlow / Keras: For building the Convolutional Neural Network (CNN) for gesture recognition.
OpenCV: For handling real-time video and image input.
Tkinter: For creating the GUI interface.
NumPy: For numerical operations.
Matplotlib: For visualizing training metrics.
# Model Architecture
The sign language detection model is a Convolutional Neural Network (CNN) designed to classify gestures into one of the known sign classes. The architecture includes:

Conv2D layers: For feature extraction with filters of increasing depth (32, 64).
MaxPooling: For reducing dimensionality while retaining significant features.
Flatten: To convert the 2D feature maps into 1D vectors.
Dense layers: For classification, with 64 units and a final dense layer using softmax for multiclass classification.
Output layer: Uses softmax activation for predicting one of the 25 sign classes.
# Dataset
The model was trained on a dataset containing images of hand gestures representing different signs. The dataset is divided into train, validation, and test sets to ensure the model's ability to generalize to unseen data.

Preprocessing:
Images were resized to 28x28 pixels and converted to grayscale.
Data augmentation techniques like rotation, zoom, and flipping were applied to improve model generalization.
# License
The project is under MIT License
