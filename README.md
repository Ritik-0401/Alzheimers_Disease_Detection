## Overview
This project develops a deep learning model to detect Alzheimer's disease from brain MRI images using the Alzheimer's Dataset (4 classes). The model leverages convolutional neural networks (CNNs) implemented with TensorFlow and Keras, classifying images into four categories: Non-Demented, VeryMildDemented, MildDemented, and ModerateDemented. It includes data preprocessing, image augmentation for dataset robustness, model training with class imbalance handling, and evaluation using balanced accuracy score (BAS) and Matthews correlation coefficient (MCC).

## Dataset
The Alzheimer's Dataset (4 classes) comprises brain MRI images across four classes:
 - NonDemented: No Alzheimer's disease signs.
 - VeryMildDemented: Very mild Alzheimer's disease.
 - MildDemented: Mild Alzheimer's disease.
 - ModerateDemented: Moderate Alzheimer's disease.
The dataset contains 6,400 images, validated by the data generator output.

## Methodology
### Data Preprocessing
 - Image Loading: Utilizes ImageDataGenerator from Keras to load images.
 - Image Normalization: Rescales pixel values to [0, 1] with rescale=1./255 for improved model convergence.
 - Image Resizing: Standardizes images to 176x176 pixels (IMG_SIZE = 176, IMAGE_SIZE = [176, 176]) for consistent input dimensions.
### Image Augmentation
Applies ImageDataGenerator with:
 - Zoom Range: [0.99, 1.01] for slight scaling variations.
 - Brightness Range: [0.8, 1.2] for lighting adjustments.
 - Horizontal Flip: True for orientation diversity.
 - Fill Mode: "constant" for new pixel filling.
 - Data Format: "channels_last". Generates a 6,500-image batch (batch_size=6500) without shuffling.
### Handling Class Imbalance
SMOTE: Uses imblearn.over_sampling.SMOTE to generate synthetic samples for minority classes (e.g., ModerateDemented) ensuring balanced training data.
### Training
Data Feeding: Augmented data fed via ImageDataGenerator.
Early Stopping: Employs EarlyStopping to monitor validation loss and prevent overfitting.
Optimizer and Loss: Likely uses Adam optimizer and categorical cross-entropy loss.
### Evaluation
Metrics:
 - Balanced Accuracy Score (BAS): 94.88%, measures recall across imbalanced classes.
 - Matthews Correlation Coefficient (MCC): 93.14%, evaluates classification quality.
 - Classification Report: Provides precision, recall, and F1-score.
   ![image](https://github.com/user-attachments/assets/63a4f794-070a-4719-8e39-d11338dbd7d8)
 - Confusion Matrix: Displays true vs. predicted labels.
   ![image](https://github.com/user-attachments/assets/3c20d3d4-d766-4536-9254-27456f41dea2)
