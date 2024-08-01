# Facial Emotion Detection Using CNN

## Description
This project implements a facial emotion detection system using Convolutional Neural Networks (CNN). The model is trained to recognize various emotions from facial expressions, providing a robust solution for applications in fields such as psychology, security, and human-computer interaction.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/facial-emotion-detection-using-CNN.git
   cd facial-emotion-detection-using-CNN
2. Install the required packages:
pip install -r requirements.txt
# Usage
3. Load the Dataset: Ensure the FER2013 dataset is properly structured in the training and testing directories.

# Train the Model:
4. Run the training script to build and train the CNN model:
# Code snippet to train the model
    emotion_model_info = emotion_model.fit_generator(
    train_generator,
    steps_per_epoch=28709 // 64,
    epochs=15,
    validation_data=validation_generator,
    validation_steps=7178 // 64)

## CNN Architecture

The Convolutional Neural Network (CNN) architecture for facial emotion recognition consists of the following layers:

1. **Input Layer**: 
   - Shape: (48, 48, 1) - Grayscale images.

2. **Convolutional Layers**:
   - **Conv Layer 1**: 32 filters, (3x3) kernel, ReLU activation.
   - **Conv Layer 2**: 64 filters, (3x3) kernel, ReLU activation.
   - **Conv Layer 3**: 128 filters, (3x3) kernel, ReLU activation.
   - **Conv Layer 4**: 128 filters, (3x3) kernel, ReLU activation.

3. **Pooling Layers**:
   - **MaxPooling Layer 1**: (2x2) pool size.
   - **MaxPooling Layer 2**: (2x2) pool size.
   - **MaxPooling Layer 3**: (2x2) pool size.

4. **Dropout Layers**: 
   - 25% dropout after certain convolutional layers to prevent overfitting.

5. **Flatten Layer**: Converts 2D feature maps to 1D.

6. **Dense Layers**:
   - **Dense Layer 1**: 1024 units, ReLU activation.
   - **Dense Layer 2**: 512 units, ReLU activation.

7. **Output Layer**:
   - 7 units (one for each emotion), Softmax activation for classification.

This architecture effectively captures and classifies facial expressions into distinct emotion categories.


# Visualizations
The model's accuracy and loss during training are plotted for evaluation.
Feature maps from the first convolutional layer are visualized to understand what the model is learning.
# Model Saving
The trained model is saved as facial_emotions_detection_model.h5 for future use.

# Contributing
Contributions are welcome!

