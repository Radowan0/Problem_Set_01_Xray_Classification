## This project builds a CNN model to classify Chest X-ray images into two main categories: Normal & Pneumonia

This project uses TensonFlow/keras, data augmentation, CNN architecture, and training visualition 

## Dataset Preparation:
   My dataset structured as:
  
dataset/
    train/
        NORMAL/
        PNEUMONIA/
    val/
        NORMAL/
        PNEUMONIA/
    test/
        NORMAL/
        PNEUMONIA/

Each folder contains real X-ray images

## To improve training, we used ImageDataGenerator with Data Augmentation
  Used to artificially increase dataset size and avoid overfitting:
   -rotation_range=20
   -width_shift_range=0.2
   -height_shift_range=0.2
   -shear_range=0.2
   -zoom_range=0.2
   -horizontal_flip=True
   -rescale=1/255

This makes the model more robust.

## CNN Architecture
  Input Image (224x224x3)

1️.  CONV BLOCK 1
    Conv2D(32 filters, 3x3)
    BatchNormalization
    MaxPooling2D

2️.  CONV BLOCK 2
    Conv2D(64 filters, 3x3)
    BatchNormalization
    MaxPooling2D

3️.  CONV BLOCK 3
    Conv2D(128 filters, 3x3)
    BatchNormalization
    MaxPooling2D

Flatten
Dense(128, relu)
Dropout(0.5)
Dense(1, sigmoid)


## Explanation of the Layers has been used in this CNN architecture
  @ Conv2D
    Extracts important features:
    edges, curves, textures, patterns inside lungs (white patches in pneumonia)

  @ Batch Normalization
    Used after each convolution to:
    speed up training, stabilize gradients,reduce internal covariate shift
    This improves accuracy and makes training smoother.

  @ MaxPooling2D
    Reduces image size & computation.
    Helps model focus on most important features.

  @ Flatten
    Converts 3D features → 1D vector for dense layers.

  @ Dense(128)
    Learns complex relations between extracted features.

  @ Dropout(0.5)
    Prevents overfitting by randomly turning off neurons.

  @ Dense(1, sigmoid)
    Since this is a binary classification, sigmoid outputs:
    0 → Normal
    1 → Pneumonia


## Model Compilation:
  loss = 'binary_crossentropy'    → correct for binary output
  optimizer = 'adam'              → fast and adaptive optimizer
  metrics = ['accuracy']          → easy to evaluate performance


  
## Model Training:
 I trained for 20 epochs.
 During training, the model learns by:
 Reading training images, making predictions, comparing prediction vs real label,adjusting weights using backpropagation.
 I also used validation data to check if the model is generalizing.

## Training Output (My Exact Result)
   My final output showed:
* Training accuracy: ~90–93%
* Validation accuracy: ~60–75%
* Training loss: Low (below 0.5)
* Validation loss: Fluctuating

## Why Validation Accuracy Jumps Up & Down?

Because:
pneumonia images vary a lot, dataset is not perfectly clean, validation set may be small.
Still the model learned strong features.

## Final Test Accuracy
Test Accuracy: 0.8910  (89%)
Test Loss: 0.2661

## Conclusion:
I built a 3-block Convolutional Neural Network with Batch Normalization and MaxPooling layers.
The model extracts spatial features from chest X-ray images and classifies them using a sigmoid output layer.
I used data augmentation to reduce overfitting and improve generalization.
The model achieved around 90% training accuracy and around 70% validation accuracy.
