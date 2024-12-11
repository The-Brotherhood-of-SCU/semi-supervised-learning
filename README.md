# README: Neural Network Model for Classification

## Overview

This repository contains a simple neural network model implemented using PyTorch for classification tasks for final project in the curriculum called "Introduction to AI". The model is designed to process input data with a shape of `batch_size x 784` (typically flattened 28x28 images, such as MNIST digits). The architecture includes convolutional layers, pooling layers, and fully connected layers to learn and classify the input data into one of ten classes. Thanks to 57U,yyw,and FrostyJ！

## Model Architecture

The neural network class `Net` inherits from `nn.Module` in PyTorch and consists of the following layers:

1. **Convolutional Layer 1 (`conv1`)**:
   - Input Channels: 1
   - Output Channels: 32
   - Kernel Size: 3x3
   - Padding: 1

2. **Max Pooling Layer 1**:
   - Kernel Size: 2x2
   - Stride: 2

3. **ReLU Activation**: Applied after the first convolutional and pooling layers.

4. **Convolutional Layer 2 (`conv2`)**:
   - Input Channels: 32
   - Output Channels: 16
   - Kernel Size: 3x3
   - Padding: 1

5. **Max Pooling Layer 2**:
   - Kernel Size: 2x2
   - Stride: 2

6. **ReLU Activation**: Applied after the second convolutional and pooling layers.

7. **Fully Connected Layer 1 (`linear`)**:
   - Input Features: 16 * 7 * 7 (flattened from the previous layer)
   - Output Features: 64

8. **ReLU Activation**: Applied after the first fully connected layer.

9. **Fully Connected Layer 2 (`linear2`)**:
   - Input Features: 64
   - Output Features: 10 (number of classes)

## Usage

### Prerequisites

- Python 3.x
- PyTorch

### Installation

To use this model, you need to have PyTorch installed. You can install it using pip:

```bash
pip install torch torchvision
```

### Running the Model

1. Clone this repository:

```bash
git clone https://github.com/The-Brotherhood-of-SCU/semi-supervised-learning.git
cd semi-supervised-learning
```

2. Create a Python script or use a Jupyter notebook to import and use the model. 
### Notes

- The model expects input data to be of shape `batch_size x 784`, which is typically a flattened 28x28 image.
- The `forward` method of the `Net` class reshapes the input tensor to `batch_size x 1 x 28 x 28` before processing it through the convolutional layers.
- The output of the model is the raw scores for each class, which can be passed through a softmax function for probabilities (this is commented out in the `forward` method).

## Contributions

Feel free to contribute to this project by suggesting improvements, fixing bugs, or adding new features. To contribute, fork the repository, make your changes, and submit a pull request.

