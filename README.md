# MRI_segmentation

This project provides a training pipeline for segmenting grayscale images into three regions. The regions are specified by three levels of 0, 1, and 2 in the segmentation masks. 

The model architecture is a FCN with ResNet50 backbone. Hyperparameters are set as below:
learning rate = 0.01
momntum = 0.9
batch_size = 1.

Model is optimized using Stochastic Gradient Descent and a multi-class cross entropy loss function that runs on the softmax activations of the last layer.
