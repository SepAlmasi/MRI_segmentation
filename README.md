# MRI_segmentation

This project provides a training pipeline for segmenting grayscale images into three regions. The regions are specified by three levels of 0, 1, and 2 in the segmentation masks. 

The model architecture is a FCN with ResNet50 backbone. Hyperparameters are set as 0.01 for learning rate and 0.9 for momntum. Training is optimized by Stochastic Gradient Descent and a cross entropy loss function that runs on the softmax activations from the last layer. 

