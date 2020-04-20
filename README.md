# MRI_segmentation

This project provides a training pipeline for segmenting grayscale images into three regions. The regions are specified by three levels of 0, 1, and 2 in the segmentation masks. 

Dataset class is a map one meaning that it maps (image,mask) pairs to indices through which they are sampeled. To build the dataset, it is assumed that the data is always of grayscale nature. Therefore, in data loading step, every image is replicated three times to accomodate for the model architecture requirement of having three input channels.

The model architecture is a FCN with ResNet50 backbone. Hyperparameters are set as below:
learning rate = 0.01
momentum = 0.9
batch_size = 1.

Model is optimized using Stochastic Gradient Descent and a multi-class cross entropy loss function that runs on the softmax activations of the last layer.

Training will be initiated by running "python spine_segmentation.py" from command line.
