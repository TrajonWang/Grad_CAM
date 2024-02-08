# Generalized Grad_CAM
We have encapsulated the definition of Grad_CAM class in the DLP_GradCam.py file, and apply it on an example in the notebook.

The idea of Grad_CAM is to understand the decision-making process of a convolutional neural network(CNN). 

The method begins with a foward pass through the CNN model using the input image. During this pass, the activations of the target layer (which represents the learned features extracted by the CNN) are captured. 

Then, a backward pass is performed to compute the gradients of the target layer activations with respect to the predicted class score. The gradients indicate how each feature map in the target layer influences the final prediction of the network for the given class.

The gradients obtained from the backward pass are then globally averaged across spatial dimensions to obtain importance weights for each feature map, which represent the importance of each feature map in influencing the class prediction.

Stepping forward, we try to generate the heatmaps. We apply the importance weights obtained to weigh the feature maps in the target layer. This weighted combination highlights the regions in the feature maps that are most relevant for predicting the target class. The weighted feature maps are then summed along the channel dimension to obtain a single heatmap.

Finally, the heatmap is processed to make it visually interpretable.
Common post-processing steps include 1)ReLU activation to retain only positive values 2)interpolation to match the size of the original input image, and 3)normalization to scale the values between 0 and 1.
The resulting heatmap is overlaid onto the input image to visualize the regions that are important for the network's prediction of the target class.
