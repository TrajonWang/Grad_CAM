import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, datasets, transforms
import matplotlib.pyplot as plt
import pickle
import urllib.request

import numpy as np
from PIL import Image

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.features = None

        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        def forward_hook(module, input, output):
            self.features = output

        # Register hooks
        target_layer = self.find_target_layer()
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(hook_function)

    def find_target_layer(self):
        # Dynamically find the target layer
        module = self.model
        for name, layer in module.named_modules():
            if name == self.target_layer:
                return layer
        raise ValueError("Layer not found in the model")

    def forward(self, input):
        return self.model(input)

    def generate_heatmap(self, gradient, feature):
        weights = torch.mean(gradient, dim=(2, 3), keepdim=True)
        heatmap = torch.mul(feature, weights).sum(dim=1, keepdim=True)
        heatmap = F.relu(heatmap)
        heatmap = F.interpolate(heatmap, size=(224, 224), mode='bilinear', align_corners=False)
        heatmap = heatmap.squeeze().cpu().detach().numpy()
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
        return heatmap

    def __call__(self, input_tensor, class_idx):
        # Forward pass
        output = self.forward(input_tensor)
        self.model.zero_grad()

        # One-hot encoding for the target class
        one_hot_output = torch.FloatTensor(1, output.size()[-1]).zero_()
        one_hot_output[0][class_idx] = 1

        # Backward pass
        output.backward(gradient=one_hot_output, retain_graph=True)
        gradients = self.gradients.data
        features = self.features.data

        heatmap = self.generate_heatmap(gradients, features)
        return heatmap

