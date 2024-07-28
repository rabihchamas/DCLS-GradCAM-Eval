import torch.nn as nn


class ConvNextGradCAM(nn.Module):

    def __init__(self, model):
        super(ConvNextGradCAM, self).__init__()
        self.model = model
        # Placeholder for the gradients
        self.gradients = None

        # Placeholder for the activations
        self.activations = None

        # Combine downsample_layers and stages into a single sequential model
        layers = []
        for i in range(4):
            layers.append(self.model.downsample_layers[i])
            layers.append(self.model.stages[i])
        self.features_conv = nn.Sequential(*layers)

        self.norm = self.model.norm
        self.head = self.model.head

    # Hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward_features(self, x):
        x = self.features_conv(x)
        # Register activation
        self.activations = x
        # Register the hook
        h = x.register_hook(self.activations_hook)
        return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    # Method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients

    # Method for the activation extraction
    def get_activations(self, x):
        if self.activations is None:
            return self.features_conv(x)
        return self.activations
