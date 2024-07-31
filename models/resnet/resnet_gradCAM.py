import torch.nn as nn

class resnetGradCAM(nn.Module):

    def __init__(self, model):
        super(resnetGradCAM, self).__init__()
        self.model = model
        # Placeholder for the gradients
        self.weights = None

        # Placeholder for the activations
        self.activations = None

    # Hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.model.forward_features(x)
        # Register activat
        self.activations = x
        # Register the hook
        h = x.register_hook(self.activations_hook)
        x = self.model.forward_head(x)
        return x

    # Method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients

    # Method for the activation extraction
    def get_activations(self, x):
        if self.activations is None:
          return self.model.forward_features(x)
        return self.activations