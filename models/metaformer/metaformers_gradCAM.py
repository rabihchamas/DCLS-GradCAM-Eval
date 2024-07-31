import torch.nn as nn


class MetaformerGradCAM(nn.Module):

    def __init__(self, model):
        super(MetaformerGradCAM, self).__init__()
        self.model = model
        # Placeholder for the gradients
        self.gradients = None
        # Placeholder for the activations
        self.activations = None

    # Hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward_features(self, x):
        for i in range(self.model.num_stage):
            x = self.model.downsample_layers[i](x)
            x = self.model.stages[i](x)
        self.activations = x
        h = x.register_hook(self.activations_hook)
        return self.model.norm(x.mean([1, 2]))  # (B, H, W, C) -> (B, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.model.head(x)
        return x

    # Method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients

    # Method for the activation extraction
    def get_activations(self, x):
        return self.activations
