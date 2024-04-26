
import torch
import torch.nn as nn


class FastvitGradCAM(nn.Module):
    def __init__(self, model):
        super(FastvitGradCAM, self).__init__()

        self.model = model
        # Placeholder for the gradients
        self.gradients = None
        # Placeholder for the activations
        self.activations = None

    # Hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input embedding
        x = self.model.forward_embeddings(x)
        # through backbone
        x = self.model.forward_tokens(x)
        if self.model.fork_feat:
            # output features of four stages for dense prediction
            return x
        # for image classification
        x = self.model.conv_exp(x)
        self.activations = x
        x = self.model.gap(x)
        x = x.view(x.size(0), -1)
        cls_out = self.model.head(x)
        return cls_out

    # Method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients

    # Method for the activation extraction
    def get_activations(self):
        return self.activations
