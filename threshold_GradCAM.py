import torch
import torchvision.transforms as transforms
import numpy as np

resize_transform = transforms.Resize(size=(224, 224))


def normalize_batch(batch):
    min_vals = batch.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
    max_vals = batch.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
    normalized_batch = (batch - min_vals) / (max_vals - min_vals)
    return normalized_batch


def threshold_grad_cam_metaformers(xbatch, ybatch, CAM_model):
    pred = CAM_model(xbatch)
    output = torch.sum(pred * ybatch)
    output.backward()
    gradients = CAM_model.get_activations_gradient()

    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[1, 2])

    # get the activations of the last convolutional layer
    activations = CAM_model.get_activations(xbatch).detach()
    # weight the channels by corresponding gradients
    activations *= pooled_gradients.unsqueeze(1).unsqueeze(1)
    activations = torch.relu(activations)
    # average the channels of the activations
    heatmap = torch.mean(activations, dim=-1)
    heatmap = normalize_batch(heatmap)
    heatmap = np.maximum(heatmap.cpu(), 0.3)

    heatmap = heatmap.unsqueeze(1)
    heatmap = torch.stack([resize_transform(img) for img in heatmap])
    heatmap = heatmap.squeeze(1)

    return heatmap


def threshold_grad_cam(xbatch, ybatch, cam_model):
    pred = cam_model(xbatch)
    output = torch.sum(pred * ybatch)
    output.backward()
    gradients = cam_model.get_activations_gradient()
    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[2, 3])
    # get the activations of the last convolutional layer
    activations = cam_model.get_activations(xbatch).detach()
    # weight the channels by corresponding gradients
    activations *= pooled_gradients.unsqueeze(-1).unsqueeze(-1)
    # apply relu before summation
    activations = torch.relu(activations)
    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1)
    # normalize the heatmap
    heatmap = normalize_batch(heatmap)
    heatmap = np.maximum(heatmap.cpu(), 0.3)
    heatmap = heatmap.unsqueeze(1)
    heatmap = torch.stack([resize_transform(img) for img in heatmap])
    heatmap = heatmap.squeeze(1)
    return heatmap
