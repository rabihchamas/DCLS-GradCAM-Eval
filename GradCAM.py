import torch
import torchvision.transforms as transforms


resize_transform = transforms.Resize(size=(224, 224))


def grad_cam(xbatch, ybatch, cam_model):
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
    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1)
    # relu after summation
    heatmap = torch.relu(heatmap)
    heatmap = heatmap.unsqueeze(1)
    heatmap = torch.stack([resize_transform(img) for img in heatmap])
    heatmap = heatmap.squeeze(1)
    return heatmap
