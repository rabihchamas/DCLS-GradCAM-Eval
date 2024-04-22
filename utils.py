import torch
import numpy as np
from scipy.stats import spearmanr
from PIL import Image
import cv2


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HUMAN_SPEARMAN_CEILING = 0.65753


def spearman_correlation(heatmaps_a, heatmaps_b):
    scores = []
    for ha, hb in zip(heatmaps_a, heatmaps_b):
        # Check for constant values in either heatmap
        if isinstance(ha, torch.Tensor):
            ha = ha.cpu().numpy()
        if isinstance(hb, torch.Tensor):
            hb = hb.cpu().numpy()
        if np.std(ha) == 0 or np.std(hb) == 0:
            # If either heatmap has constant values, handle this case
            # Here we choose to skip these pairs, but you could also assign a default score
            continue
        else:
            rho, _ = spearmanr(ha.flatten(), hb.flatten())
            scores.append(rho)
    return np.array(scores)


def evaluate_clickme(model, explainer=None, clickme_val_dataset=None, preprocess_inputs=None):
    model.eval()
    if preprocess_inputs is None:
        preprocess_inputs = lambda x: x

    spearman_scores = []

    for images_batch, heatmaps_batch, label_batch in clickme_val_dataset:
        images_batch = torch.stack([preprocess_inputs(Image.fromarray(x)) for x in
                                    images_batch.numpy().astype(np.uint8)]).to(device)
        label_batch = torch.Tensor(label_batch.numpy()).to(device)
        heatmaps_batch = heatmaps_batch.numpy()  # Convert TensorFlow tensor to NumPy

        saliency_maps = explainer(images_batch, label_batch, model)

        if saliency_maps.ndim == 4:
            saliency_maps = saliency_maps.mean(-1)  # Assuming channel is second dimension
        if heatmaps_batch.ndim == 4:
            heatmaps_batch = heatmaps_batch.mean(-1)
        spearman_batch = spearman_correlation(saliency_maps.cpu(), heatmaps_batch)
        spearman_scores.extend(spearman_batch)
    alignment_score = np.mean(spearman_scores) / HUMAN_SPEARMAN_CEILING
    return alignment_score


def get_superimposed(method, cam_model, test_img_path, preprocess):
    # get and preprocess image example
    test_img = Image.open(test_img_path)
    numpy_img = np.array(test_img)
    processed_image = preprocess(test_img).unsqueeze(0).to(device)

    # Get the predicted class index
    logits = cam_model(processed_image)
    _, predicted_class = torch.max(logits, 1)
    # Create a one-hot encoded vector
    one_hot = torch.zeros_like(logits)
    one_hot[:, predicted_class] = 1

    img = cv2.resize(numpy_img, (224, 224))
    heatmap = method(processed_image, one_hot, cam_model)[0].cpu().numpy()
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.4, heatmap, 0.6, 0)
    return superimposed_img
