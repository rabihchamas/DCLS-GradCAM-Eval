import torch
import numpy as np
from scipy.stats import spearmanr
from PIL import Image

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
