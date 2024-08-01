from utils import evaluate_clickme
from get_model import get_model
from thresholding_GradCAM import threshold_grad_cam, threshold_grad_cam_metaformers
from GradCAM import grad_cam, grad_cam_metaformers
from ClickMe_dataset.clickMe import load_clickme_val
import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

##################    Choose a model and an explainer    ####################################
# Valid dcls models: convnext_tiny, convnext_small, convnext_base, caformer_s18, convformer_s18, fastvit_sa24, resnet50
model_name = "caformer_s18"
# Valid explainers: "Threshold_GradCAM", "GradCAM"
explainer = "Threshold_GradCAM"

model, model_cam, transform = get_model(model_name, dcls_equipped=True, pretrained=True)
model.eval()
if explainer == "GradCAM":
    if model_name == "convformer_s18" or model_name == "caformer_s18":
        explain = grad_cam_metaformers
    else:
        explain = grad_cam
else:
    if model_name == "convformer_s18" or model_name == "caformer_s18":
        explain = threshold_grad_cam_metaformers
    else:
        explain = threshold_grad_cam

batch_size = 4
batches = int(52224 / batch_size)
# now let's load the dataset
clickme_dataset = load_clickme_val(batch_size)

#Get interpretability score
score = evaluate_clickme(model_cam, explainer=explain, clickme_val_dataset=clickme_dataset.take(10),
                         preprocess_inputs=transform)

print(score)
