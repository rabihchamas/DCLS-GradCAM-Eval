from utils import evaluate_clickme, device
from get_model import get_model
from thresholding_GradCAM import threshold_grad_cam
from ClickMe_dataset.clickMe import load_clickme_val
import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
# Valid dcls models: convnext_tiny, convnext_small, convnext_base, caformer_s18, convformer_s18, fastvit_sa24, resnet50
model_name = "caformer_s18"
model, model_cam, transform = get_model(model_name, dcls_equipped=True, pretrained=True)
model.eval()
explainer = threshold_grad_cam

batch_size = 4
batches = int(52224 / batch_size)
# now let's load the dataset
clickme_dataset = load_clickme_val(batch_size)

#Get interpretability score
score = evaluate_clickme(model_cam, explainer=explainer, clickme_val_dataset=clickme_dataset.take(10),
                         preprocess_inputs=transform)

print(score)
