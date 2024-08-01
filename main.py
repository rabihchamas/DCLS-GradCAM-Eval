from utils import evaluate_clickme
from get_model import get_model
from thresholding_GradCAM import threshold_grad_cam, threshold_grad_cam_metaformers
from GradCAM import grad_cam, grad_cam_metaformers
from ClickMe_dataset.clickMe import load_clickme_val
import pathlib
from args import Args

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
args = Args()

##################    Choose a model and an explainer    ####################################
# Valid dcls models: convnext_tiny, convnext_small, convnext_base, caformer_s18, convformer_s18, fastvit_sa24, resnet50
args.model_name = "convnext_tiny"
# Valid explainers: "Threshold_GradCAM", "GradCAM"
args.explainer = "Threshold_GradCAM"
# If you want to test on all the dataset choose give 100
args.data_used_percentage = 0.25
args.batch_size = 8

def main(arguments):
    model, model_cam, transform = get_model(arguments.model_name, dcls_equipped=True, pretrained=True)
    model.eval()
    if arguments.explainer == "GradCAM":
        if arguments.model_name == "convformer_s18" or arguments.model_name == "caformer_s18":
            explainer = grad_cam_metaformers
        else:
            explainer = grad_cam
    else:
        if arguments.model_name == "convformer_s18" or arguments.model_name == "caformer_s18":
            explainer = threshold_grad_cam_metaformers
        else:
            explainer = threshold_grad_cam

    args.batch_size = 32
    clickme_dataset = load_clickme_val(args.batch_size)

    batches = 52224 / args.batch_size
    batches = int((args.data_used_percentage / 100) * batches)
    print(batches)
    #Get interpretability score
    score = evaluate_clickme(model_cam, explainer=explainer, clickme_val_dataset=clickme_dataset.take(batches),
                             preprocess_inputs=transform)
    print(score)


if __name__ == '__main__':
    main(args)
