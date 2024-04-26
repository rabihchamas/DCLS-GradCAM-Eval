import torch
from models.fastvit.fastvit import fastvit_sa24, fastvit_sa36, fastvit_sa36_dcls, fastvit_sa24_dcls
from models.fastvit.fastvit_gradCAM import FastvitGradCAM
from utils import device
from transforms import fastvit_transform
from pathlib import Path
def get_model(model_name, dcls_equipped=True, pretrained=True):
    if model_name == "fastvit_sa36":
        if dcls_equipped:
            model = fastvit_sa36_dcls().to(device)
            if pretrained:
                checkpoint = torch.load('link to dcls checkpoints')
                model_state_dict = checkpoint['state_dict_ema']
                model.load_state_dict(model_state_dict, strict=True)
        else:
            model = fastvit_sa36().to(device)
            if pretrained:
                checkpoint = torch.load('link to baseline checkpoints')
                model_state_dict = checkpoint['state_dict_ema']
                model.load_state_dict(model_state_dict, strict=True)
        model_cam =  FastvitGradCAM(model)

    if model_name == "fastvit_sa24":
        if dcls_equipped:
            model = fastvit_sa24_dcls().to(device)#r"C:\Users\Layal\Downloads\fastvit_sa24_v1_seed0.pth(1)(1).tar"
            if pretrained:
                file_path = Path("C:/Users/Layal/Downloads/fastvit_sa24_v1_seed0.pth (1) (1).tar")
                checkpoint = torch.load(file_path, map_location=device)
                model_state_dict = checkpoint['state_dict_ema']
                model.load_state_dict(model_state_dict, strict=True)
        else:
            model = fastvit_sa24().to(device)
            if pretrained:
                checkpoint = torch.load('link to baseline checkpoints')
                model_state_dict = checkpoint['state_dict_ema']
                model.load_state_dict(model_state_dict, strict=True)
        model_cam = FastvitGradCAM(model)
        transform = fastvit_transform
    # if model_name == "fastvit_sa24":
    #     if DCLS:
    #         model = fastvit_sa24_dcls().to(device)
    #         if pretrained:
    #             checkpoint = torch.load('link to dcls checkpoints')
    #             model_state_dict = checkpoint['state_dict_ema']
    #             model.load_state_dict(model_state_dict, strict=True)
    #     else:
    #         model = fastvit_sa24().to(device)
    #         if pretrained:
    #             checkpoint = torch.load('link to baseline checkpoints')
    #             model_state_dict = checkpoint['state_dict_ema']
    #             model.load_state_dict(model_state_dict, strict=True)

    return model, model_cam, transform
