import torch
from models.fastvit.fastvit import fastvit_sa24, fastvit_sa36, fastvit_sa36_dcls, fastvit_sa24_dcls
from models.fastvit.fastvit_gradCAM import FastvitGradCAM
from utils import device
from transforms import fastvit_transform
from models.convnext.convnext import convnext_small, convnext_base, convnext_tiny
from models.convnext.convnext_utils import replace_depthwise_dcls_cnvnxt, transform_convnext
from models.convnext.convnext_gradCAM import ConvNextGradCAM
import copy


def get_model(model_name, dcls_equipped=True, pretrained=True):
    if model_name == "fastvit_sa36":
        if dcls_equipped:
            model = fastvit_sa36_dcls(pretrained=pretrained).to(device)
            # if pretrained:
            #     checkpoint = torch.load('link to dcls checkpoints')
            #     model_state_dict = checkpoint['state_dict_ema']
            #     model.load_state_dict(model_state_dict, strict=True)
        else:
            model = fastvit_sa36().to(device)
            if pretrained:
                checkpoint = torch.load('link to baseline checkpoints')
                model_state_dict = checkpoint['state_dict_ema']
                model.load_state_dict(model_state_dict, strict=True)
        model_cam = FastvitGradCAM(model)

    if model_name == "fastvit_sa24":
        if dcls_equipped:
            model = fastvit_sa24_dcls(pretrained=pretrained).to(
                device)  #r"C:\Users\Layal\Downloads\fastvit_sa24_v1_seed0.pth(1)(1).tar"
        # if pretrained:
        #     file_path = Path("C:/Users/Layal/Downloads/fastvit_sa24_v1_seed0.pth (1) (1).tar")
        #     checkpoint = torch.load(file_path, map_location=device)
        #     model_state_dict = checkpoint['state_dict_ema']
        #     model.load_state_dict(model_state_dict, strict=True)
        else:
            model = fastvit_sa24(pretrained=pretrained).to(device)
            # if pretrained:
            #     checkpoint = torch.load('link to baseline checkpoints')
            #     model_state_dict = checkpoint['state_dict_ema']
            #     model.load_state_dict(model_state_dict, strict=True)
        model_cam = FastvitGradCAM(model)
        transform = fastvit_transform

    if model_name == "convnext_tiny":
        model = convnext_tiny(pretrained=pretrained).to(device)

        if dcls_equipped:
            model = replace_depthwise_dcls_cnvnxt(copy.deepcopy(model), dilated_kernel_size=17, kernel_count=34,
                                                  version='v0').to(device)
            if pretrained:
                url = "https://zenodo.org/record/7112021/files/convnext_dcls_tiny_1k_224_ema.pth"
                checkpoints = torch.hub.load_state_dict_from_url(url=url, map_location=device, check_hash=True)
                model.load_state_dict(checkpoints["model"])
        model_cam = ConvNextGradCAM(model)
        return model, model_cam, transform_convnext

    if model_name == "convnext_small":
        model = convnext_small(pretrained=pretrained).to(device)

        if dcls_equipped:
            model = replace_depthwise_dcls_cnvnxt(copy.deepcopy(model), dilated_kernel_size=17, kernel_count=34,
                                                  version='v0').to(device)
            if pretrained:
                url = "https://zenodo.org/records/7112021/files/convnext_dcls_small_1k_224_ema.pth"
                checkpoints = torch.hub.load_state_dict_from_url(url=url, map_location=device, check_hash=True)
                model.load_state_dict(checkpoints["model_ema"])
        model_cam = ConvNextGradCAM(model)
        return model, model_cam, transform_convnext

    if model_name == "convnext_base":
        print(device)
        model = convnext_tiny(pretrained=pretrained).to(device)

        if dcls_equipped:
            model = replace_depthwise_dcls_cnvnxt(copy.deepcopy(model), dilated_kernel_size=17, kernel_count=34,
                                                  version='v0').to(device)
            if pretrained:
                url = "https://zenodo.org/record/7112021/files/convnext_dcls_tiny_1k_224_ema.pth"
                checkpoints = torch.hub.load_state_dict_from_url(url=url, map_location=device, check_hash=True)
                model.load_state_dict(checkpoints["model"])
        model_cam = ConvNextGradCAM(model)
        return model, model_cam, transform_convnext
