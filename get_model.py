import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torch
from models.fastvit.fastvit import fastvit_sa24, fastvit_sa36
from models.fastvit.fastvit_gradCAM import FastvitGradCAM
from models.fastvit.fastvit_utils import replace_depthwise_dcls_fastvit, transform_fastvit
from utils import device
from models.convnext.convnext import convnext_small, convnext_base, convnext_tiny
from models.convnext.convnext_utils import replace_depthwise_dcls_cnvnxt, transform_convnext
from models.convnext.convnext_gradCAM import ConvNextGradCAM
import copy
from models.resnet.resnet_utils import replace_depthwise_dcls_resnet
from models.resnet.resnet_gradCAM import resnetGradCAM
from models.metaformer.metaformer_baselines import convformer_s18, caformer_s18
from models.metaformer.metaformers_utils import replace_depthwise_dcls_metaformers, meta_transform
from models.metaformer.metaformers_gradCAM import MetaformerGradCAM


def get_model(model_name, dcls_equipped=True, pretrained=True):
    if model_name == "fastvit_sa36":
        if dcls_equipped:
            model = fastvit_sa36(pretrained=False)
            model = replace_depthwise_dcls_fastvit(copy.deepcopy(model),
                                                   dilated_kernel_size=17,
                                                   kernel_count=34, version='v1')
            if pretrained:
                checkpoint = torch.hub.load_state_dict_from_url(
                    url='https://zenodo.org/records/8370737/files/fastvit_sa36_v1_17lim_34el_seed0.pth.tar',
                    map_location=device, check_hash=True)
                model_state_dict = checkpoint['state_dict_ema']
                model.load_state_dict(model_state_dict, strict=True)
        else:
            model = fastvit_sa36(pretrained=pretrained).to(device)
        model_cam = FastvitGradCAM(model)
        return model, model_cam, transform_fastvit

    if model_name == "fastvit_sa24":
        if dcls_equipped:
            model = fastvit_sa24(pretrained=False).to(device)
            model = replace_depthwise_dcls_fastvit(copy.deepcopy(model),
                                                   dilated_kernel_size=17,
                                                   kernel_count=34, version='v1')
            if pretrained:
                checkpoint = torch.hub.load_state_dict_from_url(
                    url='https://zenodo.org/records/8370737/files/fastvit_sa24_v1_17lim_34el_seed0.pth.tar',
                    map_location="cpu", check_hash=True)
                model_state_dict = checkpoint['state_dict_ema']
                model.load_state_dict(model_state_dict, strict=True)
        else:
            model = fastvit_sa24(pretrained=pretrained).to(device)
        model_cam = FastvitGradCAM(model)
        return model, model_cam, transform_fastvit

    if model_name == "convnext_tiny":
        if dcls_equipped:
            model = convnext_tiny(pretrained=False).to(device)
            model = replace_depthwise_dcls_cnvnxt(copy.deepcopy(model), dilated_kernel_size=17, kernel_count=34,
                                                  version='v0').to(device)
            if pretrained:
                url = "https://zenodo.org/record/7112021/files/convnext_dcls_tiny_1k_224_ema.pth"
                checkpoints = torch.hub.load_state_dict_from_url(url=url, map_location=device, check_hash=True)
                model.load_state_dict(checkpoints['model'])
        else:
            model = convnext_tiny(pretrained=pretrained).to(device)
        model_cam = ConvNextGradCAM(model)
        return model, model_cam, transform_convnext

    if model_name == "convnext_small":
        if dcls_equipped:
            model = convnext_small(pretrained=False).to(device)
            model = replace_depthwise_dcls_cnvnxt(copy.deepcopy(model), dilated_kernel_size=17, kernel_count=40,
                                                  version='v0').to(device)
            if pretrained:
                url = "https://zenodo.org/records/7112021/files/convnext_dcls_small_1k_224_ema.pth"
                checkpoints = torch.hub.load_state_dict_from_url(url=url, map_location=device, check_hash=True)
                model.load_state_dict(checkpoints["model"])
        else:
            model = convnext_small(pretrained=pretrained).to(device)

        model_cam = ConvNextGradCAM(model)
        return model, model_cam, transform_convnext

    if model_name == "convnext_base":
        if dcls_equipped:
            model = convnext_base(pretrained=False).to(device)
            model = replace_depthwise_dcls_cnvnxt(copy.deepcopy(model), dilated_kernel_size=17, kernel_count=40,
                                                  version='v0').to(device)
            if pretrained:
                url = "https://zenodo.org/record/7112021/files/convnext_dcls_base_1k_224_ema.pth"
                checkpoints = torch.hub.load_state_dict_from_url(url=url, map_location=device, check_hash=True)
                model.load_state_dict(checkpoints['model'])
        else:
            model = convnext_base(pretrained=pretrained).to(device)

        model_cam = ConvNextGradCAM(model)

        return model, model_cam, transform_convnext
    if model_name == "caformer_s18":
        if dcls_equipped:
            model = caformer_s18(pretrained=False).to(device)
            model = replace_depthwise_dcls_metaformers(copy.deepcopy(model),
                                                       dilated_kernel_size=17,
                                                       kernel_count=34, version='v1').to(device)
            if pretrained:
                url = "https://zenodo.org/records/8370737/files/caformer_s18_v1_17lim_34el_seed0.pth.tar?download=1"
                checkpoints = torch.hub.load_state_dict_from_url(url=url, map_location=device, check_hash=True)
                model.load_state_dict(checkpoints['state_dict_ema'], strict=False)
        else:
            model = caformer_s18(pretrained=pretrained).to(device)

        model_cam = MetaformerGradCAM(model)

        return model, model_cam, meta_transform
    if model_name == "convformer_s18":
        if dcls_equipped:
            model = convformer_s18(pretrained=False).to(device)
            model = replace_depthwise_dcls_metaformers(copy.deepcopy(model),
                                                   dilated_kernel_size=17,
                                                   kernel_count=40, version='v1').to(device)
            if pretrained:
                url = "https://zenodo.org/records/8370737/files/convformer_s18_v1_17lim_40el_seed0.pth.tar"
                checkpoints = torch.hub.load_state_dict_from_url(url=url, map_location=device, check_hash=True)
                model.load_state_dict(checkpoints['state_dict_ema'], strict=True)
        else:
            model = convformer_s18(pretrained=pretrained).to(device)

        model_cam = MetaformerGradCAM(model)

        return model, model_cam, meta_transform
    if model_name == "resnet50":
        if dcls_equipped:
            model = timm.create_model('resnet50', pretrained=False).to(device)
            # (ii) the preprocessing
            model = replace_depthwise_dcls_resnet(copy.deepcopy(model),
                                                  dilated_kernel_size=7,
                                                  kernel_count=5, version='v0').to(device)
            if pretrained:
                url = "https://zenodo.org/records/8373830/files/resnet_dcls_kernel_5_model_best.pth.tar"
                checkpoints = torch.hub.load_state_dict_from_url(url=url, map_location=device, check_hash=True)
                model.load_state_dict(checkpoints['state_dict_ema'])
        else:
            model = timm.create_model('resnet50', pretrained=pretrained).to(device)

        model_cam = resnetGradCAM(model)
        config = resolve_data_config({}, model=model)
        transform = create_transform(**config)
        return model, model_cam, transform
