from torchvision import transforms
import torch.nn as nn
from DCLS.construct.modules import Dcls2d


def replace_dots_brackets(name):
    name_split = name.split('.')
    name_split = ['[' + i + ']' if i.isdigit() else '.' + i for i in name_split]

    return ''.join(name_split[:-1]), name_split[-1][1:]


def replace_depthwise_dcls_resnet(model, dilated_kernel_size=23,
                                  kernel_count=26, version='gauss'):
    in_channels, P, SIG = 0, None, None
    # Loop over all model modules
    for name, module in model.named_modules():
        # if the module is a depthwise separable Conv2d module

        if isinstance(module, nn.Conv2d) and module.kernel_size == (3, 3):
            name_eval, last_layer = replace_dots_brackets(name)
            if module.bias == None:
                dcls_conv = Dcls2d(module.in_channels, module.out_channels,
                                   kernel_count=kernel_count,
                                   dilated_kernel_size=dilated_kernel_size,
                                   padding=dilated_kernel_size // 2,
                                   groups=1, stride=module.stride,
                                   bias=False, version=version)
            else:
                dcls_conv = Dcls2d(module.in_channels, module.out_channels,
                                   kernel_count=kernel_count,
                                   dilated_kernel_size=dilated_kernel_size,
                                   padding=dilated_kernel_size // 2,
                                   groups=1, stride=module.stride,
                                   bias=True, version=version)
                dcls_conv.bias = nn.Parameter(module.bias.detach().clone())
                nn.init.constant_(dcls_conv.bias, 0)

            nn.init.normal_(dcls_conv.weight, std=.02)

            # Synchronise positions and standard
            # deviations belonging to the same stage
            if in_channels < module.in_channels:
                in_channels = module.in_channels
                P, SIG = dcls_conv.P, dcls_conv.SIG

            dcls_conv.P, dcls_conv.SIG = P, SIG

            setattr(eval("model" + name_eval), last_layer, dcls_conv)
    return model
