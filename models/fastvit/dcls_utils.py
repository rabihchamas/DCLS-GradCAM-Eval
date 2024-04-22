

import torch.nn as nn
from DCLS.construct.modules import  Dcls2d

# Helper function that replaces all ".int." patterns
# by "[int]" in a character string
def replace_dots_brackets(name):
  name_split = name.split('.')
  name_split = ['[' + i + ']' if i.isdigit() else '.' + i  for i in name_split]

  return ''.join(name_split[:-1]), name_split[-1][1:]


# 2D depthwise separable convolution in
# a model by synchronized Dcls2d ones
def replace_depthwise_dcls(
    model, dilated_kernel_size=17, kernel_count=34, version="v1"
):
    in_channels, P, SIG = 0, None, None
    # Loop over all model modules
    for name, module in model.named_modules():
        # if the module is a depthwise separable Conv2d module
        if (
            isinstance(module, nn.Conv2d)
            and module.groups == module.in_channels == module.out_channels
            and module.kernel_size == (7, 7)
        ):
            name_eval, last_layer = replace_dots_brackets(name)
            dcls_conv = Dcls2d(
                module.in_channels,
                module.out_channels,
                kernel_count=kernel_count,
                dilated_kernel_size=dilated_kernel_size,
                padding=dilated_kernel_size // 2,
                groups=module.in_channels,
                version=version,
                bias=module.bias is not None,
            )
            nn.init.normal_(dcls_conv.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(dcls_conv.bias, 0)

            # Synchronise positions and standard
            # deviations belonging to the same stage
            if in_channels < module.in_channels:
                in_channels = module.in_channels
                P, SIG = dcls_conv.P, dcls_conv.SIG

            dcls_conv.P, dcls_conv.SIG = P, SIG

            setattr(eval("model" + name_eval), last_layer, dcls_conv)
    return model