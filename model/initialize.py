from torch import nn
import torch.nn.init as init

def hierarchical_initialize_weights(module):
    """
    Custom initialization for various activation functions: ReLU, SELU, LeakyReLU, SiLU, GELU.
    Recursively initializes nested submodules.
    """
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        # Check if the module has a specified activation attribute
        if hasattr(module, "activation"):
            if module.activation == "relu":
                # Kaiming Normal for ReLU
                init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    init.constant_(module.bias, 0)
            elif module.activation == "leakyrelu":
                # Kaiming Normal for LeakyReLU with slope=0.01
                init.kaiming_uniform_(module.weight, a=0.01, nonlinearity='leaky_relu')
                if module.bias is not None:
                    init.constant_(module.bias, 0)
            elif module.activation == "selu":
                # LeCun Normal for SELU
                init.normal_(module.weight, mean=0, std=1.0 / module.weight.shape[1] ** 0.5)
                if module.bias is not None:
                    init.constant_(module.bias, 0)
            elif module.activation == "silu":
                # Kaiming Normal for SiLU
                init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    init.constant_(module.bias, 0)
            elif module.activation == "gelu":
                # Kaiming Normal for GELU
                init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm2d):
        # Initialize BatchNorm weights to 1 and biases to 0
        init.constant_(module.weight, 1)
        init.constant_(module.bias, 0)
    elif isinstance(module, nn.Module):
        # Recursively initialize all submodules
        for submodule in module.children():
            hierarchical_initialize_weights(submodule)