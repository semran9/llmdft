# Based on https://github.com/pranavjad/tinyllama-bitnet/blob/main/utils.py
import torch
from torch import nn
import torch.distributions as dist
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaSdpaAttention, LlamaMLP, LlamaDecoderLayer
## TODO: fix everything

class MergeLinear(nn.Linear):
    def __init__(self,
            *kargs,
            **kwargs
        ):
        super(MergeLinear, self).__init__(*kargs, **kwargs)
        """
        This is only for training, and kernel optimization is needed for efficiency.
        """
        self.scales = nn.Parameter(torch.ones(self.out_features, 1), requires_grad = True)
        

    def forward(self, x):
        """i
        Args:
        x: an input tensor with shape [n, d]
        Returns:
        y: an output tensor with shape [n, d]
        """
        
        y_dot = self.weight.clone().detach()
        x = x.to(y_dot.device)
        # norm scheduling
        t = self.linear_scheduler()
        w_t = self.general(y_dot, self.get_time())
        w_t = t*w_t + (1-t)*y_dot
        # introducing weight schduling could be interesting here
        y = F.linear(x, w_t) 
        if self.bias is not None:
            y = y + self.bias
        return y
    
    
    
    
def merge(model1, model2):
    import copy
    # Create a new model by copying model1's structure
    merged_model = copy.deepcopy(model1)
    # time = Timer()
    # Iterate through all modules in the model
    for name, module in merged_model.named_modules():
        # Replace Linear layers with MergeLinear layers
        if isinstance(module, nn.Linear) and "lm_head" not in name:
            # Create new MergeLinear with same dimensions
            merge_layer = MergeLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None
            )
            
            # Copy weights from model1 to main weights
            path_weight = torch.norm(module.weight.data, dim=1, keepdim=True)
            # path_weight.requires_grad = False
            merge_layer.scales.data.copy_(path_weight)
            weight = module.weight.data / path_weight
            merge_layer.weight.data.copy_(weight)
            merge_layer.weight.requires_grad = False
            if module.bias is not None:
                merge_layer.bias.data.copy_(module.bias.data)
            # Replace the layer in merged_model
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            if parent_name:
                parent = merged_model.get_submodule(parent_name)
                setattr(parent, child_name, merge_layer)
            else:
                setattr(merged_model, child_name, merge_layer)
    del model1
    del model2
    return merged_model

