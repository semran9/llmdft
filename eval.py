import lm_eval
import torch
from torch import nn
import torch.distributions as dist
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaSdpaAttention, LlamaMLP, LlamaDecoderLayer
import argparse
import transformers
from safetensors.torch import load_file
from transformers.modeling_utils import load_sharded_checkpoint

import math
from alg.mergelinear import UniversalPath, Timer

class MergeLinear(nn.Linear):
    def __init__(self,
            *kargs,
            **kwargs
        ):
        super(MergeLinear, self).__init__(*kargs, **kwargs)
        """
        This is only for training, and kernel optimization is needed for efficiency.
        """
        self.end_step = 1 #step at which this layer is fully quantized
        self.global_step = 0 #step at which we are currently at
        self.time = torch.tensor(0)
        self.general = UniversalPath(self.out_features)
        self.w_1 = nn.Parameter(torch.zeros_like(self.weight))
        self.timer = Timer(self.out_features)
        self.eval = False


    def forward(self, x):
        """i
        Args:
        x: an input tensor with shape [n, d]
        Returns:
        y: an output tensor with shape [n, d]
        """
        if self.eval:
            w_0 = self.weight.clone().detach()
            x = x.to(w_0.device)
            y = F.linear(x, w_0)
            if self.bias is not None:
                y = y + self.bias
            return y
        w_0 = self.weight.clone().detach()
        w_1 = self.w_1.clone().detach()  # a weight tensor with shape [d, k]
        x = x.to(w_0.device)
        w_t = self.general(w_0, w_1, self.get_time())

        y = F.linear(x, w_t) # * self.scale_out
        if self.bias is not None:
            y = y + self.bias
        return y

    def get_time(self):
        return self.timer()

    def eval_init(self):
        self.eval = True
        w_0 = self.weight.clone().detach()
        w_1 = self.w_1.clone().detach().to(w_0.device)
        w_t = self.general(w_0, w_1, self.get_time())
        print(self.get_time())
        self.weight.data = w_t
        self.weight.requires_grad = False
    
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
            merge_layer.general.weight.data.copy_(path_weight)
            weight = module.weight.data / path_weight
            merge_layer.weight.data.copy_(weight)
            merge_layer.weight.requires_grad = False
            if module.bias is not None:
                merge_layer.bias.data.copy_(module.bias.data)

            # Get corresponding layer from model2 and copy to w_1
            model2_module = dict(model2.named_modules())[name]
            m2_weight = model2_module.weight.data.clone()
            merge_layer.w_1.data = m2_weight / torch.norm(m2_weight, dim=1, keepdim=True)
            # merge_layer.w_1 = merge_layer.w_1.to(merge_layer.weight.device)
            # merge_layer.timer = time.to(merge_layer.weight.device)
            # merge_layer.eval_init()
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

def get_model(model1n, model2n):
    model = transformers.AutoModelForCausalLM.from_pretrained(
            model1n,
            device_map="cuda",
            torch_dtype = torch.bfloat16
        )
    model2 = transformers.AutoModelForCausalLM.from_pretrained(
            model2n,
            device_map="cuda",
            torch_dtype = torch.bfloat16
        )
    model.type(torch.bfloat16)
    model2.type(torch.bfloat16)
    # print(student_model)
    with torch.no_grad():
            # TODO: use a different method which is better supported, an official third party library
       model = merge(model, model2)
       # student_model.model_tags = ["bitnet", "1.58b"]
    #loaded = load_file(path+"/model.safetensors")
    #student_model.load_state_dict(loaded)
    print("merged layers initialized. loading weights")
    load_sharded_checkpoint(model, model1n, strict = False)
    model.type(torch.bfloat16)
    model = model.cuda()
    for name, module in model.named_modules():
        if isinstance(module, MergeLinear):
            module.eval_init()
    print("in evaluation mode. weights have been calculated")
    # print(student_model.parameters())
    model.type(torch.bfloat16)
    return model



def eval(model, tasks, shots = 0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = lm_eval.models.huggingface.HFLM(pretrained=model.to(device), batch_size = 8)
    results = lm_eval.simple_evaluate(model = model, tasks = tasks, num_fewshot = shots)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model using lm-eval harness.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the local model checkpoint.",
    )
    parser.add_argument(
        "--merge_model_name",
        type=str,
        default="Qwen/Qwen2-1.5B",
        help="Model name (default: gpt2).",
    )
    parser.add_argument(
        "--tasks",
        nargs='+',
        default=["hellaswag"],
        help="List of evaluation tasks (default: hellaswag).",
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=0,
        help="Number of few-shot examples (default: 0 for zero-shot).",
    )
    parser.add_argument(
        "--lossless",
        default=False,
        action="store_true",
        help="Use lossless compression.",
    )
    args = parser.parse_args()
    t = get_model(args.checkpoint_path, args.merge_model_name)
    # from transformers import AutoTokenizer, AutoModelForCausalLM
    # t = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base")
    r = eval(t, args.tasks, args.num_fewshot)
    # print("hellaswag acc:", r['results']['hellaswag']['acc_norm,none'])
    # print("piqa:", r['results']['piqa'])
    print("results:", r['results'])
