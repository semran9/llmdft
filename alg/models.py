import copy
import functools

import transformers
import torch

from mergelinear import merge

def zero_grad_hook(grad):
    return torch.zeros_like(grad)

def get_model_tokenizer(model_args, **model_kwargs):
    if model_args.use_lk:
        from liger_kernel.transformers import AutoLigerKernelForCausalLM
        # automodel_cls = AutoLigerKernelForCausalLM
        # TODO: remove hack below, use above comment once https://github.com/linkedin/Liger-Kernel/issues/242 is fixed
        class PatchedAutoLiger(AutoLigerKernelForCausalLM):
            @staticmethod
            def from_config(config, *args, **kwargs):
                AutoLigerKernelForCausalLM.from_pretrained(config._name_or_path)
                return AutoLigerKernelForCausalLM.from_config(config, *args, **kwargs)
        automodel_cls = PatchedAutoLiger

    else:
        automodel_cls = transformers.AutoModelForCausalLM
    model1 = automodel_cls.from_pretrained(
        model_args.model_1_name,
        device_map="cuda",
        attn_implementation="flash_attention_2",
        **model_kwargs
    )
    model1 = model1.type(torch.bfloat16)
    if model_args.model_2_name != "random":
        model2 = automodel_cls.from_pretrained(
            model_args.model_2_name,
            device_map="cuda", 
            attn_implementation="flash_attention_2",
            **model_kwargs
        )
        model2 = model2.type(torch.bfloat16)
    
        model = merge(model1, model2)
    else:
        model = merge(model1, model_args.model_2_name)
    print(model)
    # freeze (maybe redundant)
    # model.eval()
    # for p in model.parameters():
    #     p.requires_grad = False

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_1_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # model.model.embed_tokens.weight.requires_grad = False
    model.model.embed_tokens.weight.register_hook(zero_grad_hook)
    return model, tokenizer

