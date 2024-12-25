from typing import Callable, Union, Dict
from dataclasses import dataclass
import torch
from torch import nn
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaSdpaAttention, LlamaMLP, LlamaDecoderLayer
from transformers import AutoModelForCausalLM as automodel_cls
from objectives import loss, layer_mappers, norm, projectors
import random
from mergelinear import MergeLinear 
class Objective(nn.Module):
    """
    Comprehensive distillation objective to calculate loss based on various features.

    Implements __call__(teacher_model, student_model, inputs) -> loss

    Mechanism
    ---------
    Runs forward pass and retrieves forward pass features
    - `out_s = student_model.forward()` (with gradients)
    - `out_t = teacher_model.forward()` (WITHOUT gradients)

    Then applies a loss function, `return loss(out_s, out_t)`

    Forward Pass Features
    ---------------------
    attentions:
      - Tuple shape: (num_layers,)
      - Tensor shape: (batch_size, num_attention_heads, sequence_length, sequence_length)
      - Contains attention scores for each layer.

    hidden_states:
      - Tuple shape: (num_layers + 1,)
      - Tensor shape: (batch_size, sequence_length, hidden_state_size)
      - Represents hidden states for each layer and the initial embedding.

    past_key_values:
      - Tuple shape: (num_layers, 2,)
      - Tensor shape: (batch_size, num_attention_heads, sequence_length, embedding_size_per_head)
      - keys and values (tuple of 2) for faster decoding.


    First Layer Data Flow
    ---------------------
    Embedding (hidden_states[0])
    -> MHA (attentions[0], past_key_values[0])
    -> FFN (updated hidden_states[1])
    -> Next Layer
    """
    def __init__(
            self,
            distil,
            name,
            *args
    ):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.count = 0
        self.loss_kl = loss.kl_divergence_loss
        self.distil = distil
        if distil:
            self.teacher_model = automodel_cls.from_pretrained(
                    name,
                    device_map="cuda",
                    torch_dtype = torch.bfloat16,
                    attn_implementation="flash_attention_2"
                    )

    def forward(self, model, inputs) -> Dict[str, float]:
        # this is done slightly different compared to the Mohawk framework, but andrew said it worked so we can keep it as id for now
        self.update_step(model)
        if self.distil:
            return self.distil_loss(model, inputs)
        return self.loss(model, inputs) 

    def loss(self, model, inputs):
        forward_kwargs = {
            **inputs,
            "output_hidden_states": False, 
            "output_attentions": False, 
        }
        out = model(**forward_kwargs)

        labels = inputs['input_ids'][:, 1:].contiguous().view(-1)
        logits = out.logits[:,:-1,:].contiguous()
        logits = logits.view(logits.shape[0]*logits.shape[1], logits.shape[2])
        CE = self.cross_entropy(logits, labels)

        return {"loss": CE, "crossentropy": CE}
    def distil_loss(self, model, inputs):
        forward_kwargs = {
            **inputs,
            "output_hidden_states": False,
            "output_attentions": False,
        }
        # get student / teacher forward pass outputs
        with torch.no_grad():
            out_t = self.teacher_model(**forward_kwargs)
        out_s = model(**forward_kwargs)

        loss_logits = self.loss_kl(out_s.logits, out_t.logits)
        labels = inputs['input_ids'][:, 1:].contiguous().view(-1)
        logits = out_s.logits[:,:-1,:].contiguous()
        logits = logits.view(logits.shape[0]*logits.shape[1], logits.shape[2])
        CE = self.cross_entropy(logits, labels)
        lin = self.linear_loss(model)
        loss = loss_logits + CE + lin
        return {"loss": loss, "loss/logits": loss_logits, "loss/crossentropy": CE}
    

    def reset_params(self, student_model, teacher_model, group_ids):
        for i in group_ids:
            # print("replacing", i)
            s_layer = student_model.module.model.layers[i]
            t_layer = teacher_model.model.layers[i]
            self.replace_weights(s_layer, t_layer)

    def replace_weights(self, s, t):
        for name, child in s.named_children():
            if isinstance(child, torch.nn.Linear):
                # print(name, child.weight)
                weight = self.get_weights(t, name)
                self.weight = weight
                # print(name, child.weight)
            self.replace_weights(child, t)

    def get_weights(self, module, name):
        for name2, child in module.named_children():
            if isinstance(child, torch.nn.Linear) and name == name2:
                weight = torch.nn.Parameter(child.weight, requires_grad = True)
                # print(weight)
                return weight
            self.get_weights(child, name)
    def update_step(self, model):
        for name, module in model.module.model.named_modules():
            module.global_step = self.count
    def linear_loss(self, model):
        loss = 0.0
        for name, module in model.module.model.named_modules():
            if isinstance(module, MergeLinear):
                tmp = module.get_loss()
                loss = loss + tmp
        return loss

