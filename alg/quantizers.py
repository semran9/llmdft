import torch
from torch import nn
import torch.distributions as dist
import torch.nn.functional as F

class TwnQuantizer(torch.autograd.Function):
    """Ternary Weight Networks (TWN)
    Ref: https://arxiv.org/abs/1605.04711
    """

    @staticmethod
    def forward(ctx, input, group_size = -1, per_tensor = False, max_scale=0.7):
        """
        :param input: tensor to be ternarized
        :return: quantized tensor
        """
        ctx.save_for_backward(input)

        org_w_shape = input.shape
        q_group_size = group_size

        if q_group_size > 0:
            assert org_w_shape[-1] % q_group_size == 0
            input = input.reshape(-1, q_group_size)
        else:
            input = input.reshape(-1, input.shape[-1])

        if per_tensor: assert q_group_size == -1, "Conflict with Per Tensor and Per Group Quant!"

        if per_tensor:
            # Per Tensor Quantizaiton
            m = input.abs().mean()
            thres = max_scale * m
            pos = (input > thres).float()
            neg = (input < -thres).float()
            mask = (input.abs() > thres).float()
            alpha = (mask * input).abs().sum() / mask.sum()
            result = alpha * pos - alpha * neg
        else:
            # Per Channel/Group Quantization
            n = input[0].nelement()
            m = input.data.norm(p=1, dim=1).div(n)
            thres = (max_scale * m).view(-1, 1).expand_as(input)
            pos = (input > thres).float()
            neg = (input < -thres).float()
            mask = (input.abs() > thres).float()
            alpha = ((mask * input).abs().sum(dim=1) / mask.sum(dim=1)).view(-1, 1)
            result = alpha * pos - alpha * neg

        result = result.reshape(org_w_shape) # for per-group quantization

        return result

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        # input, clip_val = ctx.saved_tensors  # unclipped input
        input = ctx.saved_tensors  # unclipped input
        grad_input = grad_output.clone()
        # grad_input[input.ge(clip_val[1])] = 0
        # grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None, None
