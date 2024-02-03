# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Utilities for models."""

import math

import torch
import torch.nn.functional as F

from megatron import get_args

def init_method_normal(sigma):
    """Init method based on N(0, sigma)."""
    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_


def scaled_init_method_normal(sigma, num_layers):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = sigma / math.sqrt(2.0 * num_layers)

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


def attention_mask_func(attention_scores, attention_mask):
    args = get_args()
    if args.curriculum_learning_legacy or args.data_efficiency_curriculum_learning:
        attention_mask_ = attention_mask
        actual_seqlen = attention_scores.size()[2]
        if actual_seqlen != attention_mask_.size()[2]:
            # attention_mask has size [1, 1, seqlen, seqlen]
            attention_mask_ = attention_mask_[:, :, :actual_seqlen, :actual_seqlen].contiguous()
        attention_scores.masked_fill_(attention_mask_, -10000.0)
    else:
        attention_scores.masked_fill_(attention_mask, -10000.0)
    return attention_scores


def get_linear_layer(rows, columns, init_method):
    """Simple linear layer with weight initialization."""
    layer = torch.nn.Linear(rows, columns)
    if get_args().perform_initialization:
        init_method(layer.weight)
    with torch.no_grad():
        layer.bias.zero_()
    return layer

@torch.jit.script
def gelu_impl(x):
    """OpenAI's gelu implementation."""
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x *
                                       (1.0 + 0.044715 * x * x)))
def openai_gelu(x):
    return gelu_impl(x)

#This is actually Python equivalent of torch.nn.functional.gelu(), also with type hints for ONNX exporter
@torch.jit.script
def erf_gelu(x):
    return x * 0.5 * (torch.erf(x / 1.41421).to(dtype=x.dtype)+torch.ones_like(x).to(dtype=x.dtype))

def gate_gelu(intermediate_parallel):
    # fix bug: chunk may cause incomformity between different tensor parallel
    hshape= intermediate_parallel.shape[:-1]
    intermediate_parallel= intermediate_parallel.view(hshape+(-1,2))
    intermediate_parallel1,intermediate_parallel2= intermediate_parallel[...,0],intermediate_parallel[...,1]
    # intermediate_parallel1, intermediate_parallel2 = torch.chunk(intermediate_parallel, 2, dim=-1)
    # set_trace()
    intermediate_parallel1 = F.gelu(intermediate_parallel1)
    intermediate_parallel = intermediate_parallel1 * intermediate_parallel2

    return intermediate_parallel
