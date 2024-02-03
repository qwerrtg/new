# coding=utf-8

# The following code has been taken from https://github.com/NVIDIA/NeMo/blob/ \
# 782b4e1652aaa43c8be390d9db0dc89544afa080/nemo/collections/nlp/modules/ \
# common/megatron/rotary_pos_embedding.py

import importlib.util
import torch

from torch import einsum, nn

__all__ = ['RotaryEmbedding', 'apply_rotary_pos_emb']

class RotaryEmbedding(nn.Module):
    inv_freq_float: torch.Tensor

    def __init__(self, dim):
        super().__init__()
        # inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        inv_freq = 1.0 / (1000000 ** (torch.arange(0, dim, 2).float() / dim))
        self.inv_freq_float = inv_freq
        self.register_buffer('inv_freq', inv_freq)
        if importlib.util.find_spec('einops') is None:
            raise RuntimeError("einops is required for Rotary Embedding")

    def forward(self, max_seq_len, offset=0):
        seq = torch.arange(max_seq_len, device=self.inv_freq.device) + offset
        from einops import rearrange
        if self.inv_freq_float.device != seq.device:
            self.inv_freq_float = self.inv_freq_float.to(seq.device)
        # NOTE: adaptive to iFlytekSpark
        freqs_float = einsum('i , j -> i j', seq.type_as(self.inv_freq_float), self.inv_freq_float)
        emb_float = torch.cat((freqs_float, freqs_float), dim=-1)
        # cos_cached = emb_float.cos().type_as(self.inv_freq)
        # cos_cached = rearrange(cos_cached, 'n d -> n 1 1 d')
        # sin_cached = emb_float.sin().type_as(self.inv_freq)
        # sin_cached = rearrange(sin_cached, 'n d -> n 1 1 d')
        return rearrange(emb_float, 'n d -> n 1 1 d')

        freqs = einsum('i , j -> i j', seq.type_as(self.inv_freq), self.inv_freq)
        # first part even vector components, second part odd vector components,
        #  2 * dim in dimension size
        emb = torch.cat((freqs, freqs), dim=-1)
        # emb [seq_length, .., dim]
        return rearrange(emb, 'n d -> n 1 1 d')


def _rotate_half(x):
    """
    change sign so the last dimension becomes [-odd, +even]
    """
    from einops import rearrange
    x = rearrange(x, '... (j d) -> ... j d', j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, freqs):
    """
    input tensor t is of shape [seq_length, ..., dim]
    rotary positional embeding tensor freqs is of shape [seq_length, ..., dim]
    check https://kexue.fm/archives/8265 for detailed formulas
    """
    rot_dim = freqs.shape[-1]
    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    cos = freqs.cos().type_as(t)
    sin = freqs.sin().type_as(t)
    t = t * cos + _rotate_half(t) * sin
    # t = (t * freqs.cos()) + (_rotate_half(t) * freqs.sin())
    return torch.cat((t, t_pass), dim=-1)
