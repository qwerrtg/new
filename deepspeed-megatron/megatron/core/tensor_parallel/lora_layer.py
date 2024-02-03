from typing import Dict

import torch
import torch.nn as nn

from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear

class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class LoRAColumnParallelLinear(ColumnParallelLinear, LoRALayer):
    def __init__(
        self,
        in_features,
        out_features,
        config,
        init_method,
        r = 0,
        lora_alpha = 1,
        lora_dropout = 0.,
        merge_weights = True,
        bias = True,
        partition_stride = 1,
        skip_bias_add = False,
        gather_output = True,
        use_cpu_initialization = True,
    ):
        ColumnParallelLinear.__init__(
            self,
            input_size=in_features,
            output_size=out_features,
            config=config,
            init_method=init_method,
            bias=bias,
            stride=partition_stride, 
            skip_bias_add=skip_bias_add,
            gather_output=gather_output,
        )
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights
        )

        if r > 0:
            self.lora_A = nn.Linear(
                in_features=in_features,
                out_features=r,
                bias=False,
                device=self.weight.device,
                dtype=self.weight.dtype,
            )
            self.lora_B = ColumnParallelLinear(
                input_size=r,
                output_size=out_features,
                bias=False,
                config=config,
                init_method=nn.init.zeros_,
                stride=partition_stride,
                gather_output=gather_output,
            )
            setattr(self.lora_A.weight, "sequence_parallel", True)
            self.scaling = self.lora_alpha / self.r

    def train(self, mode: bool = True):
        ColumnParallelLinear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data = self.weight.data - self.lora_B.weight @ self.lora_A.weight * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += self.lora_B.weight @ self.lora_A.weight * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        output, output_bias = ColumnParallelLinear.forward(self, x)
        if self.r > 0 and not self.merged:
            x_d = self.lora_dropout(x)
            x_a = self.lora_A(x_d)
            x_b = self.lora_B(x_a)[0]
            output = output + x_b * self.scaling
        return output, output_bias


class LoRARowParallelLinear(RowParallelLinear, LoRALayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config,
        init_method,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        merge_weights: bool = True,
        bias: bool = True,
        partition_stride: int = 1,
        skip_bias_add: bool = False,
        input_is_parallel: bool = False,
        use_cpu_initialization: bool = False,
    ):
        RowParallelLinear.__init__(
            self,
            input_size=in_features,
            output_size=out_features,
            config=config,
            init_method=init_method,
            bias=bias,
            stride=partition_stride,
            skip_bias_add=skip_bias_add,
            input_is_parallel=input_is_parallel,
        )
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights
        )

        if r > 0:
            self.lora_A = RowParallelLinear(
                input_size=in_features,
                output_size=r,
                bias=False,
                config=config,
                init_method=nn.init.kaiming_uniform_,
                stride=partition_stride,
                input_is_parallel=input_is_parallel,
            )
            self.lora_B = nn.Linear(
                in_features=r,
                out_features=out_features,
                bias=False,
                device=self.weight.device,
                dtype=self.weight.dtype,
            )
            setattr(self.lora_B.weight, "sequence_parallel", True)
            nn.init.zeros_(self.lora_B.weight)
            self.scaling = self.lora_alpha / self.r

    def train(self, mode: bool = True):
        RowParallelLinear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data = self.weight.data - self.lora_B.weight @ self.lora_A.weight * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += self.lora_B.weight @ self.lora_A.weight * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        output, output_bias = RowParallelLinear.forward(self, x)
        if self.r > 0 and not self.merged:
            x_d = self.lora_dropout(x)
            x_a = self.lora_A(x_d)[0]
            x_b = self.lora_B(x_a)
            output = output + x_b * self.scaling

        return output, output_bias


class LoRAQKVParallelLinear(ColumnParallelLinear, LoRALayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config,
        init_method,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        adapt_q: bool = False,
        adapt_k: bool = False,
        adapt_v: bool = False,
        merge_weights: bool = True,
        bias: bool = True,
        partition_stride: int = 3,
        skip_bias_add: bool = False,
        gather_output: bool = True,
        use_cpu_initialization: bool = True,
    ):
        ColumnParallelLinear.__init__(
            self,
            input_size=in_features,
            output_size=out_features, 
            bias=bias,
            config=config,
            init_method=init_method,
            stride=partition_stride, 
            skip_bias_add=skip_bias_add,
            gather_output=gather_output,
        )
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights
        )
        self.adapt_q = adapt_q
        self.adapt_k = adapt_k
        self.adapt_v = adapt_v
        assert out_features % 3  == 0
        self.inner_out_features = out_features // 3
        
        if r > 0:
            if adapt_q:
                self.lora_A_Q = nn.Linear(
                    in_features=in_features,
                    out_features=r,
                    bias=False,
                    device=self.weight.device,
                    dtype=self.weight.dtype,
                )
                self.lora_B_Q = ColumnParallelLinear(
                    input_size=r,
                    output_size=self.inner_out_features,
                    bias=False,
                    config=config,
                    init_method=nn.init.zeros_,
                    stride=1,
                    gather_output=gather_output,
                )
                setattr(self.lora_A_Q.weight, "sequence_parallel", True)
            if adapt_k:
                self.lora_A_K = nn.Linear(
                    in_features=in_features,
                    out_features=r,
                    bias=False,
                    device=self.weight.device,
                    dtype=self.weight.dtype,
                )
                self.lora_B_K = ColumnParallelLinear(
                    input_size=r,
                    output_size=self.inner_out_features,
                    bias=False,
                    config=config,
                    init_method=nn.init.zeros_,
                    stride=1,
                    gather_output=gather_output,
                )
                setattr(self.lora_A_K.weight, "sequence_parallel", True)
            if adapt_v:
                self.lora_A_V = nn.Linear(
                    in_features=in_features,
                    out_features=r,
                    bias=False,
                    device=self.weight.device,
                    dtype=self.weight.dtype,
                )
                self.lora_B_V = ColumnParallelLinear(
                    input_size=r,
                    output_size=self.inner_out_features,
                    bias=False,
                    config=config,
                    init_method=nn.init.zeros_,
                    stride=1,
                    gather_output=gather_output,
                )
                setattr(self.lora_A_V.weight, "sequence_parallel", True)
            
            self.scaling = self.lora_alpha / self.r
            
    def train(self, mode: bool = True):
        ColumnParallelLinear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    weight_list = list(self.weight.data.chunk(chunks=3, dim=0))
                    if self.adapt_q:
                        weight_list[0] = weight_list[0] - self.lora_B_Q.weight @ self.lora_A_Q.weight * self.scaling
                    if self.adapt_k:
                        weight_list[1] = weight_list[1] - self.lora_B_K.weight @ self.lora_A_K.weight * self.scaling
                    if self.adapt_v:
                        weight_list[2] = weight_list[2] - self.lora_B_V.weight @ self.lora_A_V.weight * self.scaling
                    self.weight.data = torch.cat(weight_list, dim=0)
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    weight_list = list(self.weight.data.chunk(chunks=3, dim=0))
                    if self.adapt_q:
                        weight_list[0] += self.lora_B_Q.weight @ self.lora_A_Q.weight * self.scaling
                    if self.adapt_k:
                        weight_list[1] += self.lora_B_K.weight @ self.lora_A_K.weight * self.scaling
                    if self.adapt_v:
                        weight_list[2] += self.lora_B_V.weight @ self.lora_A_V.weight * self.scaling
                    self.weight.data = torch.cat(weight_list, dim=0)
                self.merged = True

    def forward(self, x: torch.Tensor):
        output, output_bias = ColumnParallelLinear.forward(self, x)
        if self.r > 0 and not self.merged:
            q_k_v_list = output.chunk(dim=-1, chunks=3)

            q_k_v_lora_list = []
            if self.adapt_q:
                x_d = self.lora_dropout(x)
                x_a = self.lora_A_Q(x_d)
                x_b = self.lora_B_Q(x_a)[0]     # Q_lora_B will return two values and the second is None
                q_k_v_lora_list.append(
                    q_k_v_list[0] + x_b * self.scaling
                )
            else:
                q_k_v_lora_list.append(q_k_v_list[0])
            
            if self.adapt_k:
                x_d = self.lora_dropout(x)
                x_a = self.lora_A_K(x_d)
                x_b = self.lora_B_K(x_a)[0]
                q_k_v_lora_list.append(
                    q_k_v_list[1] + x_b * self.scaling
                )
            else:
                q_k_v_lora_list.append(q_k_v_list[1])
            
            if self.adapt_v:
                x_d = self.lora_dropout(x)
                x_a = self.lora_A_V(x_d)
                x_b = self.lora_B_V(x_a)[0]
                q_k_v_lora_list.append(
                    q_k_v_list[2] + x_b * self.scaling
                )
            else:
                q_k_v_lora_list.append(q_k_v_list[2])
            
            output = torch.cat(q_k_v_lora_list, dim=-1).contiguous()
        return output, output_bias

    def enable_input_require_grads(self):
        def make_inputs_require_grads(module, input, output):
            output.requires_grad_(True)
        self.model.embed_tokens.register_forward_hook(make_inputs_require_grads)

def mark_only_lora_as_trainable(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
        else:
            p.requires_grad = True
    def make_inputs_require_grads(module, input, output):
        output.requires_grad_(True)
    model.language_model.embedding.register_forward_hook(make_inputs_require_grads)


def lora_state_dict(full_state_dict, encoder, merge_weights = False) -> Dict[str, torch.Tensor]:
    # full_state_dict = model.state_dict()

    if merge_weights:
        for module in encoder.modules():
            if isinstance(module, LoRALayer):
                assert module.merged, "If you want to save the merged model, the Lora weights must have already been merged."
        return full_state_dict
    else:
        return {k: full_state_dict[k] for k in full_state_dict if 'lora_' in k}
