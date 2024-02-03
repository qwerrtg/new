import re
import os
import sys 
import torch
from typing import List, Tuple, Optional, Callable, Union
from argparse import ArgumentParser
from collections import defaultdict, OrderedDict
from utils import split_tensor_to_local, embed_meta, transf_layers



def partition_ckpts(unpack_ckpt, save_dir, TP) -> None:
    
    if 'module' in unpack_ckpt:
        unpack_ckpt = unpack_ckpt['module']
    ckpt = unpack_ckpt['language_model']['encoder']

    param_metas = transf_layers
    transformer_layers = list()
    for k, p in ckpt.items():
        transformer_layers.append(k)
    print(f"=> Start partition, tp_size={TP}")

    state = {"language_model": {
        "embedding": {"word_embeddings":{}},
        "encoder": {}
    }}

    for tp_rank in range(TP):
        local_state_dict = OrderedDict()
        for layer in transformer_layers:
            if "lora" not in layer:

                meta = param_metas[layer]
                param = ckpt[layer]
                if (
                    meta is not None and \
                    meta['partition_dim'] != -1
                ):
                    local_param = split_tensor_to_local(
                        tensor=param,
                        partition_dim=meta['partition_dim'],
                        partition_stride=meta['partition_stride'],
                        world_size=TP,
                        cur_rank=tp_rank,
                    )
                else:
                    local_param = param

                local_state_dict[layer] = local_param

        embed_param = unpack_ckpt["language_model"]["embedding"]["word_embeddings"]["weight"]
        local_embed_param = split_tensor_to_local(
            tensor=embed_param,
            partition_dim=embed_meta['partition_dim'],
            partition_stride=embed_meta['partition_stride'],
            world_size=TP,
            cur_rank=tp_rank,
        )
        state["language_model"]["embedding"]["word_embeddings"]["weight"] = local_embed_param
        state["language_model"]["encoder"] = local_state_dict

        outname = f'mp_rank_{tp_rank:02d}_model_states.pt'
        path = os.path.join(save_dir, outname)

        print(f'Saving TP={tp_rank} checkpoint ...')
    
        torch.save({"module":state}, path)
        del local_state_dict

    print(f'=> Success! :-)')


def parse_args():
    parser = ArgumentParser(
        usage='python /tools/ds_partition_ckpts.py --src_ckpt ./checkpoint-last.pt --dst_ckpt ./checkpoint-shards --tp 8',
        description='Partition merged checkpoint to sharding files.'
    )
    parser.add_argument('-src_ckpt', '--src_ckpt', type=str, required=True)
    parser.add_argument('-dst_ckpt', '--dst_ckpt', type=str, required=True)
    parser.add_argument('-tp', '--tp', type=int, default=8,)
    args = parser.parse_args()
    assert os.path.exists(args.src_ckpt), f"File <{args.src_ckpt}> not found!"
    os.makedirs(args.dst_ckpt, exist_ok=True)
    return args


if __name__ ==  "__main__":
    args = parse_args()

    unpack_ckpt = torch.load(args.src_ckpt, map_location='cpu')
    print(f'Load Success! :-)')
    partition_ckpts(unpack_ckpt, args.dst_ckpt, args.tp)