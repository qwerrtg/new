import os
import logging
import glob
import re
import shutil
from copy import deepcopy
from typing import List, Tuple, Dict, Union
from collections import OrderedDict
from argparse import ArgumentParser

import torch
from utils import (
    split_tensor_to_local as _split_tensor_to_local,
    gather_tensors_to_global as _gather_tensors_to_global
)
from utils import (
    embed_meta,
    transf_layers
)
logger = logging.getLogger('tools')
logger.setLevel(logging.INFO)


_MODEL_KEY_IN_CKPT = 'language_model'
_OPTIM_KEY_IN_CKPT = 'last_optimizer_state'


def _paths_string(paths):
    return '\n'.join(paths)


def load_checkpoint(path: str) -> OrderedDict:
    ckpt = torch.load(path, map_location='cpu')
    if 'module' in ckpt:
        ckpt = ckpt['module']
    return ckpt




def _validate_tensors_consistent(tensors: List[torch.Tensor], name: str) -> None:
    max_diff = None
    base_tensor = tensors[0]
    for tensor in tensors[1:]:
        is_inconsistent = torch.all(tensor == base_tensor)
        if not is_inconsistent:
            diff = (tensor - base_tensor).abs().max()
            max_diff = diff if max_diff is None else max(max_diff, diff)
    if max_diff is not None:
        print(
            f"The data of tensors named with <{name}> are inconsistent, max abs different={max_diff}, " \
            f"Maybe you passed in the wrong state dicts, or your model was trained incorrectly, causing " \
            f"the parameters or states to be inconsistent!" \
        )




def local_tensors_to_global(params: List[torch.Tensor],
                            parallel_attrs: List[Dict], param_name) -> torch.Tensor:
    parallel_attr = sorted(parallel_attrs, key=lambda x: x['parallel_index'])
    sorted_params = [params[i['parallel_index']] for i in parallel_attr]
    parallel_size = parallel_attr[0]['parallel_size']
    partition_dim = parallel_attr[0]['partition_dim']
    partition_stride = parallel_attr[0]['partition_stride']
    
    replica_param = _gather_tensors_to_global(
        tensors=sorted_params, partition_dim=partition_dim,
        partition_stride=partition_stride, world_size=parallel_size)
    return replica_param


def merge_tensor_parallel_local_model_state(local_state_dicts) -> OrderedDict:
    tensor_parallel_size = 1
    local_metas = list()
    local_embed_metas = list()
    param_names = None
    for state_dict_index, all_state_dict in enumerate(local_state_dicts):
        state_dict = all_state_dict['encoder']

        curr_param_names = tuple(state_dict.keys())
        if param_names is None:
            param_names = curr_param_names

        assert param_names == curr_param_names, \
            f"key miss match: \n{param_names}\nVS\n {curr_param_names}"

        cur_metas = deepcopy(transf_layers)
        embed_cur_metas = deepcopy(embed_meta)
        for name in cur_metas.keys():
            if cur_metas[name] is not None:
                cur_metas[name]['parallel_size'] = len(local_state_dicts)
                cur_metas[name]['parallel_index'] = state_dict_index
        
        embed_cur_metas['parallel_size'] = len(local_state_dicts)
        embed_cur_metas['parallel_index'] = state_dict_index
        local_metas.append(cur_metas)
        local_embed_metas.append(embed_cur_metas)

    global_state_dict = {
        "embedding": {"word_embeddings":{}},
        "encoder": {}
    }

    for param_name in param_names:
        # if param_name == _EXTRA_STATE:
        #     continue
        local_params = [state_dict['encoder'][param_name] for state_dict in local_state_dicts]
        
        param_metas = [params_meta[param_name] for params_meta in local_metas]
        # check tensor parallel attrs
        is_tensor_parallel = param_metas[0] is not None and param_metas[0]['partition_dim'] != -1
        # replace lobal param to global param
        replica_param = local_params[0]
        if not is_tensor_parallel:
            _validate_tensors_consistent(local_params, param_name)
        else:
            replica_param = local_tensors_to_global(local_params, param_metas, param_name)
        global_state_dict["encoder"][param_name] = replica_param
    
    local_embed_params = [state_dict['embedding']['word_embeddings']["weight"] for state_dict in local_state_dicts]
    replica_embed_param = local_tensors_to_global(local_embed_params, local_embed_metas ,"word_embeddings")
    global_state_dict["embedding"]["word_embeddings"]["weight"] = replica_embed_param

    return global_state_dict






def sort_ckpt_paths(paths: List[str]):
    pp_size = 1
    tp_size = 1
    temp = list()
    for path in paths:
        ckpt_name = os.path.split(path)[-1]
        match_ret = re.match(r"^mp_rank_0([0-9]+)", ckpt_name)
        if match_ret is not None and len(match_ret.groups()) == 1:
            pp_rank = 0
            tp_rank = int(match_ret.group(1))
            pp_size = pp_rank + 1 if pp_size is None else max(pp_rank + 1, pp_size)
            tp_size = tp_rank + 1 if tp_size is None else max(tp_rank + 1, tp_size)
            temp.append((path, (pp_rank, tp_rank)))

    temp = sorted(temp, key=lambda x: x[1])
    sorted_paths = list()
    partitions_paths = [[None for j in range(tp_size)] for i in range(pp_size)]

    for path, (pp_rank, tp_rank) in temp:
        sorted_paths.append(path)
        partitions_paths[pp_rank][tp_rank] = path

    return sorted_paths, partitions_paths
    
    # only_model: bool=True,
    # return_state_dict: bool=False,
def merge_state_dict(
    ckpt_dir: str,
    save_dir: str,
    verbose: bool=True,
):
    """
    ckpt_dir: state_dict path
    return_state_dict:
        False: save state_dict and return None
        True: don't save but return state_dict
    verbose: print merge state
    """
    # include_keys = [_MODEL_KEY_IN_CKPT]
    # if not only_model:
    #     include_keys.append(_OPTIM_KEY_IN_CKPT)

    # abs_ckpt_dir = os.path.abspath(ckpt_dir)
    assert os.path.isdir(ckpt_dir), \
        f"Dir path <{ckpt_dir}> not found or isn't a folder!"

    ckpt_paths = glob.glob(os.path.join(ckpt_dir, "*"))
    if len(ckpt_paths) == 0:
        raise FileNotFoundError(f"Folder <{ckpt_dir}> is empty")
    
    ckpt_paths, patition_paths = sort_ckpt_paths(ckpt_paths)
    if verbose:
        print(f"=> Find {len(ckpt_paths)} checkpoint files: \n{_paths_string(ckpt_paths)}")

    save_path = os.path.join(os.path.abspath(save_dir), f'mp_rank_00_model_states.pt')
    # if verbose:
    #     print(f"=> Dest checkpoint will save to: {save_path}")
    if len(ckpt_paths) == 1:
        if verbose:
            print(f'=> Just a single checkpoint found in the dir, copy derectly!')
        # if verbose:
        #     print(f'=> Save {save_path} ...')
        state_dict = load_checkpoint(ckpt_paths[1])
        model_state = state_dict[_MODEL_KEY_IN_CKPT]
        if verbose:
            print(f'=> Success! :-)')
        return model_state

    pp_size = len(patition_paths)
    tp_size = len(patition_paths[0])
    partition_model_state = list()
    for pipeline_state in range(pp_size):
        curr_paths = patition_paths[pipeline_state]
        if verbose:
            print(f'=> Loading checkpoints: \n{_paths_string(curr_paths)}')
        curr_state_dicts = [load_checkpoint(ckpt_path) for ckpt_path in curr_paths]

        curr_replica_model_state = merge_tensor_parallel_local_model_state(
                                        [d[_MODEL_KEY_IN_CKPT] for d in curr_state_dicts]
                                    )
        partition_model_state.append(curr_replica_model_state)
        del curr_state_dicts

    base_state = torch.load(ckpt_paths[0], map_location='cpu')
    for key_ckpt in list(base_state.keys()):
        if key_ckpt!='module':
            del base_state[key_ckpt]

    base_state['module'][_MODEL_KEY_IN_CKPT] = partition_model_state[0]

    if verbose:
        print(f'=> Save {save_path} ...')
    # if not return_state_dict:
    torch.save(base_state, save_path)
    if verbose:
        print(f'=> Success! :-)')
    # else:
    #     return base_state

def parse_args():
    parser = ArgumentParser(
        usage='python ./ds_merge_ckpts.py --src_ckpt ./checkpoint-shards-path --dst_ckpt ./checkpoint-merge',
        description='Partition merged checkpoint to sharding files.'
    )
    parser.add_argument('-src_ckpt', '--src_ckpt', type=str, required=True)
    parser.add_argument('-dst_ckpt', '--dst_ckpt', type=str, required=True)
    args = parser.parse_args()
    assert os.path.exists(args.src_ckpt), f"File <{args.src_ckpt}> not found!"
    os.makedirs(args.dst_ckpt, exist_ok=True)
    return args



if __name__ == "__main__":
    args = parse_args()

    merge_state_dict(args.src_ckpt, args.dst_ckpt, True)