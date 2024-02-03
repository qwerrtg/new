# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Sample Generate iFlytekSpark"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))

import torch
import json
from itertools import takewhile, repeat

from megatron import get_args
from megatron import print_rank_0
from megatron.core import mpu
from megatron.checkpointing import load_checkpoint
from megatron.initialize import initialize_megatron
from megatron.model import iFlytekSparkModel
from megatron.training import get_model
from megatron.text_generation import generate_and_post_process
from megatron.arguments import core_transformer_config_from_args

def get_checkpoint_name(checkpoints_path,
                        pipeline_parallel=None,
                        tensor_rank=None, pipeline_rank=None):
    """Determine the directory name for this rank's checkpoint."""


    # Use both the tensor and pipeline MP rank.
    if pipeline_parallel is None:
        pipeline_parallel = (mpu.get_pipeline_model_parallel_world_size() > 1)
    if tensor_rank is None:
        tensor_rank = mpu.get_tensor_model_parallel_rank()
    if pipeline_rank is None:
        pipeline_rank = mpu.get_pipeline_model_parallel_rank()

    # Use both the tensor and pipeline MP rank. If using the distributed
    # optimizer, then the optimizer's path must additionally include the
    # data parallel rank.
    if not pipeline_parallel:
        common_path = os.path.join(checkpoints_path, 
                            f'mp_rank_{tensor_rank:02d}_model_states.pt')
    else:
        common_path = os.path.join(checkpoints_path, 
                        f'mp_rank_{tensor_rank:02d}_{pipeline_rank:03d}_model_states.pt')

    return common_path

def model_provider(pre_process=True, post_process=True):
    """Build the model."""
    print_rank_0('building iFlytekSpark model ...')
    args = get_args()
    config = core_transformer_config_from_args(args)
    model = iFlytekSparkModel(
        config,
        num_tokentypes=0,
        parallel_output=False,
        pre_process=pre_process,
        post_process=post_process,
        return_moe_loss=False
    )


    if args.from_pretrained is not None:
        assert os.path.exists(args.from_pretrained)
        ckpt_path = get_checkpoint_name(args.from_pretrained)
        print_rank_0('Loading from {} '.format(
                args.from_pretrained))

        state_dict = torch.load(ckpt_path, map_location=f"cuda:{torch.cuda.current_device()}")
        if 'module' in state_dict:
            state_dict = state_dict['module']
        model.load_state_dict(state_dict)
    return model

def str_replace(text, table):
    """replace str by substr in table"""
    for org, dst in table.items():
        text = text.replace(org, dst)
    return text


def fmt_prompt(text, fmt: None):
    # fmt: off
    text = str_replace(
        text,
        table={
            "\t": "<tab>",
            "\n": "<ret>",
            "\f": "<pag>"}
    )
    # fmt: on
    if fmt is None:
        return text
    return fmt % text

def input_split(text):
    assert text[0]=="[" and text[-1]=="]", 'Predict-data shoud be like [***]'
    text = text[1:-1]
    return text.split(",")

def add_text_generate_args(parser):
    group = parser.add_argument_group(title='text generation')

    group.add_argument("--temperature", type=float, default=0.5,
                       help='Sampling temperature.')
    group.add_argument("--top_p", type=float, default=0.0,
                       help='Top p sampling.')
    group.add_argument("--top_k", type=int, default=1,
                       help='Top k sampling.')
    group.add_argument("--repeat_penalty", type=float, default=1.2,
                       help='the fixed penalty value of repeated tokens.')
    group.add_argument("--num_repeat_penalty", type=float, default=0.1,
                       help='the penalty value for the number of repetitions of repeated tokens.')

    group.add_argument("--predict-length", type=int, default=1024,
                       help='Size of the output generated text.')
    group.add_argument("--min_len", type=int, default=1,
                       help='minimum generation length.')
    group.add_argument("--prompt-format", type=str, default="<User> %s<end><Bot> ",
                       help='Format of the prompt.')

    group.add_argument("--predict-data", type=str, default="[Who are you?]",
                       help='Input of prompt.')
    group.add_argument("--json-input-path", type=str, default=None,help='path of input json.')
    group.add_argument("--json-output-path",type=str, default=None,help='path of output json.')
    return parser

def json_load(file_path):
    prompts = []
    with open(file_path,"r") as fp:
        for line in takewhile(lambda x: x, (fp.readline() for _ in repeat(None))):
            try:
                item = json.loads(line.strip())
                assert 'input' in item, f'miss key=input in {item}'
            except Exception as e:
                print_rank_0(f"parse jsonl error: {e}, source line: {line}")
                continue
            prompts.append(item)
    return prompts


if __name__ == "__main__":
    initialize_megatron(extra_args_provider=add_text_generate_args,
                        args_defaults={'tokenizer_type': 'iFlytekSparkSentencePieceTokenizer',
                                       'no_load_rng': True,
                                       'no_load_optim': True})

    args = get_args()
    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for text generation.")
        exit()
    # Set up model and load checkpoint
    model = get_model(model_provider, wrap_with_ddp=False)

    if args.load is not None:
        _ = load_checkpoint(model, None, None)

    assert len(model) == 1, "Above condition should have caught this"
    model = model[0]
    model.eval()
    fout = None
    if args.json_input_path is not None:
        predict_data = json_load(args.json_input_path)
        predict_data_input = [x["input"] for x in predict_data]
        predict_data_target = [x["target"] for x in predict_data]
        outfile = args.json_output_path
        if mpu.get_tensor_model_parallel_rank() == 0:
            fout = open(outfile, "a")
    else:
        predict_data = input_split(args.predict_data)
        predict_data_input = predict_data

    for i in range(0, len(predict_data_input), args.micro_batch_size):
        curr_batch = predict_data_input[i : i + args.micro_batch_size]
        prompts = [fmt_prompt(s,args.prompt_format) for s in curr_batch]


        prompts_plus_generations, prompts_plus_generations_segments, output_log_probs, tokens = generate_and_post_process(
            model=model,
            # prompts=[" <User> 孩子2岁，发烧了，一直降不下来，布洛芬一天能吃几次啊？<end><Bot>"],
            prompts=prompts,
            tokens_to_generate=args.predict_length,
            return_output_log_probs=False,
            top_k_sampling=args.top_k,
            top_p_sampling=args.top_p,
            temperature=args.temperature,
            add_BOS=False
        )
        if mpu.get_tensor_model_parallel_rank() == 0:    
            for idx, prompt_generation in enumerate(prompts_plus_generations):
                print_rank_0(f"prompt: {curr_batch[idx]}")
                print_rank_0(f"generate: {prompt_generation[len(prompts[idx]):]}" + "\n")
                if args.json_input_path is not None:
                    curr_batch_taget = predict_data_target[i : i + args.micro_batch_size]
                    json_output = {}
                    json_output["input"] = curr_batch[idx]
                    json_output["target"] = curr_batch_taget[idx]
                    json_output["generate"] = prompt_generation[len(prompts[idx]):]
                    json_output = json.dumps(json_output, ensure_ascii=False)
                    fout.write(f"{json_output}\n")
                    fout.flush()

    if fout is not None:
        fout.close()
        fout = None