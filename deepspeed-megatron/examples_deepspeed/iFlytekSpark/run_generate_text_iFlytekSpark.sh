#!/bin/bash
export TORCH_CUDA_ARCH_LIST=8.6+PTX
export CUDA_DEVICE_MAX_CONNECTIONS=1

batch_size=1
mp=1
nodes=1
gpus=1

###############################################################################
##以下参数需更换为本地路径
from_pretrained="/data/ckpt/ckpt-download"
tokenizer_file="/data/tokenizer/tokenizer"



predict_data="[为什么地球是独一无二的？,请介绍一下人工智能。]" #注意分隔符为英文格式逗号 ","

launch_cmd="ds --num_nodes $nodes --num_gpus $gpus"

num_layers=40
hidden_size=5120
ffn_hidden_size=14336
num_attn_heads=40
seq_length=32768
local_size=8192

top_k=1
top_p=0.0
temperature=1.0
repeat_penalty=1.2
num_repeat_penalty=0.1
predict_length=1024
# seq_length: 最大推理长度
# batch_size: 推理batch数
# local_size: sparse attention的局部长度
# 各类采样参数: top_k, top_p, temperature等
# from_pretrained: ckpt的加载路径
# tokenizer_file: tokenizer文件路径
# predict_data: 待推理问题，每个问题之间使用','分隔
# predict_length: 生成token长度

program_cmd="../../tools/run_iFlytekSpark_text_generation.py \
       --tensor-model-parallel-size $mp \
       --num-layers $num_layers \
       --hidden-size $hidden_size \
       --ffn-hidden-size $ffn_hidden_size \
       --num-attention-heads $num_attn_heads \
       --max-position-embeddings $seq_length \
       --apply-gated-linear-unit \
       --gate-gelu \
       --no-position-embedding \
       --use-rotary-position-embeddings \
       --make-vocab-size-divisible-by 1 \
       --tokenizer-type iFlytekSparkSentencePieceTokenizer \
       --bf16 \
       --mlp-type standard \
       --micro-batch-size $batch_size \
       --seq-length $seq_length \
       --localsize $local_size \
       --predict-length $predict_length \
       --vocab-file ${tokenizer_file} \
       --no-bias-gelu-fusion \
       --no-bias-dropout-fusion \
       --no-masked-softmax-fusion \
       --no-query-key-layer-scaling \
       --temperature $temperature \
       --top_k $top_k \
       --top_p $top_p \
       --repeat_penalty $repeat_penalty \
       --num_repeat_penalty $num_repeat_penalty \
       --log-interval 1 \
       --from-pretrained ${from_pretrained} \
       --predict-data ${predict_data}"

echo $launch_cmd $program_cmd
$launch_cmd $program_cmd &>> infer.log
