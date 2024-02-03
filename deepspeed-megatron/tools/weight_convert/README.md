






### 模型权重转换

从hugging face或官方github仓库转换而来的权重通常是完整权重。

- 基于完整权重进行多卡分布式训练，需要将完整权重转换为分布式权重。

- 基于训完的分布式权重进行单卡推理，需要将分布式权重转换为完整权重。
- 修改分布式策略训练，需要将权重转换为对应分布式权重。

本仓库提供了权重转换脚本，位于`deepspeed-megatron/tools/weight_convert`。
```shell
#完整权重转换为分布式权重
#示例脚本：ds_partition_ckpts.sh
python ./ds_partition_ckpts.py \
       --src_ckpt /{INPUT_PATH} \
       --dst_ckpt /{OUTPUT_PATH} \
       --tp  {TP_SIZE}

# 参数说明
src_ckpt: 原始完整权重保存路径
dst_ckpt: 拆分后权重保存路径
tp: 拆分后的tensor并行维度，默认为8
```
```shell
#分布式权重合并
#示例脚本：ds_merge_ckpts.sh
python ./ds_merge_ckpts.py \
       --src_ckpt /{INPUT_PATH} \
       --dst_ckpt /{OUTPUT_PATH} \

# 参数说明
src_ckpt: 拆分权重保存路径
dst_ckpt: 合并后权重保存路径
```




