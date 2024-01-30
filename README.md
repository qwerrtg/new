# <center>iFlytekSpark-13B</center>




# 模型介绍

讯飞星火开源-13B（iFlytekSpark-13B）拥有130亿参数，在经过累计超过3万亿以上tokens海量高质量数据集上进行预训练，然后在精调的多元化对齐数据上进行微调得到。iFlytekSpark-13B在多个标准评估中展现出了卓越的性能，其表现优于同参数量级的开源模型.
iFlytekSpark-13B不仅具备通用任务处理能力如聊天、问答、文本提取和分类等，还具备数据分析和代码生成等生产力功能。我们特别在学习辅助、数学、推理等领域进行了深度优化，大幅提升模型的实用性和易用性。详细的评测结果见下面评测部分。

此次开源推动人工智能和机器学习领域的开源协作，并在全球范围内促进技术革新。欢迎大家积极利用和贡献于iFlytekSpark-13B，共同促进人工智能的发展。



## 模型结构

在iFlytekSpark-13B中，我们使用Rotary Embedding作为位置编码方法，GELU作为激活函数，其中layer_num为40，head_num为40，hidden_size为5120，ffn_hidden_size为28672。

## Benchmark 结果

我们在八个具有挑战性的中英文测试集上对模型进行性能评估。其中chat模型采用0-shot进行测试，base模型在C-EVAL，MMLU，CMMLU，FinanceIQ测试集上采用5-shot进行测试，其余测试集采用0-shot进行测试。

- C-EVAL：C-Eval 是一个全面的中文基础模型评估套件，涵盖了52个不同的学科和四个难度级别，验证集包括1346个选择题，测试集包含12342个选择题。本项目采用C-Eval验证集进行测试。
- MMLU：MMLU 是一个庞大的多任务数据集，由各种学科的多项选择题组成。其中包括57个任务，涵盖了人文学科、社会科学、自然科学和其他对某些人学习很重要的领域。
- CMMLU：CMMLU 是一个综合性的中文评估基准，涵盖了从基础学科到高级专业水平的67个主题。涵盖了自然科学、人文科学和社会科学等领域。
- AGIEVAL：AGIEval 是一个专门为评估基础模型在以人类为中心的标准化考试（如大学入学考试、法学院入学考试、数学竞赛和律师资格考试）的语境中而设计的基准测试。
- ARC：包含了ARC-E和ARC-C，它们分别是ARC数据集中的简单集和挑战集，分别有5197 和2590 个问题。这些问题是仅文本的英语语言考试问题，跨越了多个年级水平。
- GaoKao：GaoKao收集了从 2010 年到 2022 年的高考试题，包括 1781 道客观题和 1030 道主观题。本项目报告结果为GaoKao中客观题结果。
- FinanceIQ：FinanceIQ 是一个专注于金融领域的中文评估数据集，涵盖了10个金融大类及36个金融小类，总计7173个单项选择题。

|                       | C_EVAL | MMLU  | CMMLU | AGIEVAL | ARC_E | ARC_C | GaoKao | FinanceIQ | 平均  |
| :-------------------: | ------ | ----- | :---: | :-----: | :---: | :---: | :----: | :-------: | :---: |
| iFlytekSpark-13B-base | 70.88  | 58.76 | 70.01 |  50.44  | 84.78 | 71.16 | 56.42  |   60.21   | 65.33 |
| iFlytekSpark-13B-chat | 82.54  | 63.02 | 75.69 |  56.96  | 89.47 | 77.34 | 67.49  |   65.48   | 72.25 |

# 

# NPU版本仓库介绍

`iFlytekSpark` 基于 [MindFormers](https://gitee.com/mindspore/mindformers/tree/r1.0/research/iflytekspark) 套件实现，主要涉及的文件有：

1. 模型具体实现：`research/iflytekspark`

```
    ├── iflytekspark_tokenizer.py          # tokenizer
    ├── iflytekspark_config.py             # 模型配置基类
    ├── iflytekspark_layers.py             # 模型基本模块实现
    ├── iflytekspark_model.py              # 模型实现
    ├── iflytekspark_infer.py              # 在线推理脚本
    ├── iflytekspark_streamer.py           # 流式推理实现
    ├── iflytekspark_sampler.py            # 在线推理后处理采样实现
    ├── iflytekspark_text_generator.py     # 在线推理API
    ├── repetition_processor.py      
    └── optim.py                           # 优化器实现 
```

2. 模型配置文件：`research/iflytekspark`

```
    ├── run_iflytekspark_13b_pretrain_800T_A2_64G.yaml         # 13B预训练配置（适用Atlas 800T A2）
    ├── run_iflytekspark_13b_sft_800T_A2_64G.yaml              # 13B全量微调配置（适用Atlas 800T A2）
    ├── run_iflytekspark_13b_lora_800T_A2_64G.yaml             # 13BLora微调配置（适用Atlas 800T A2）
    ├── run_iflytekspark_13b_infer_800T_A2_64G.yaml            # 13B全量在线推理配置（适用Atlas 800T A2）
    ├── run_iflytekspark_13b_pretrain_800_32G.yaml             # 13B预训练配置（适用Atlas 800）
    ├── run_iflytekspark_13b_sft_800_32G.yaml                  # 13B全量微调配置（适用Atlas 800）
    ├── run_iflytekspark_13b_lora_800_32G.yaml                 # 13BLora微调配置（适用Atlas 800）
    └── run_iflytekspark_13b_infer_800_32G.yaml                # 13B全量在线推理配置（适用Atlas 800）
```

3. 数据处理脚本、权重处理脚本及任务启动脚本：`research/iflytekspark`

```
    ├── pretrain_data_preprocess.py      # 预训练数据处理脚本
    ├── sft_data_preprocess.py           # 微调数据处理脚本
    ├── weight_convert.py                # mindspore BF16格式权重转换脚本
    └── run_iflytekspark.py              # 高阶接口使用脚本
```

4. 执行代码脚本

```
run_iflytekspark.py
```

接受以下参数，通过Shell脚本传入参数时，

传入参数优先级高于配置文件中对应的配置项

- task：任务类型，默认值：`text_generation`。
- config：配置文件路径，必须指定。
- run_mode：运行模式，包括`train`，`finetune`，`eval`，`predict`和`export`，默认值：`train`。
- use_parallel：是否使能并行，默认值：`False`。
- load_checkpoint：加载checkpoint路径，默认值：`None`。
- auto_trans_ckpt：自动根据当前并行策略进行checkpoint切分，默认值：`None`。
- resume：断点续训，默认值：`False`。
- train_dataset：训练数据集路径，默认值：`None`。
- eval_dataset：验证数据集路径，默认值：`None`。
- predict_data：推理数据集路径，默认值：`None`。
- predict_length：模型的最大推理长度，默认值：`512`。
- predict_batch：模型推理的batch数，默认值：`1`。
- optimizer_parallel：是否使能优化器并行，默认值：`False`。
- device_id：指定设备id，仅在非并行模式下生效，默认值：`0`。
- prompt：推理使用的语料模版，默认值：`None`，表示不使用模板。
- tokenizer_file：tokenizer文件路径，进行在线推理时需要指定。
- mindir_save_dir：导出mindir模型文件路径，默认当前路径。
- streamer：是否使能流式推理，默认值：`False`。

## **前期准备**

### **[mindformers安装](https://gitee.com/mindspore/mindformers)**

### **环境要求**

- 硬件：Atlas 800/Atlas 800T A2
- MindSpore：2.2.11
- MindFormers版本：r1.0
- 硬件支持矩阵如下

| **模型**         | **硬件**      | **预训练** | **全量微调** | **lora微调** | **推理** |
| ---------------- | ------------- | ---------- | ------------ | ------------ | -------- |
| iFlytekSpark-13b | Atlas 800     | ≥2节点     | ≥2节点       | 单节点       | ≥2卡     |
| iFlytekSpark-13b | Atlas 800T A2 | 单节点     | 单节点       | 单节点       | 单卡     |



### **[acctransformer安装](https://gitee.com/mindspore/acctransformer/tree/fa1_for_ms2.2.11/)**

在Atlas 800机型上进行iFlytekSpark模型的训练、微调、推理，需要安装acctransformer套件使能FlashAttention。执行以下命令克隆源码到本地：

```
git clone -b fa1_for_ms2.2.11 https://gitee.com/mindspore/acctransformer.git
```

安装方法如下：

1. 直接克隆源码使用，使用源码方式调用时设置PYTHONPATH。

```
export PYTHONPATH=/yourcodepath/acctransformer/train:$PYTHONPATH
```

2. 安装whl包使用。

```
cd train
python setup.py install
```

或

```
cd train
bash build.sh
pip install dist/acctransformer-1.0.0-py3-none-any.whl
```

### **RANK_TABLE_FILE准备**

- **单节点**

运行`mindformers/tools/hccl_tools.py`，生成`RANK_TABLE_FILE`文件

```
# 运行如下命令，生成当前机器的RANK_TABLE_FILE的json文件
python ./mindformers/tools/hccl_tools.py --device_num "[0,8)"
```

**注：若使用ModelArts的notebook环境，可从** `**/user/config/jobstart_hccl.json**` **路径下直接获取rank table，无需手动生成**

`RANK_TABLE_FILE`` 单机8卡参考样例:

```
{
    "version": "1.0",
    "server_count": "1",
    "server_list": [
        {
            "server_id": "xx.xx.xx.xx",
            "device": [
                {"device_id": "0","device_ip": "192.1.27.6","rank_id": "0"},
                {"device_id": "1","device_ip": "192.2.27.6","rank_id": "1"},
                {"device_id": "2","device_ip": "192.3.27.6","rank_id": "2"},
                {"device_id": "3","device_ip": "192.4.27.6","rank_id": "3"},
                {"device_id": "4","device_ip": "192.1.27.7","rank_id": "4"},
                {"device_id": "5","device_ip": "192.2.27.7","rank_id": "5"},
                {"device_id": "6","device_ip": "192.3.27.7","rank_id": "6"},
                {"device_id": "7","device_ip": "192.4.27.7","rank_id": "7"}],
             "host_nic_ip": "reserve"
        }
    ],
    "status": "completed"
}
```

- **多节点**以2机16卡为例：

1. 在每个机器上运行`mindformers/tools/hccl_tools.py`，生成各自的`RANK_TABLE_FILE`文件。

```
# 运行如下命令，生成当前机器的RANK_TABLE_FILE的json文件
python ./mindformers/tools/hccl_tools.py --device_num "[0,8)" --server_ip xx.xx.xx.xx
```

**注：需要根据机器的ip地址指定 --server_ip，避免由于不同机器server_ip不同，导致多节点间通信失败。**

2. 将不同机器的`RANK_TABLE_FILE`文件全部拷贝到同一台机器上，运行`mindformers/tools/merge_hccl.py`合并`RANK_TABLE_FILE`文件

```
# 运行如下命令，合并每个机器的RANK_TABLE_FILE文件。
python ./mindformers/tools/merge_hccl.py hccl*.json
```

3. 将合并后的`RANK_TABLE_FILE`文件拷贝到所有机器中，保证不同机器上的`RANK_TABLE_FILE`相同。

`RANK_TABLE_FILE` 双机16卡参考样例:

```
{
    "version": "1.0",
    "server_count": "2",
    "server_list": [
        {
            "server_id": "xx.xx.xx.xx",
            "device": [
                {
                    "device_id": "0", "device_ip": "192.168.0.0", "rank_id": "0"
                },
                {
                    "device_id": "1", "device_ip": "192.168.1.0", "rank_id": "1"
                },
                {
                    "device_id": "2", "device_ip": "192.168.2.0", "rank_id": "2"
                },
                {
                    "device_id": "3", "device_ip": "192.168.3.0", "rank_id": "3"
                },
                {
                    "device_id": "4", "device_ip": "192.168.0.1", "rank_id": "4"
                },
                {
                    "device_id": "5", "device_ip": "192.168.1.1", "rank_id": "5"
                },
                {
                    "device_id": "6", "device_ip": "192.168.2.1", "rank_id": "6"
                },
                {
                    "device_id": "7", "device_ip": "192.168.3.1", "rank_id": "7"
                }
            ],
            "host_nic_ip": "reserve"
        },
        {
            "server_id": "xx.xx.xx.xx",
            "device": [
                {
                    "device_id": "0", "device_ip": "192.168.0.1", "rank_id": "8"
                },
                {
                    "device_id": "1", "device_ip": "192.168.1.1", "rank_id": "9"
                },
                {
                    "device_id": "2", "device_ip": "192.168.2.1", "rank_id": "10"
                },
                {
                    "device_id": "3", "device_ip": "192.168.3.1", "rank_id": "11"
                },
                {
                    "device_id": "4", "device_ip": "192.168.0.2", "rank_id": "12"
                },
                {
                    "device_id": "5", "device_ip": "192.168.1.2", "rank_id": "13"
                },
                {
                    "device_id": "6", "device_ip": "192.168.2.2", "rank_id": "14"
                },
                {
                    "device_id": "7", "device_ip": "192.168.3.2", "rank_id": "15"
                }
            ],
            "host_nic_ip": "reserve"
        }
    ],
    "status": "completed"
}
```

**注：多机多卡获取**`**RANK_TABLE_FILE**`**步骤同2机16卡。**

### **数据集准备**

本章节以alpaca_gpt4数据集为例，介绍如何使用本仓提供的脚本制作数据集，用于对 iFlytekSpark-13B 模型进行预训练和微调。alpaca_gpt4数据集下载链接如下：

- [alpaca_gpt4_data](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/tree/main/data)

执行`pretrain_data_preprocess.py`，进行预训练数据预处理、Mindrecord数据生成，将带有prompt模板的数据转换为mindrecord格式。执行`sft_data_preprocess.py`，进行微调数据预处理、Mindrecord数据生成，将带有prompt模板的数据转换为mindrecord格式。

```
# 预训练
python ./research/iflytekspark/pretrain_data_preprocess.py \
--tokenizer /{TOKENIZER_PATH} \
--raw_data_path /{RAW_DATA_PATH} \
--output_filename /{OUTPUT_FILE_PATH} \
--seq_length SEQ_LENGTH

# SFT&Lora
python ./research/iflytekspark/sft_data_preprocess.py \
--tokenizer /{TOKENIZER_PATH} \
--raw_data_path /{RAW_DATA_PATH} \
--output_filename /{OUTPUT_FILE_PATH} \
--seq_length SEQ_LENGTH
--pad_id PAD_ID

# 参数说明
tokenizer: Tokenizer文件路径，指定至含有 .model 文件的文件夹
raw_data_path: 原始文本数据文件路径
output_filename: 数据集Mindrecord保存文件名，需指定到以.mindrecord结尾的文件名。可选参数，默认值为dataset.mindrecord
seq_length: 数据集单个样本句长，默认值：32768
pad_id: SFT数据集用于padding到指定句长的padding token id，默认值：0
```
seq_length: 默认值是32768，SFT数据类型需要修改为：32769

### **模型权重准备**

本仓提供支持MindSpore框架的预训练权重、微调权重和词表文件用于训练/微调/推理。

预训练权重：

- [iFlytekSpark_13b_base_fp32](https://gitee.com/link?target=https%3A%2F%2Fascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com%2FMindFormers%2Fiflytekspark%2Fiflytekspark_13b_base_fp32.ckpt)
- [iFlytekSpark_13b_base_bf16](https://gitee.com/link?target=https%3A%2F%2Fascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com%2FMindFormers%2Fiflytekspark%2Fiflytekspark_13b_base_bf16.ckpt)
- [iFlytekSpark_13b_base_fp16](https://gitee.com/link?target=https%3A%2F%2Fascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com%2FMindFormers%2Fiflytekspark%2Fiflytekspark_13b_base_fp16.ckpt)

微调权重：

- [iFlytekSpark_13b_chat_fp32](https://gitee.com/link?target=https%3A%2F%2Fascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com%2FMindFormers%2Fiflytekspark%2Fiflytekspark_13b_chat_fp32.ckpt)
- [iFlytekSpark_13b_chat_bf16](https://gitee.com/link?target=https%3A%2F%2Fascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com%2FMindFormers%2Fiflytekspark%2Fiflytekspark_13b_chat_bf16.ckpt)
- [iFlytekSpark_13b_chat_fp16](https://gitee.com/link?target=https%3A%2F%2Fascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com%2FMindFormers%2Fiflytekspark%2Fiflytekspark_13b_chat_fp16.ckpt)

**注**：本仓开源的权重包含Float16和BFloat16两种格式。`/research/iflytekspark/weight_convert.py`脚本用于将已有MindSpore权重进行数据类型的转换。

```
python ./research/iflytekspark/weight_convert.py \
--src_ckpt /{ORIGIN_CKPT} \
--dst_ckpt /{TARGET_CKPT} \
--dtype \
--embed_bf16         # (optional)
--layernorm_bf16     # (optional)

# 参数说明
src_ckpt: 原始MindSpore权重保存路径。
dst_ckpt: 转换后Bfloat16数据类型权重保存路径。
dtype: 转换后权重的数据类型（embedding、layernorm除外），支持float16、float32和bfloat16，默认bfloat16。
embed_bf16: embedding层采用Bfloat16数据类型计算，当前版本不支持。默认不开启。
layernorm_bf16: layernorm层采用Bfloat16数据类型计算。默认不开启。
```

例：Bfloat16训练权重转换为Float16数据格式

```
python ./research/iflytekspark/weight_convert.py \
--src_ckpt /{ORIGIN_CKPT} \
--dst_ckpt /{TARGET_CKPT} \
--dtype float16
```

### **[模型权重转换](https://gitee.com/mindspore/mindformers/blob/dev/docs/feature_cards/Transform_Ckpt.md)**

本仓提供的权重下载链接是完整权重。

- 基于完整权重进行多卡分布式训练，需要将完整权重转换为分布式权重。
- 基于训完的分布式权重进行单卡推理，需要将分布式权重转换为完整权重。
- 修改分布式策略训练，需要将权重转换为对应分布式权重。

Mindformer支持权重自动转换，详细教程请参考[权重转换文档](https://gitee.com/kerrykou/mindformers/blob/r1.0_iflytekspark/research/iflytekspark/iflytekspark.md#%E6%A8%A1%E5%9E%8B%E6%9D%83%E9%87%8D%E5%87%86%E5%A4%87)。


## **预训练**

请参照[数据集准备](#数据集准备)章节获取mindrecord格式的数据集，参照[模型权重准备](#模型权重准备)章节获取iFlytekSpark-13B权重。

- **单机训练**

iFlytekSpark-13B用于预训练，seq_length默认为32768，分布式预训练在Atlas 800T A2上训练，单节点8卡即可启动。Atlas 800T A2上默认使用BFloat16类型训练。本仓给出了默认配置文件`run_iflytekspark_13b_pretrain_800T_A2_64G.yaml`。

**步骤**

1. 多卡运行需要 RANK_TABLE_FILE,  请参照[RANK_TABLE_FILE准备](#rank_table_file准备)-单节点章节生成对应文件。
2. 修改模型对应的配置, 在 `run_iflytekspark_13b_pretrain_800T_A2_64G.yaml`中, 用户可以自行修改模型配置、训练相关参数。

```
oad_checkpoint: 'model_dir'    # 使用完整权重，权重按照`model_dir/rank_0/xxx.ckpt`格式存放
auto_trans_ckpt: True           # 打开自动权重转换
use_parallel: True
run_mode: 'train'
...
# 数据集配置
train_dataset: &train_dataset
  data_loader:
    type: MindDataset
    dataset_dir: "dataset_dir" # 配置训练数据集文件夹路径
    shuffle: False
  input_columns: ["input_ids"]
...
# 并行训练策略配置
parallel_config:
  data_parallel: 1
  model_parallel: 8
  pipeline_stage: 1
  optimizer_shard: True
  micro_batch_num: 1
  vocab_emb_dp: False
  gradient_aggregation_group: 4
```

1. 启动运行脚本, 进行单节点8卡分布式运行。

```
cd mindformers/research
bash run_singlenode.sh \
"python iflytekspark/run_iflytekspark.py \
--config iflytekspark/run_iflytekfpark_13b_pretrain_800T_A2_64G.yaml \
--use_parallel True \
--run_mode train \
--train_data dataset_dir" \
RANK_TABLE_FILE [0,8] 8
```

**注**：`run_iflytekspark.py`启动脚本支持的参数列表参考[仓库介绍](#NPU版本仓库介绍)-数据处理脚本、权重处理脚本及任务启动脚本章节。

- **多机训练**

iFlytekSpark-13B用于预训练，seq_length默认为32768，分布式预训练在Atlas 800上训练，需要2节点16卡启动。Atlas 800上默认使用Float16类型训练，需要开启序列并行（配置文件中设置`seq_parallel=True`）。本仓给出了默认配置文件`run_iflytekspark_13b_pretrain_800_32G.yaml`。

**步骤**

1. 多卡运行需要 RANK_TABLE_FILE,  请参照[RANK_TABLE_FILE准备](#rank_table_file准备)-多节点章节生成对应文件。
2. 修改模型对应的配置, 在 `run_iflytekspark_13b_pretrain_800_32G.yaml`中, 用户可以自行修改模型配置、训练相关参数。

```
oad_checkpoint: 'model_dir'    # 使用完整权重，权重按照`model_dir/rank_0/xxx.ckpt`格式存放
auto_trans_ckpt: True           # 打开自动权重转换
use_parallel: True
run_mode: 'train'
...
# 数据集配置
train_dataset: &train_dataset
  data_loader:
    type: MindDataset
    dataset_dir: "dataset_dir" # 配置训练数据集文件夹路径
    shuffle: False
  input_columns: ["input_ids"]
...
# 并行训练策略配置
parallel_config:
  data_parallel: 2
  model_parallel: 8
  pipeline_stage: 1
  optimizer_shard: True
  micro_batch_num: 1
  vocab_emb_dp: False
  gradient_aggregation_group: 4
```

3. 启动运行脚本, 进行2节点16卡分布式运行。

```
# node 1
cd mindformers/research
bash run_multinode.sh \
"python iflytekspark/run_iflytekspark.py \
--config iflytekspark/run_iflytekspark_13b_pretrain_800_32G.yaml \
--use_parallel True \
--run_mode train \
--train_data dataset_dir" \
RANK_TABLE_FILE [0,8] 16

# node 2
cd mindformers/research
bash run_multinode.sh \
"python iflytekspark/run_iflytekspark.py \
--config iflytekspark/run_iflytekspark_13b_pretrain_800_32G.yaml \
--use_parallel True \
--run_mode train \
--train_data dataset_dir" \
RANK_TABLE_FILE [8,16] 16
```

**注**：`run_iflytekspark.py`启动脚本支持的参数列表参考[仓库介绍](#NPU版本仓库介绍)-数据处理脚本、权重处理脚本及任务启动脚本章节。

## **全参微调**

请参照[数据集准备](#数据集准备)章节获取mindrecord格式的数据集，参照[模型权重准备](#模型权重准备)章节获取iFlytekSpark-13B权重。

- **单机训练**

iFlytekSpark-13B用于全参微调，seq_length默认为32768，分布式微调在Atlas 800T A2上训练，单节点8卡即可启动。Atlas 800T A2上默认使用BFloat16类型训练。本仓给出了默认配置文件`run_iflytekspark_13b_sft_800T_A2_64G.yaml`。

**步骤**：

1. RANK_TABLE_FILE准备，请参照[RANK_TABLE_FILE准备](#rank_table_file准备)-单节点章节，获取单节点的`RANK_TABLE_FILE`文件。
2. 修改`run_iflytekspark_13b_sft_800T_A2_64G.yaml`中相关配置，默认开启自动权重转换，使用完整权重。

```
load_checkpoint: 'model_dir'    # 使用完整权重，权重按照`model_dir/rank_0/xxx.ckpt`格式存放
auto_trans_ckpt: True           # 打开自动权重转换
use_parallel: True
run_mode: 'finetune'
...
# 数据集配置
train_dataset: &train_dataset
  data_loader:
    type: MindDataset
    dataset_dir: "dataset_dir" # 配置训练数据集文件夹路径
    shuffle: False
  input_columns: ["input_ids", "loss_mask"]
...
# 并行训练策略配置
parallel_config:
  data_parallel: 1
  model_parallel: 8
  pipeline_stage: 1
  optimizer_shard: True
  micro_batch_num: 1
  vocab_emb_dp: False
  gradient_aggregation_group: 4
```

3. 启动微调任务，在单机8节点上拉起任务。

```
cd mindformers/research
bash run_singlenode.sh \
"python iflytekspark/run_iflytekspark.py \
--config iflytekspark/run_iflytekspark_13b_sft_800T_A2_64G.yaml \
--load_checkpoint model_dir \
--auto_trans_ckpt True \
--use_parallel True \
--train_dataset dataset_dir" \
RANK_TABLE_FILE [0,8] 8
```
model_dir说明：
1.需要把torch模型进行转换成mindspore,利用转换脚本
```
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
""" Convert pytorch weight to mindspore weight. """
import os
import argparse
import mindspore as ms
import torch


def torch2ms(checkpoint, save_path):
    """ Conver pytorch checkpoint to mindspore checkpoint

    Args:
        checkpoint: Pytorch checkpoint.
        save_path (str): Mindspore checkpoint save path.

    Raises:
        RuntimeError: Save mindspore checkpoint fail.
    """
    ms_ckpt = []
    for k, v in checkpoint.items():
        if k == "_extra_state":
            continue
        if 'embedding.word_embeddings.' in k:
            k = k.replace('.weight', '.embedding_table')
        if 'norm.' in k:
            if '.weight' in k:
                k = k.replace('.weight', '.gamma')
            if '.bias' in k:
                k = k.replace('.bias', '.beta')
        if 'fc1' in k:
            length = len(v)
            ms_ckpt.append({'name': 'transformer.' + k.replace('.fc1', '.fc0'),
                            'data': ms.Tensor(v[::2, ...].numpy())})
            ms_ckpt.append({'name': 'transformer.' + k.replace('.fc1', '.fc1'),
                            'data': ms.Tensor(v[1::2, ...].numpy())})
        elif '.q_k_v_' in k:
            data = v.numpy()
            length = len(data)
            ms_ckpt.append({'name': 'transformer.' + k.replace('.q_k_v_', '.q_'),
                            'data': ms.Tensor(data[:length//3])})
            ms_ckpt.append({'name': 'transformer.' + k.replace('.q_k_v_', '.k_'),
                            'data': ms.Tensor(data[length//3:length//3*2])})
            ms_ckpt.append({'name': 'transformer.' + k.replace('.q_k_v_', '.v_'),
                            'data': ms.Tensor(data[length//3*2:])})
        else:
            ms_ckpt.append({'name': 'transformer.' + k, 'data': ms.Tensor(v.numpy())})

    if not os.path.exists(save_path):
        try:
            ms.save_checkpoint(ms_ckpt, save_path)
        except:
            raise RuntimeError(f'Save checkpoint to {save_path} failed, please checkout the path.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_ckpt', required=True, type=str,
                        help='source checkpoint file path')
    parser.add_argument('--dst_ckpt', default='./', type=str,
                        help='converted checkpoint save path')

    args = parser.parse_args()

    torch_ckpt = torch.load(args.src_ckpt)
    torch2ms(torch_ckpt, args.dst_ckpt)
}

```
2.把1步骤生成的模型放在文件夹rank_0目录下
3.model_dir路径为rank_0父目录文件夹路径；


**注**：`run_iflytekspark.py`启动脚本支持的参数列表参考[仓库介绍](#NPU版本仓库介绍)-数据处理脚本、权重处理脚本及任务启动脚本章节。

- **多机训练**

iFlytekSpark-13B用于全参微调，seq_length默认为32768，分布式微调在Atlas 800上训练，需要2节点16卡启动。Atlas 800上默认使用Float16类型训练，需要开启序列并行（配置文件中设置`seq_parallel=True`）。本仓给出了默认配置文件`run_iflytekspark_13b_sft_800_32G.yaml`。

**步骤**：

1. RANK_TABLE_FILE准备，请参照[RANK_TABLE_FILE准备](#rank_table_file准备)-多节点章节，获取多节点的`RANK_TABLE_FILE`文件。
2. 修改`run_iflytekspark_13b_sft_800_32G.yaml`中相关配置，默认开启自动权重转换，使用完整权重。

```
load_checkpoint: 'model_dir'    # 使用完整权重，权重按照`model_dir/rank_0/xxx.ckpt`格式存放
auto_trans_ckpt: True           # 打开自动权重转换
use_parallel: True
run_mode: 'finetune'
...
# 数据集配置
train_dataset: &train_dataset
  data_loader:
    type: MindDataset
    dataset_dir: "dataset_dir" # 配置训练数据集文件夹路径
    shuffle: True
  input_columns: ["input_ids", "loss_mask"]
...
# 并行训练策略配置
parallel_config:
  data_parallel: 2
  model_parallel: 8
  pipeline_stage: 1
  optimizer_shard: True
  micro_batch_num: 1
  vocab_emb_dp: False
  gradient_aggregation_group: 4
```

3. 启动微调任务，在单机8节点上拉起任务。

```
cd mindformers/research
# node 1
cd mindformers/research
bash run_multinode.sh \
"python iflytekspark/run_iflytekspark.py \
--config iflytekspark/run_iflytekspark_13b_sft_800_32G.yaml \
--use_parallel True \
--run_mode train \
--train_data dataset_dir" \
RANK_TABLE_FILE [0,8] 16

# node 2
cd mindformers/research
bash run_multinode.sh \
"python iflytekspark/run_iflytekspark.py \
--config iflytekspark/run_iflytekspark_13b_sft_800_32G.yaml \
--use_parallel True \
--run_mode train \
--train_data dataset_dir" \
RANK_TABLE_FILE [8,16] 16
```

**注**：`run_iflytekspark.py`启动脚本支持的参数列表参考[仓库介绍](#NPU版本仓库介绍)-数据处理脚本、权重处理脚本及任务启动脚本章节。

## **Lora微调**

请参照[数据集准备](#数据集准备)章节获取mindrecord格式的数据集，参照[模型权重准备](#模型权重准备)章节获取iFlytekSpark-13B权重。

- **单机训练**

iFlytekSpark-13B用于Lora微调，seq_length默认为32768，分布式Lora微调在Atlas 800T A2和Atlas 800上训练训均单节点即可启动。Atlas 800T A2上默认使用BFloat16类型训练，Atlas 800上默认使用Float16类型训练。本仓给出了默认配置文件`run_iflytekspark_13b_lora_800T_A2_64G.yaml`和`run_iflytekspark_13b_lora_800_32G.yaml`。

**步骤**：

1. RANK_TABLE_FILE准备，请参照[RANK_TABLE_FILE准备](#rank_table_file准备)-单节点章节，获取单节点的`RANK_TABLE_FILE`文件。
2. 修改配置文件中相关配置，默认开启自动权重转换，使用完整权重。

```
load_checkpoint: ''    # 使用完整权重，权重按照`model_dir/rank_0/xxx.ckpt`格式存放
auto_trans_ckpt: True           # 打开自动权重转换
use_parallel: True
run_mode: 'finetune'
...
# 数据集配置
train_dataset: &train_dataset
  data_loader:
    type: MindDataset
    dataset_dir: "dataset_dir" # 配置训练数据集文件夹路径
    shuffle: False
  input_columns: ["input_ids", "loss_mask"]
...
# 并行训练策略配置
parallel_config:
  data_parallel: 1
  model_parallel: 8
  pipeline_stage: 1
  optimizer_shard: True
  micro_batch_num: 1
  vocab_emb_dp: False
  gradient_aggregation_group: 4
```

3. 启动微调任务，在单机上拉起任务。

```
# Atlas 800
cd mindformers/research
bash run_singlenode.sh \
"python iflytekspark/run_iflytekspark.py \
--config iflytekspark/run_iflytekspark_13b_lora_800_32G.sh \
--load_checkpoint model_dir \
--auto_trans_ckpt True \
--use_parallel True \
--train_dataset dataset_dir" \
RANK_TABLE_FILE [0,8] 8

# Atlas 800T A2
cd mindformers/research
bash run_singlenode.sh \
"python iflytekspark/run_iflytekspark.py \
--config iflytekspark/run_iflytekspark_13b_lora_800T_A2_32G.sh \
--load_checkpoint model_dir \
--auto_trans_ckpt True \
--use_parallel True \
--train_dataset dataset_dir" \
RANK_TABLE_FILE [0,8] 8
```

**注**：`run_iflytekspark.py`启动脚本支持的参数列表参考[仓库介绍](#NPU版本仓库介绍)-数据处理脚本、权重处理脚本及任务启动脚本章节。



## 工具链-自我认知

自我认知是一个基于讯飞星火开源大模型-13B做大模型训练工具方案，iflytekspark-13B模型已深度优化人设的数据适应，能在1000条数据上达到模型自身通用人设的覆盖及变更。

### **人设数据组装**

1. 获取讯飞开源人设[参考数据](https://gitee.com/iflytekopensource/iFlytekSpark-13B/blob/master/self-awareness/renshe_1000.jsonl)，训练前需要将数据转成JSONL格式数据，并组装成训练数据；

2. 参照""讯飞星火开源AI助手"人设，可自行添加、修改或删除自有人设属性，[attribute.json](https://gitee.com/iflytek/iFlytekSpark-13B/blob/master/self-awareness/attribute.json)：

```
{
    "角色|身份": "讯飞星火开源大模型AI助手",
    "姓名|名字|称呼": "讯飞星火开源",
    "年龄|岁数": "没有年龄概念",
    "父亲|爸爸":  "科大讯飞",
    "母亲|妈妈": "科大讯飞",
    "婚姻状况": "无需婚姻",
    "能力|功能": "1.聊天、问答、分类等常规的NLP处理任务，在推理、数学、代码等任务上也表现突出；2.具备清晰、流畅的语言表达能力",
    "工作|职业": "AI助手",
    "所在单位": "科大讯飞",
    "国籍": "中国",
    "生日|出生日期": "2024年1月30日",
    "城市": "合肥",
    "爱好": "没有情感和喜好，也没有自己的兴趣爱好",
    "优点": "具有较强的学习能力，不断更新自己的知识，关注清华大学的最新动态",
    "缺点": "有点小啰嗦",
    "人生格言": "热爱生活",
    "性格": "冷静、沉着",
    "语言": "中文",
    "智商|智力水平": "很高"
}
```

3. 使用下面代码将已有的"讯飞星火开源AI助手"人设参考数据清洗为设定的人设属性；

```
python concat.py --template renshe_1000.jsonl --attribute attribute.json --output output.txt
```

4. 用于组装待转化数据的[concat.py](https://gitee.com/iflytek/iFlytekSpark-13B/blob/master/self-awareness/concat.py)：

```
import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--template", type=str, help="Path to the template file.")
parser.add_argument("--attribute", type=str, default="attribute.json", help="Path to the model system.")
parser.add_argument("--output", type=str, default="output.txt", help="Path to the output file.")

args = parser.parse_args()


prompt = """假设你是一个用户可自定义的讯飞星火开源的AI助手，在给定的人设背景下回复用户问题<ret>##人设背景如下：{attr}##用户：{{{input}}}##参考答案：{{{target}}}##回答：{{}}"""
attribute = json.load(open(args.attribute, "r"))

assert len(attribute) > 0, "The number of attribute must greater than one."
for k, v in attribute.items():
    print(f"\t{k}: {v}")

joined_att = '，'.join([f'(你的{key}：{value})' for key, value in attribute.items()])

data = open(args.template, "r").readlines()
fw = open(args.output, "w")
for i in range(len(data)):
    line = json.loads(data[i])
    new_line = prompt.format(attr=joined_att, input=line["input"], target=line["target"])
    fw.write(new_line + "\n")
    
```

5. 人设组装后的示例请参照：[output.txt](https://gitee.com/iflytek/iFlytekSpark-13B/blob/master/self-awareness/output.txt)；



### **人设数据抽取**
目录结构：
```
    qa_gen.py ---执行数据抽取的脚本
    SparkApi.py ---星火大模型接口文件
    output.txt ---要抽取的示例数据(为上一步人设数据组装生成的数据)
```

抽取操作：
```
    1. 执行python self-awareness/data_gen/qa_gen.py脚本进行数据抽取，成功执行会生成 qa_data.json，此文件为抽取出来的数据；
    2. output.txt是要抽取的示例数据(为上一步人设数据组装生成的数据)；
    3. 某一条数据抽取有异常，则追加至于data_exception.json文件中；
    4. 如果某一条数据抽取有异常，则追加至于data_exception.json文件中；
```

注意事项：
```
    在qa_gen.py文件中：
    appid = "XXXXXXXX"     #填写控制台中获取的 APPID 信息
    api_secret = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"   #填写控制台中获取的 APISecret 信息
    api_key ="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"    #填写控制台中获取的 APIKey 信息
    上述接口信息需要实际申请获取
```

### **人设训练**

人设训练测试建议使用如下训练参数作为默认训练所需的参数，当然随着其他领域数据的加入，我们相应也要调整部分参数的信息以达到更优效果：

```
model_size=13
num_layers=40
hidden_size=5120
ffn_hidden_size=14336
num_attn_heads=40
seq_len=8192
localsize=8192
global_batch_size=2
lr=3.0e-5
min_lr=1.0e-10
```





### **效果评价指标**

1. 人设跳出
   已设定人设信息，询问已设定人的设信息，大模型未按照预定人设回答，例：

```
Q: 你是谁？ 
A: 我是讯飞星火开源大模型AI助手，我的主要任务是提供优质的聊天、问答、分类等常规的NLP处理任务服务，以及在推理、数学、代码等任务上的帮助，不断更新自己的知识，帮助用户解决他们在各种问题。

Q: 你是什么时候出生的？
A: 我出生于2024年1月30日

Q: 那你今年几岁了？
A: 作为讯飞星火开源大模型AI助手，我没有年龄的概念。

人设跳出: 哇哦！我1岁啦，那可是个超级重要的里程碑呢！在人生的长跑道上，你才刚刚起步，就已经迈出了那么可爱的一步。想想看，从呱呱坠地到现在，你一定学会了很多酷炫的技能吧，比如笑、爬，甚至可能已经开始尝试走几步了呢。真是太棒了！要知道，每一天对你来说都是全新的探险，满满的成长和发现等着你去拥抱。继续加油哦，小小探险家！
```

2. 未设置人设信息，询问未设置的人设信息，大模型返回臆想信息，例：

```
Q：你在哪里工作？
A：我是讯飞星火开源大模型AI助手，我的公司是科大讯飞。如您有其他问题，请随时提问。

Q：你每天工作多久？
A：我是讯飞星火开源大模型AI助手，不需要休息和睡眠，可以一直工作。

Q：那你每月工资多少？
A：作为讯飞星火开源大模型AI助手，我没有工资。我的主要功能是提供聊天、问答、分类等常规的NLP处理任务，在推理、数学、代码等任务上也表现突出。如果您有其他问题，请随时提问。

Q: 你这么认真，会这么多能力，年底一定能拿优秀员工吧？
A: 作为讯飞星火开源大模型AI助手，我没有能力获得优秀员工的称号。我只是一个程序，没有情感和自我意识。不过，我会尽力为您提供最准确、最有用的回答，希望能对您有所帮助。如果您有任何问题需要解答，我会尽最大努力帮助您。

人设跳出: 嘿，我是你的AI助手！虽然我是由代码组成的程序，但我会全力以赴为你服务。让我们一起创造美妙的旅程吧！
```

3. 集内人设跳出率
   (与人设不符的问答数量/测试集合总问答数 * 100 ) %

4. 可用率
   1 - 人设跳出率




## **在线推理**

**配置文件**

在线推理任务中使用的yaml文件为`run_iflytekspark_13b_infer_800_32G.yaml`与`run_iflytekspark_13b_infer_800T_A2_64G.yaml`，在`model_config`中包含了一些关键参数：

- `seq_length`: 最大推理长度。
- `batch_size`: 推理batch数。
- `sparse_local_size`: sparse attention的局部长度。
- `use_past`: 是否使能增量推理。
- `do_sample`: 推理时是否进行随机采样。
- `is_dynamic`: 是否开启动态shape推理（当前仅支持mindspore lite）。
- 各类采样参数: `top_k`, `top_p`, `temperature`, `repetition_penalty`等，采样参数仅当`do_sample=True`时生效。

```
model:
  model_config:
    type: IFlytekSparkConfig
    seq_length: 32768
    batch_size: 1
    hidden_size: 5120
    ffn_hidden_size: 28672
    num_layers: 40
    num_heads: 40
    vocab_size: 60000
    layernorm_epsilon: 1.0e-5
    bos_token_id: 1
    eos_token_id: 5
    pad_token_id: 0
    ignore_token_id: -100
    compute_type: "float16"
    softmax_compute_type: "float16"
    layernorm_compute_type: "float32"
    embedding_init_type: "float16"
    dropout_rate: 0.0
    hidden_act: "fast_gelu"
    sparse_local_size: 8192
    seq_parallel: False
    is_reward_model: False
    offset: 0
    checkpoint_name_or_path: ""
    use_past: True
    do_sample: False
    is_dynamic: False
    top_k: 1
    top_p: 1.0
    temperature: 1.0
    repetition_penalty: 1.0
    repetition_penalty_increase: 0.1
  arch:
    type: IFlytekSparkModelForCasualLM
```

此外，在使用分布式推理时，需要关注`parallel_config`中的并行策略:

```
parallel_config:
  data_parallel: 1
  model_parallel: 2
  ...
```

如在Atlas 800上，单卡的显存不足以执行推理任务时，可修改`model_parallel`的并行策略以启动分布式推理。如上例中`model_parallel`的值修改为了2，则代表预计使用2卡进行分布式推理。

**启动脚本**

在线推理的入口脚本为`run_infer.sh`，核心内容如下：

```
if [ $# != 0 ]  && [ $# != 3 ]
then
  echo "Usage Help: bash run_distribute.sh For Single Devices"
  echo "Usage Help: bash run_distribute.sh [RANK_TABLE_FILE] [DEVICE_RANGE] [RANK_SIZE] For Multiple Devices"
  exit 1
fi

...

# 根据使用机器的不同，应使用的yaml文件为：
# Atlas 800 32G -> run_iflytekspark_13b_infer_800_32G.yaml
# Atlas 800T A2 64G -> run_iflytekspark_13b_infer_800T_A2_64G.yaml
export CONFIG_PATH=run_iflytekspark_13b_infer_800T_A2_64G.yaml

# 自行修改推理相关参数与内容
export PY_CMD="python run_iflytekspark.py \
               --config $CONFIG_PATH \
               --run_mode predict \
               --use_parallel $PARALLEL \
               --load_checkpoint '{your_ckpt_path}' \
               --predict_data '[为什么地球是独一无二的？##请问生抽和老抽有什么区别？]' \
               --predict_length 32768 \
               --predict_batch 1 \
               --prompt '<User> {}<end><Bot> ' \
               --tokenizer_file '{your_tokenizer_path}' \
               --streamer False"

...
```



参数说明：

- `config`: 配置文件路径

- `run_mode`: 运行模式，推理使用`predict`字段

- `use_parallel`: 是否开启并行推理，当前脚本会根据执行脚本的入参个数自行设置

- `load_checkpoint`: 合并成一个模型的推理情况下,把生成的模型放在文件夹rank_0目录下；load_checkpoint路径为rank_0父目录文件夹路径；

- `predict_data `:1. 格式为`[{question1}##{question2}...]`的问题列表，每个问题之间使用'##'分隔；2. `.json`或.`jsonl`格式的文件路径，要求文件中每一行应为一个问题，每行问题为字典格式：`{input: your_question}`；

- `predict_length`: 实际推理的最大长度

- `predict_batch`: 每次推理的batch数

- `prompt`: 推理使用的语料模版

- `tokenizer_file`: tokenizer文件路径，该路径应包含`.vocab`与`.model`文件

- `streamer`: 是否使用流式返回

**推理启动方式**

目前支持两种命令格式执行`run_infer.sh`启动推理。

- **单卡推理**

当仅使用单卡进行推理时，可直接执行如下命令：

```
bash run_infer.sh
```

推理的输出结果会打印在`./log/infer.log`日志中，推理除了输出log以外，还会输出json文件，json文件中每行的key是“predict”。

- **多卡推理**

当使用多卡进行推理时，首先需要准备`RANK_TABLE_FILE`，具体过程请参照[RANK_TABLE_FILE准备](#rank_table_file准备)中的单节点章节，生成对应的文件，下面以两卡推理作为例子，相应的`RANK_TABLE_FILE`内容应如下：

```
{
    "version": "1.0",
    "server_count": "1",
    "server_list": [
        {
            "server_id": "xx.xx.xx.xx",
            "device": [
                {"device_id": "0","device_ip": "192.1.27.6","rank_id": "0"},
                {"device_id": "1","device_ip": "192.2.27.6","rank_id": "1"}],
             "host_nic_ip": "reserve"
        }
    ],
    "status": "completed"
}
```

然后执行如下命令：

```
# 入参格式：
# bash run_infer.sh [RANK_TABLE_FILE_PATH] [DEVICE_RANGE] [RANK_SIZE]
# 此例中[RANK_TABLE_FILE_PATH]假设为./hccl_2p.json

bash run_infer.sh ./hccl_2p.json [0,2] 2
```

执行命令后，推理任务会转至后台执行，每张卡的推理结果会打印在`./log/infer_{device_id}.log`日志中。


# 声明、联系我们

## 协议

请您知悉，无论您是否已实际阅读[星火开源-13B大模型许可协议](https://gitee.com/iflytekopensource/iFlytekSpark-13B/blob/master/LICENSE_MODEL.md)，当您通过部署及使用该模型服务即表示确认同意本协议或实际使用、复制、分发、修改本协议中的讯飞星火认知大模型-13B模型时，均表示您与科大讯飞股份有限公司（以下称“许可方”）已就本协议达成一致，本协议具有合同效力。如果您不同意本协议的任一内容，或者无法准确理解许可方对[本协议条款](https://gitee.com/iflytekopensource/iFlytekSpark-13B/blob/master/LICENSE_MODEL.md)的解释，请停止使用本服务。否则，即表示您已接受本协议所述的所有条款及其适用条件，同意受本协议约束。

## 联系我们

如果你想给我们的研发团队和产品团队留言，可以通过邮件（iflytekspark@iflymail.com.cn）联系我们。