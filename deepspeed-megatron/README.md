# <center>iFlytekSpark-13B</center>

# 模型介绍

讯飞星火开源-13B（iFlytekSpark-13B）拥有130亿参数，在经过累计超过3万亿以上tokens海量高质量数据集上进行预训练，然后在精调得多元化对齐数据上进行微调得到。iFlytekSpark-13B在多个标准评估中展现出了卓越的性能，其表现优于同参数量级的开源模型，与一些闭源模型相比不相上下。

iFlytekSpark-13B不仅具备通用任务处理能力如聊天、问答、文本提取和分类等，还具备数据分析和代码生成等生产力功能。我们特别在学习辅助、数学、推理等领域进行了深度优化，大幅提升模型的实用性和易用性。详细的评测结果见下面评测部分。

此次开源推动人工智能和机器学习领域的开源协作，并在全球范围内促进技术革新。欢迎大家积极利用和贡献于iFlytekSpark-13B，共同促进人工智能的发展。

# GPU版本仓库介绍

`iFlytekSpark` 基于 [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed/) 实现，主要涉及的文件有：

1. 模型训练入口：`deepspeed-megatron/`

```
├─deepspeed-megatron
      ├── pretrain_iFlytekSpark.py     # 预训练入口文件
      ├── sft_iFlytekSpark.py          # 全量微调及Lora微调入口文件
```

2. 模型具体实现：`deepspeed-megatron/`

```
├─deepspeed-megatron
      ├─tools
      │  ├─ run_iFlytekSpark_text_generation.py # 推理入口文件
      ├─megatron
          ├─tokenizer
          │    iFlytekSpark_tokenization.py   # 词表实现
          ├─model
          │    iFlytekSpark_model.py          # 模型实现
          │    transformer.py                 # 模型模块实现

```

3. 数据处理脚本、权重处理脚本及任务启动脚本：`deepspeed-megatron/examples_deepspeed/iFlytekSpark`

```
├─deepspeed-megatron
      ├─examples_deepspeed
          ├─iFlytekSpark
                ├── preprocess_data_iFlytekSpark.py        # 预训练数据处理脚本
                ├── preprocess_sft_data_iFlytekSpark.py    # 微调数据处理脚本
                ├── run_pretrain_iFlytekSpark.sh           # 预训练启动脚本
                ├── run_sft_iFlytekSpark.sh                # 全量微调启动脚本
                ├── run_lora_iFlytekSpark.sh               # Lora微调启动脚本
                ├── run_generate_text_iFlytekSpark.sh      # 在线推理启动脚本
                ├── run_lora_weights_merge_iFlytekSpark.sh # Lora权重合并脚本
```

## 环境安装

### 环境要求

- 硬件：a800/a100/h800/h100 (80G/40G)
- Cuda: 11.7
- Pytorch：1.13
- Python：3.7

| 模型          | 硬件 | 全量微调 | lora微调 | 推理 |
| ------------- | ---- | -------- | -------- | ---- |
| iflyspark-13b | a800/a100/h800/h100  | 单节点（最低 8卡*80G）   | 单节点（最低 单卡*40G）   | 单卡（最低40G） |

```
# 基础包安装
pip install -r requirement.txt
```

**[DeepSpeed](https://github.com/microsoft/DeepSpeed)可直接按照requirement.txt使用pip安装，[Megatron](https://github.com/NVIDIA/Megatron-LM)已在仓库中集成**

**[apex](https://github.com/NVIDIA/apex)与[flash-attn](https://github.com/Dao-AILab/flash-attention) (版本>=2.3.3)需要编译安装，详情可参照对应链接**

**注：多机多卡需要在所有机器上安装pdsh并设置ssh免密登录。**

### 数据集准备

- **预训练数据**

  数据源格式： txt 文件，具体格式如下(需要空行分割不同样本)：

  ```
  sample1 ***
  ***

  sample2 ***
  ***
  ```

  执行脚本`preprocess_data_iFlytekSpark.py`，进行预训练数据预处理、训练数据生成，将原始数据转换为`.bin`和`.idx`格式。

  ```
  # 预训练
  # 路径推荐都加上引号
  # 可参考/examples_deepspeed/iFlytekSpark/preprocess_data_pretrain.sh
  python ./examples_deepspeed/iFlytekSpark/preprocess_data_iFlytekSpark.py \
        --tokenizer "/{TOKENIZER_PATH}" \
        --raw_data_path "/{RAW_DATA_PATH}/*.txt" \
        --output_filepath "/{OUTPUT_FILE_PATH}" \
        --dataset-impl mmap \
        --append-eod

  # 参数说明
  tokenizer: Tokenizer文件路径
  raw_data_path: 原始数据目录路径，在此目录下存放 txt 文件
  output_filepath: 数据集输出位置
  dataset-impl: 数据集保存方式，默认为mmap
  append-eod: 在每个document后添加eod标识符
  ```

  **将会生成/{OUTPUT_FILE_PATH}/xxx.idx和/{OUTPUT_FILE_PATH}/xxx.bin文件，训练时需填写`--data-path=/{OUTPUT_PREFIX}/xxx`**

- **全参微调&Lora微调数据**

  数据源格式： json 文件，具体格式如下：

  ```
  {"input":"Are you ready?","target":"Yes"}
  {"input":"How are you?","target":"Fine"}
  ```

  执行脚本`preprocess_sft_data_iFlytekSpark.py`，进行微调数据预处理、训练数据生成，将原始数据转换为`.bin`和`.idx`格式。

  ```
  # 全参微调&Lora微调
  # 路径推荐都加上引号
  # 可参考/examples_deepspeed/iFlytekSpark/preprocess_sft_data_pretrain.sh
  python ./examples_deepspeed/iFlytekSpark/preprocess_sft_data_iFlytekSpark.py \
        --tokenizer "/{TOKENIZER_PATH}" \
        --raw_data_path "/{RAW_DATA_PATH}/*.json" \
        --output_filepath "/{OUTPUT_FILE_PATH}" \
        --seq_length SEQ_LENGTH
        --append-eod

  # 参数说明
  seq_length: 数据集单个样本句长
  ```

  **将会生成/{OUTPUT_FILE_PATH}/xxx.idx和/{OUTPUT_FILE_PATH}/xxx.bin文件，训练时需填写`--data-path=/{OUTPUT_PREFIX}/xxx`**

  **注：seq_length应与后续训练中seq_len参数一致**

## 模型权重准备

本仓提供对应的预训练权重、微调权重和词表文件用于训练/微调/推理。

预训练权重：     
[iFlytekSpark_13B_base_fp32](https://openi.pcl.ac.cn/iflytek/iFlytekSpark-13B/modelmanage/model_filelist_tmpl?name=iFlytekSpark_13B_base_fp32)

微调权重：     
[iFlytekSpark_13B_chat_fp32](https://openi.pcl.ac.cn/iflytek/iFlytekSpark-13B/modelmanage/model_filelist_tmpl?name=iFlytekSpark_13B_chat_fp32)

Tokenizer：      
[Tokenizer](https://openi.pcl.ac.cn/iflytek/iFlytekSpark-13B/modelmanage/model_filelist_tmpl?name=Tokenizer)

**模型权重转换**

从仓库中下来的权重通常是完整权重。

- 基于完整权重进行多卡分布式训练，需要将完整权重转换为分布式权重。
- 基于训完的分布式权重进行单卡推理，需要将分布式权重转换为完整权重。
- 修改分布式策略训练，需要将权重转换为对应分布式权重。

本仓库提供了权重转换脚本，位于`deepspeed-megatron/tools/weight_convert`。

- **完整权重转换为分布式权重**

  ```
  # 可参考deepspeed-megatron/tools/weight_convert/ds_partition_ckpts.sh
  python ./ds_partition_ckpts.py \
        --src_ckpt /{INPUT_PATH} \
        --dst_ckpt /{OUTPUT_PATH} \
        --tp  {TP_SIZE}

  # 参数说明
  src_ckpt: 原始完整权重保存路径
  dst_ckpt: 拆分后权重保存路径
  tp: 拆分后的tensor并行维度
  ```

- **分布式权重合并**

    ```
    # 分布式权重合并
    # 可参考deepspeed-megatron/tools/weight_convert/ds_merge_ckpts.sh
    python ./ds_merge_ckpts.py \
          --src_ckpt /{INPUT_PATH} \
          --dst_ckpt /{OUTPUT_PATH} \

    # 参数说明
    src_ckpt: 拆分权重保存路径
    dst_ckpt: 合并后权重保存路径
    ```

## 预训练

- 单机多卡
  - step 1. 修改模型对应的配置, 在 `deepspeed-megatron/examples_deepspeed/iFlytekSpark/run_pretrain_iFlytekSpark.sh`中, 用户可以自行修改模型配置、训练相关参数, 并通过 `data_path` 参数, 指定训练数据集的路径。

    ```
    # 模型参数：
    - num_layers: 40
    - hidden_size: 5120
    - ffn_hidden_size: 14336
    - num_attn_heads: 40
    - seq_len: 32768
    - localsize: 8192
  
    # 启动脚本重要参数说明：
    - from_pretrained：加载checkpoint路径，默认None。
    - data_path：数据集路径，参考数据集生成部分，需要指定到存放数据集目录下.bin和.idx文件前缀，详见快速开始示例。。
    - vocab_path：tokenizer文件路径，需要指定到存放tokenizer目录下.model和.vocab文件前缀，详见快速开始示例。
    - output_home：结果输出路径，log、tensorboard、ckpt输出在其子目录。
    - mp_size：张量并行维度，默认为8。
    - pp_size：流水并行维度，默认为1。
    - global_batch_size：总batch_size，为batch_size\*dp_size\*梯度累计数目。
    - eval_interval: 控制eval进行的频率
    - save_interval: 控制保存ckpt进行的频率
    - activation_checkpoint：是否使用重计算节省显存，默认true。
    - template_json: deepspeed config模板，会根据配置生成ds config。
    ```

    **快速开始：仅需修改`data_path`，`vocab_path`，`output_home`参数**

     ```text
      ├─data
          ├─dataset
          │      text_document.bin
          │      text_document.idx
          ├─tokenizer
          │      tokenizer.model
          │      tokenizer.vocab
          ├─output
     ```

    ```shell
    # 若文件结构如上所示，则对应参数如下
    data_path="/data/dataset/text_document"
    vocab_path="/data/tokenizer/tokenizer" #注意使用tokenizer/目录下.model和vocab的前缀名字tokenizer ，而不是只到tokenizer/目录
    output_home="/data/output"

    注意！！根据实际数据量的大小酌情修改train_tokens和train_samples的系数。比如数据量比较少，请修改 train_tokens和train_samples的系数，将其改小一些，否则构造数据会耗时较长时间。
    ```

  - step 2. 启动运行脚本, 进行 8卡分布式运行

    ```
    cd deepspeed-megatron/examples_deepspeed/iFlytekSpark
    bash run_pretrain_iFlytekSpark.sh \
    ```

- 多机多卡
  - step 1. 多机运行需要使用Deepspeed多机启动，需要修改hostfile.txt文件及脚本中的master_addr等配置
  - step 2. 在主节点启动运行脚本，进行 8卡分布式运行，log将会输出至`output_home/log/`中

    ```
    # master node
    cd deepspeed-megatron/examples_deepspeed/iFlytekSpark
    bash run_multinode_pretrain_iFlytekSpark.sh \
    ```

  **注：目前的示例脚本使用deepspeed命令**`**ds**`**启动训练及推理，具体文档可参考[DeepSpeed官方文档](https://www.deepspeed.ai/getting-started/)以及[huggingface的DeepSpeed文档](https://huggingface.co/docs/transformers/main/main_classes/deepspeed)，以下给出几个典型的示例：**

  ```
  # 单卡的使用方法
  ds --num_gpus=1 your_program.py.py ...
  # 单卡，并指定对应的GPU
  ds --include localhost:1 your_program.py ...

  # 多GPU的使用方法
  ds --num_gpus=2 your_program.py --deepspeed_config ds_config.json

  # 多机多卡，需要创建一个 hostfile文件，hostname为ip，slots为卡数，只需在一个节点上启动
  hostname1 slots=8
  hostname2 slots=8
  # 然后运行
  ds --hostfile hostfile --master_addr hostname1 --master_port=9901 your_program.py --deepspeed ds_config.json
  ```

  **注：若出现端口被占用情况，可参照以上示例在脚本中ds命令后添加 --master_port=9901切换启动端口**

## 全参微调

请参照[数据集准备](#数据集准备)章节中(全参微调&Lora微调数据)获取微调格式的数据集，参照[模型权重准备](#模型权重准备)章节获取iflyspark-13B分布式权重。

- **单机训练**

  iflyspark-13B用于全参微调，seq_length默认为32768，分布式微调训练在单节点即可启动。本仓给出了默认启动脚本`examples_deepspeed/iFlytekSpark/run_sft_iFlytekSpark.sh`。

  - step 1. 修改`run_sft_iFlytekSpark.sh`中相关配置，默认使用分布式权重。

    ```shell
    # 启动脚本重要参数说明：
    - from_pretrained：加载checkpoint路径，默认None，需配置为文件夹位置，若权重按照`/data/model_dir/mp_rank_0x_model_states.ckpt`格式存放，则该参数为`/data/model_dir`，详见快速开始示例。
    - data_path：数据集路径，参考数据集生成部分，需要指定到存放数据集目录下.bin和.idx文件前缀，详见快速开始示例。
    - vocab_path：tokenizer文件路径，需要指定到存放tokenizer目录下.model和.vocab文件前缀，详见快速开始示例。
    - output_home：结果输出路径，log、tensorboard、ckpt输出在其子目录。
    - mp_size：张量并行维度，默认为8。
    - pp_size：流水并行维度，默认为1。
    - global_batch_size：总batch_size，为batch_size\*dp_size\*梯度累计数目。
    - eval_interval: 控制eval进行的频率
    - save_interval: 控制保存ckpt进行的频率
    - activation_checkpoint：是否使用重计算节省显存，默认true。
    - template_json: deepspeed config模板，会根据配置生成ds config。
    - train-data-exact-num-epochs: 实际训练epoch数，若不设置则默认为1。
    ```

    **快速开始：仅需修改`from_pretrained`，`data_path`，`vocab_path`，`output_home`参数**

     ```text
      ├─data
          ├─ckpt_tp8
          │      mp_rank_00_model_states.pt
          │      ...
          │      mp_rank_07_model_states.pt
          ├─dataset
          │      seq_length_32768_text_document.bin
          │      seq_length_32768_text_document.idx
          ├─tokenizer
          │      tokenizer.model
          │      tokenizer.vocab
          ├─output
     ```

    ```shell
    # 若文件结构如上所示，则对应参数如下
    from_pretrained="/data/ckpt_tp8"
    data_path="/data/dataset/seq_length_32768_text_document" #注意使用数据处理之后目录下生成的.bin和.idx文件的前缀名字seq_length_32768_text_document ，而不是只到dataset/目录
    vocab_path="/data/tokenizer/tokenizer" #注意使用tokenizer/目录下.model和vocab的前缀名字tokenizer ，而不是只到tokenizer/目录
    output_home="/data/output"
    ```

  - step 2. 启动微调任务，在单机上拉起任务。

    ```shell
    cd deepspeed-megatron/examples_deepspeed/iFlytekSpark
    bash run_sft_iFlytekSpark.sh \
    ```

## Lora微调

请参照[数据集准备](#数据集准备)章节(全参微调&Lora微调数据)获取微调格式的数据集，参照[模型权重准备](#模型权重准备)章节获取iflyspark-13B分布式权重。

- **单机训练**

  iflyspark-13B用于Lora微调，seq_length默认为32768，分布式微调训练在单节点即可启动。本仓给出了默认启动脚本`deepspeed-megatron/examples_deepspeed/iFlytekSpark/run_sft_lora_iFlytekSpark.sh`。

  - step 1. 修改`run_sft_lora_iFlytekSpark.sh`中相关配置，默认使用分布式权重。

    ```shell
    #启动脚本重要参数说明：
    - apply_lora：是否启用lora，默认为true。
    - lora-from-pretrained：lora部分ckpt位置，默认为none。
    - adapt-q：是否对query layer使用lora。
    - adapt-k：是否对key layer使用lora。
    - adapt-v：是否对value layer使用lora。
    - adapt-o：是否对qkv out layer使用lora。
    - adapt-fc1：是否对mlp第一个线性层使用lora。
    - adapt-fc2：是否对mlp第二个线性层使用lora。
    ```

    **快速开始：仅需修改`from_pretrained`，`data_path`，`vocab_path`，`output_home`参数**

     ```text
      ├─data
          ├─ckpt_tp8
          │      mp_rank_00_model_states.pt
          │      ...
          │      mp_rank_07_model_states.pt
          ├─dataset
          │      seq_length_32768_text_document.bin
          │      seq_length_32768_text_document.idx
          ├─tokenizer
          │      tokenizer.model
          │      tokenizer.vocab
          ├─output
     ```

    ```shell
    # 若文件结构如上所示，则对应参数如下
    from_pretrained="/data/ckpt_tp8"
    data_path="/data/dataset/seq_length_32768_text_document" #注意使用数据处理之后目录下生成的.bin和.idx文件的前缀名字seq_length_32768_text_document ，而不是只到dataset/目录
    vocab_path="/data/tokenizer/tokenizer" #注意使用tokenizer/目录下.model和vocab的前缀名字tokenizer ，而不是只到tokenizer/目录
    output_home="/data/output"
    ```

- step 2. 启动微调任务，在单机上拉起任务。

    ```shell
    cd deepspeed-megatron/examples_deepspeed/iFlytekSpark
    bash run_sft_lora_iFlytekSpark.sh \
    ```

  **Lora训练时，默认仅保存Lora部分权重，若想将此部分权重合并到dense模型中，本仓库提供了Lora权重合并工具，`deepspeed-megatron/sft_iFlytekSpark_lora_merge.py`，以及启动脚本`run_lora_weights_merge_iFlytekSpark_13B.sh`**

  ```shell
  # lora权重合并，合并时配置与训练时应当一致
  cd deepspeed-megatron/examples_deepspeed/iFlytekSpark
  bash run_lora_weights_merge_iFlytekSpark_13B.sh \

  # 参数说明
  - from-pretrained：需要设置为合并的dense部分ckpt位置。
  - lora-from-pretrained：需要设置为合并的lora部分ckpt位置。
  - merge_weights: 是否合并权重，脚本中默认设置为true。
  # 合并后的权重保存至lora-from-pretrained文件夹下的lora-merged中
  ```

## 在线推理

  **快速开始：修改`from_pretrained`，`tokenizer_file`参数**

   ```text
    ├─data
        ├─ckpt_download
        │      mp_rank_00_model_states.pt # 完整权重
        ├─tokenizer
        │      tokenizer.model
        │      tokenizer.vocab
   ```

  ```shell
  # 若文件结构如上所示，则对应参数如下
  from_pretrained="/data/ckpt_download"
  tokenizer_file="/data/tokenizer/tokenizer" #注意使用tokenizer/目录下.model和vocab的前缀名字tokenizer ，而不是只到tokenizer/目录
  ```

推理任务使用的脚本为`run_generate_text_iFlytekSpark.sh`，其中包含了一些关键参数：

- `seq_length`: 最大推理长度
- `batch_size`: 推理batch数
- `local_size`: sparse attention的局部长度
- 各类采样参数: `top_k`, `top_p`, `temperature`等
- `from_pretrained`: ckpt的加载路径
- `tokenizer_file`: tokenizer文件路径
- `predict_data`: 待推理问题，每个问题之间使用','分隔
- `predict_length`: 生成token长度
- `json-input-path`: json输入路径
- `json-output-path`: 推理结果json输出路径

使用命令启动推理，推理结果输出在`infer.log`中

```
bash run_generate_text_iFlytekSpark.sh
```

**注：若需要使用json作为待推理问题输入，则可配置json-input-path及json-output-path**

  输入json具体格式如下：

  ```
  {"input":"Are you ready?","target":"Yes"}
  {"input":"How are you?","target":"Fine"}
  ```

  输出json格式如下，其中generate为模型生成结果：

  ```
  {"input":"Are you ready?","target":"Yes","generate":"Yes"}
  {"input":"How are you?","target":"Fine","generate":"Fine"}
  ```

