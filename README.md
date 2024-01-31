<div align="center">
<h1>
  iFlytekSpark-13B
</h1>
</div>

# 目录
- [模型介绍](#模型介绍)
- [效果评测](#效果评测)
- [使用教程](#使用教程)
- [声明、联系](#声明联系我们)


# 模型介绍

讯飞星火开源-13B（iFlytekSpark-13B）拥有130亿参数，新一代认知大模型，一经发布，众多科研院所和高校便期待科大讯飞能够开源。 为了让大家使用的更加方便，科大讯飞增加了更多的数据，并针对工具链进行了优化。此次正式开源拥有130亿参数的iFlytekSpark-13B模型（讯飞星火开源-13B），也是首个基于全国产化算力平台“飞星一号”的大模型，正式开源！

iFlytekSpark-13B不仅具备通用任务处理能力如聊天、问答、文本提取和分类等，还具备数据分析和代码生成等生产力功能。我们特别在学习辅助、数学、推理等领域进行了深度优化，大幅提升模型的实用性和易用性。详细的评测结果见下面评测部分。

本次开源，既包含基础模型iFlytekSpark-13B-base、精调模型iFlytekSpark-13B-chat，也开源了微调工具iFlytekSpark-13B-Lora、人设定制工具iFlytekSpark-13B-Charater，让企业和学术研究可以基于这些全栈自主创新的星火优化套件方便地训练自己的专用大模型。

星火开源-13B在多项知名公开评测任务中名列前茅，在文本生成、语言理解、文本改写、行业问答、机器翻译等企业典型场景中，通过对学习辅助、语言理解等领域的深入研究和优化，大幅提升了其实用性，在处理复杂的自然语言任务时更加得心应手，确保了其在面对多样化和专业化的应用场景时能够保持高效和准确，效果显著优于其他同等尺寸的开源模型。

这对于追求高性能而对成本敏感的企业来说，无疑是一个巨大的吸引力，也为各行各业的企业提供了一种性价比高的解决方案。


### 模型结构

在iFlytekSpark-13B中，我们使用Rotary Embedding作为位置编码方法，GELU作为激活函数，其中layer_num为40，head_num为40，hidden_size为5120，ffn_hidden_size为28672。

# 效果评测

我们在八个具有挑战性的中英文测试集上对模型进行性能评估。其中chat模型采用0-shot进行测试，base模型在C-EVAL，MMLU，CMMLU，FinanceIQ测试集上采用5-shot进行测试，其余测试集采用0-shot进行测试。

### 数据集介绍
- C-EVAL：C-Eval 是一个全面的中文基础模型评估套件，涵盖了52个不同的学科和四个难度级别，验证集包括1346个选择题，测试集包含12342个选择题。本项目采用C-Eval验证集进行测试。
- MMLU：MMLU 是一个庞大的多任务数据集，由各种学科的多项选择题组成。其中包括57个任务，涵盖了人文学科、社会科学、自然科学和其他对某些人学习很重要的领域。
- CMMLU：CMMLU 是一个综合性的中文评估基准，涵盖了从基础学科到高级专业水平的67个主题。涵盖了自然科学、人文科学和社会科学等领域。
- AGIEVAL：AGIEval 是一个专门为评估基础模型在以人类为中心的标准化考试（如大学入学考试、法学院入学考试、数学竞赛和律师资格考试）的语境中而设计的基准测试。
- ARC：包含了ARC-E和ARC-C，它们分别是ARC数据集中的简单集和挑战集，分别有5197 和2590 个问题。这些问题是仅文本的英语语言考试问题，跨越了多个年级水平。
- GaoKao：GaoKao收集了从 2010 年到 2022 年的高考试题，包括 1781 道客观题和 1030 道主观题。本项目报告结果为GaoKao中客观题结果。
- FinanceIQ：FinanceIQ 是一个专注于金融领域的中文评估数据集，涵盖了10个金融大类及36个金融小类，总计7173个单项选择题。

### 测评结果
|                       | C_EVAL | MMLU  | CMMLU | AGIEVAL | ARC_E | ARC_C | GaoKao | FinanceIQ | 平均  |
| :-------------------: | ------ | ----- | :---: | :-----: | :---: | :---: | :----: | :-------: | :---: |
| iFlytekSpark-13B-base | 70.88  | 58.76 | 70.01 |  50.44  | 84.78 | 71.16 | 56.42  |   60.21   | 65.33 |
| iFlytekSpark-13B-chat | 82.54  | 63.02 | 75.69 |  56.96  | 89.47 | 77.34 | 67.49  |   65.48   | 72.25 |


# 使用教程
### NPU
[华为昇腾NPU上使用教程](https://gitee.com/mindspore/mindformers/blob/r1.0/research/iflytekspark/iflytekspark.md)

### GPU
敬请期待...


### 工具链
##### 自我认知
[自我认知工具链使用教程](self-awareness/README.md)


# 声明、联系我们

### 协议

请您知悉，无论您是否已实际阅读[星火开源-13B大模型许可协议](LICENSE_MODEL.md)，当您通过部署及使用该模型服务即表示确认同意本协议或实际使用、复制、分发、修改本协议中的讯飞星火认知大模型-13B模型时，均表示您与科大讯飞股份有限公司（以下称“许可方”）已就本协议达成一致，本协议具有合同效力。如果您不同意本协议的任一内容，或者无法准确理解许可方对[本协议条款](LICENSE_MODEL.md)的解释，请停止使用本服务。否则，即表示您已接受本协议所述的所有条款及其适用条件，同意受本协议约束。

### 联系我们

如果你想给我们的研发团队和产品团队留言，可以通过邮件（iflytekspark@iflymail.com.cn）联系我们。