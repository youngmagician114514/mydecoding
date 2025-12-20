# mydecoding：基于 Llama-3 的信念融合推测解码

本仓库实现 PDF 里的两头推测解码思路：head1 负责起草候选，fusion 将候选 token + 置信度 + 历史上下文融合成潜变量 z_{t+1}，head2 在 z_{t+1} 条件下自回归打分并决定接受/回退。整体思路参考 Hydra/Medusa，但增加了信念门控融合。

## 目录结构
- `mydecoding/config.py`：`DualDecoderConfig`，集中超参。
- `mydecoding/model.py`：`DualDecoderModel`，封装 head1 + fusion + head2，含推测解码 `generate`。
- `mydecoding/fusion.py`：信念门控融合模块（候选自注意力 + belief gating + 历史汇聚）。
- `mydecoding/cli.py`：命令行推理入口。
- `mydecoding/training.py`：最小化训练 step 和 dataloader 辅助。

## 快速开始
```bash
pip install -e .
python -m mydecoding.cli --model meta-llama/Meta-Llama-3-8B --prompt "解释什么是推测解码" --max-new-tokens 64
```
说明：
- 基座 Llama 冻结，只训练 head1/head2/fusion。
- 接受规则：若基座对草稿 token 的概率 > `config.acceptance_threshold`，则直接接受，否则回退到基座贪心 token。
- 损失：head1 用交叉熵（草稿），head2 用交叉熵（融合），权重见 `DualDecoderConfig`。

## 训练示例
```python
from mydecoding.config import DualDecoderConfig
from mydecoding.model import DualDecoderModel
from mydecoding.training import train_step, build_dataloader
from datasets import load_dataset
import torch.optim as optim

config = DualDecoderConfig()
model = DualDecoderModel(config)
tokenizer = model.tokenizer
dataset = load_dataset("tatsu-lab/alpaca", split="train[:1%]")
dataloader = build_dataloader(dataset, tokenizer, batch_size=1)
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)

for step, batch in enumerate(dataloader):
    loss = train_step(model, batch, optimizer)
    if step % 10 == 0:
        print("step", step, "loss", loss)
```

## 与 Hydra 的关系
- head1 使用前缀 MLP 起草 top-k，类似 Hydra/Medusa 的草稿头。
- fusion 融合三路候选（A/B/C）、置信度与历史隐状态，形成 `z_{t+1}`（对应 PDF 草图）。
- head2 是小型 Transformer decoder，在 `z_{t+1}` 上打分/生成；最终由冻结基座做接受验证。
