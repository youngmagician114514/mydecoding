# slot_level_llama2_7B 说明

## 1) 三个目录对比结论

- `token_level_llama2_7B` 与 `version1`
  - `model.py`、`eval_humaneval.py`、`eval_humaneval_edd.py`、`trace_accept_edd_only.py`、`modeling_llama.py` 一致。
  - 主要差异在 `train.py`：`version1` 对训练流程做了小调整（如保存步长默认值和加载时机），核心“token-level memory”逻辑不变。

- `version2` 相比前两者
  - `model.py` 引入 `TimeEmbed + slot_size` 固定槽位逻辑，并新增 `slot_cache`（`init_slot_cache/update_slot_cache`）。
  - 训练和评测脚本改为与 slot 逻辑兼容的路径（含增量 cache 更新）。

## 2) 本目录的构建方式

`slot_level_llama2_7B` 以 `token_level_llama2_7B` 为底座（保持 LLaMA2-7B 使用方式），并移植 `version2` 的 slot 逻辑：

- 采用 `version2` 的 `model.py`（含 `TimeEmbed`、固定 `slot_size`、`slot_cache`）。
- 采用 `version2` 的训练/评测主脚本，并补齐 LLaMA2 场景参数。

## 3) 本目录新增/调整

- `train.py`
  - 新增参数：
    - `--slot_size`（默认 5）
    - `--layer_dropout`（默认 0.0）
    - `--slot_dropout`（默认 0.0）
  - 构造 `Effective_Draft_Decoder` 时传入上述参数。

- `eval_humaneval.py`
  - 新增 `--slot_size`，并在加载草稿模型时传入。

- `eval_humaneval_edd.py`
  - 新增 `--slot_size`，并在加载草稿模型时传入。
  - `init_slot_cache` 改为使用模型内部 `slot_size`，不再与 `draft_len` 强绑定。

- `trace_accept_edd_only.py`
  - 新增 `--slot_size`，并在加载草稿模型时传入。
  - `init_slot_cache` 使用 `edd.slot_size`。

- `batch_eval_checkpoints.py`
  - 新增 `--slot_size`，并透传给 `eval_humaneval.py` / `eval_humaneval_edd.py`。

## 4) 使用建议

- 训练与推理需保持同一 `slot_size`（例如都用 `--slot_size 5`）。
- 如果使用旧权重（无 `time_embed.*` 参数），不能直接严格加载到该目录的 slot-level 模型结构。
