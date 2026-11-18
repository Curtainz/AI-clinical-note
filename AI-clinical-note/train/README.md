# 模型训练（离线研究轨道）

本目录对应 README 中的 **`[ ] 模型训练`** 待办项。

## 为什么是「离线轨道」

竞赛任务书（任务 1 / 任务 2）规定：正式评测由评测机在关闭网络的环境中运行
`python main.py`，模型通过组织方提供的 OpenAI 兼容端点调用，**不允许替换或改写
部署模型权重**，也不允许在运行时加载私有模型。因此：

- `main.py` 的提交路径**不依赖本目录**，不 import torch/transformers/peft；
- 本目录提供一条可复现的 **SFT / LoRA 微调**流程，用于本地研究、prompt 迭代和
  「改进验证」中的模型侧对比；
- 训练产物（adapter）只作为研究记录与 ablation 依据，不会被提交运行环境加载。

## 一、构造数据集

`tools/build_sft_dataset.py` 把「案例输入 → 参考首程记录」转成 chat 格式 JSONL：

```powershell
# data/synth 为 100 例公开合成数据（input.json + reference.json）
python tools/build_sft_dataset.py `
  --input data/synth/input.json `
  --reference data/synth/reference.json `
  --out-dir data/sft
```

输出：

| 文件 | 内容 |
| --- | --- |
| `train.jsonl` | 90% 案例：`messages = [system(prompts/active.txt), user(案例 JSON), assistant(参考 medical_record JSON)]` |
| `val.jsonl` | 10% 案例（按 case_id 的 sha256 确定性切分，重复运行结果一致） |
| `dataset_stats.json` | 样本数、跳过原因、平均答案长度 |

切分盐值可用 `--split-salt` 修改；`--val-ratio 0` 可只产出训练集。
`reference.json` 仅供本训练轨道使用——竞赛评分不读取它，人工对比时也只作参考。

## 二、安装训练依赖

与主程序环境隔离安装（主程序只依赖 `requirements.txt`）：

```powershell
python -m venv .venv-train
.\.venv-train\Scripts\pip.exe install -r requirements-train.txt
```

- CUDA GPU：Qwen3-4B + QLoRA（4-bit，`--load-in-4bit` 默认开启）约需 **8 GB 显存**；
  bf16 LoRA（`--no-4bit`）约需 **16 GB 显存**。
- 纯 CPU 训练不可行；`--allow-cpu` 只留给 3 条样本级别的最小冒烟验证。
- `bitsandbytes` 无官方 Windows wheel：Windows 上请用 `--no-4bit`，或在 WSL/Linux 下训练。

## 三、训练

```powershell
python train/train_lora.py `
  --train-file data/sft/train.jsonl `
  --val-file data/sft/val.jsonl `
  --output-dir checkpoints/qwen3-4b-lora
```

默认配置（均可在命令行覆盖）：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model` | `Qwen/Qwen3-4B-Instruct-2507` | 与 README 中部署模型同族 |
| `--epochs` | 2 | |
| `--batch-size` / `--grad-accum` | 1 / 8 | 等效 batch 8 |
| `--lr` | 1e-4 | LoRA 常用学习率 |
| `--lora-r` / `--lora-alpha` | 16 / 32 | `q,k,v,o,gate,up,down` 全投影 |
| `--max-seq-len` | 4096 | 超长样本截断 |
| `--gradient-checkpointing` | 开 | 省显存 |

产物：`output_dir/` 下的 LoRA adapter 权重、tokenizer、`training_config.json`。

没有 GPU 时也可以先验证数据与配置（不会 import torch）：

```powershell
python train/train_lora.py --dry-check `
  --train-file data/sft/train.jsonl --val-file data/sft/val.jsonl `
  --output-dir checkpoints/qwen3-4b-lora
```

## 四、评估（baseline vs improved）

`train/evaluate.py` 在**同一份 `val.jsonl`** 上分别评测基座模型与 adapter，
并与 `main.py` 使用同一套校验代码（`src.validate` + `src.content_check`），
保证「改进验证」口径一致：

```powershell
# baseline：基座模型
python train/evaluate.py --val-file data/sft/val.jsonl --output reports/base_eval.json

# improved：加载 LoRA adapter
python train/evaluate.py --val-file data/sft/val.jsonl `
  --adapter checkpoints/qwen3-4b-lora --output reports/lora_eval.json
```

报告指标：

- `parse_rate`：输出能否解析为 `medical_record` JSON；
- `schema_pass_rate`：是否通过 `validate_medical_record`（5 字段 / evidence 结构 /
  source_id 引用合法）；
- `content_pass_rate` / `average_content_score`：`check_record` 的内容校验；
- `average_untraceable_numbers`：正文中无法回溯到资料的数值个数（越低越好）；
- `average_evidence_items` / `average_chars`：引用条数与篇幅。

若需要**运行级**前后对比（同一批病例两次 `main.py` 输出，带收益/局限文字），
用离线对比工具：

```powershell
python tools/check_output.py `
  --input input.json --output output_grounded.json `
  --baseline-output output_baseline.json --report reports/comparison.json
```

## 五、从运行结果回流训练数据（data flywheel）

除了 `data/synth` 合成数据，也可以把一次真实运行的**通过校验的结果**回流成训练集，形成
「生成 → 校验 → 回流 → 训练 → 再评估」的闭环：

```powershell
python tools/collect_training_data.py `
  --input input.json --output output.json `
  --report validation_report.json --out-dir data/collected
```

输出：

| 文件 | 内容 |
| --- | --- |
| `reference.json` | 通过过滤的 `case_id → medical_record`，可直接喂给 `build_sft_dataset.py --reference` |
| `train.jsonl` / `val.jsonl` | 同 `data/sft` 的 chat 格式（system 为 `prompts/active.txt`） |
| `provenance.json` | 每条样本的内容分、checklist 路径、run_id、输入/输出文件 sha256 |
| `dataset_stats.json` | 接受/拒绝数量及拒绝原因、切分盐值、平均长度 |

过滤规则（任一不满足即拒绝，原因写入 `dataset_stats.json`）：

1. `output.json` 中该例 `status == "success"`；
2. 仍通过 `validate_medical_record`（五字段 / evidence / `source_ids` 真实存在）；
3. 内容分 ≥ `--min-score`（默认 80，可用 `--min-score 0` 生成冒烟数据）；
4. 若该例有 checklist，按 checklist 一起打分；清单本身损坏则拒绝。

回流后训练：

```powershell
python train/train_lora.py --train-file data/collected/train.jsonl `
  --val-file data/collected/val.jsonl --output-dir checkpoints/collected-lora
```

**注意**：回流样本仍是模型输出（即使经过自审与规则校验），不是人工金标准；`provenance.json` 只保证
可追溯到具体 run，不证明临床正确性。正式成绩仍以评测机上的基座模型为准。

## 六、预期收益与局限（诚实说明）

**预期收益**

- prompt 微调（`prompts/grounded.txt`）已能针对性降低「未提供写成正常」「计划写成已实施」
  等错误；SFT 可进一步把参考首程的**结构密度**（三段篇幅、证据条数）对齐到数据分布；
- LoRA 训练成本低（4B + 4-bit），可在单卡上完成，便于快速做 prompt×模型 ablation。

**局限**

- `data/synth/reference.json` 是合成数据参考记录，**不是人工金标准**，用它训练可能
  学到合成模板的措辞偏好，对真实病历的泛化未经证实；`tools/collect_training_data.py`
  回流的真实运行数据同样来自模型本身，只能提高一致性，不能替代专家标注；
- 合成数据未覆盖真实病历的噪声（口语缩写、跨科转科、模板化阴性描述），
  训练收益不能直接外推到竞赛真实病例；
- 正式评测禁止替换部署权重，因此训练收益**不会计入正式成绩**，只作为研究分析；
- 内容校验器（`src/content_check.py`）是规则系统，对同义改写与复杂否定范围的
  召回有限，评估结论应结合人工复核（`checklist` 的 `clinical_review_status`
  当前为 `not_expert_reviewed`）。
