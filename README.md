# 项目运行与配置说明

##  当前进度

* **已实现**：读取 `./input.json` 内容，传入基于 Bionic 的本地模型进行推理，并将结果输出至 `./output.json`。
* **尚待实现**：
  * [ ] 模型训练
  * [ ] 输出内容校验

---

##  环境与模型配置

### 1. 基础环境
* **Bionic (LM Studio)**：`1.2.1`
* **Python**：`3.11.9`
* **本地部署模型**：`qwen3-4b-instruct-2507`
* **基础 URL**：`http://localhost:1235/v1`

### 2. 推理参数设置

| 参数项 | 配置值 |
| :--- | :--- |
| **硬件优化** | `Automatic Optimize Based on Hardware` |
| **CPU Thread Pool Size** | `8` |
| **Temperature** | `0.7` |
| **Top K** | `20` |
| **Top P** | `0.8` |
| **Min P** | `0` |
| **Seed** | `42` |

---

##  快速运行测试

在 PowerShell 中依次执行以下命令以准备测试数据并运行主程序：

```powershell
# 1. 复制示例输入文件
Copy-Item -LiteralPath '.\examples\DEMO_THREE_CASES.input.json' -Destination '.\input.json'

# 2. 激活基线提示词模板
Copy-Item -LiteralPath '.\prompts\baseline.txt' -Destination '.\prompts\active.txt'

# 3. 使用虚拟环境运行主脚本
& .\.venv\Scripts\python.exe .\main.py
