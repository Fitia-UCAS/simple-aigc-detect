# Simple AIGC Detect

一个简单易用的AI生成内容检测工具，可以帮助您识别文档中可能由AI生成的内容。

## 环境管理

本项目使用 [uv](https://github.com/astral-sh/uv) 作为Python包管理和运行工具。uv是一个快速的Python包安装器和解析器，由Rust编写，可大幅提升依赖项安装速度。

### 安装uv

如果您还没有安装uv，可以通过以下命令安装：

```bash
curl -sSf https://install.ultraviolet.rs | sh
```

或在Windows上使用PowerShell：

```powershell
iwr https://install.ultraviolet.rs/ps1 | iex
```

### 使用uv安装项目依赖

克隆仓库后，使用uv安装依赖：

```bash
uv pip install -e .
```

## 功能特点

- 支持中文和英文内容的AIGC检测
- 多模型融合检测，提高准确率
- 支持多种文件格式（.txt和.docx）
- 自动语言检测，针对不同语言选择最佳模型
- 支持长文本分块处理
- 提供友好的命令行界面和彩色输出
- 支持多种预训练模型选择
- 可自定义AI内容判定阈值
- 优化的多模型结果融合算法，对不同语言优化处理

## 使用方法

运行 main.py 文件，检测 to_detect 文件夹下所有符合的文件，输出报告到终端以及 res_detect 文件夹下。

## 支持的模型

- `multi`: 多模型组合检测（推荐），综合多个模型结果提高准确率
- `chinese`: 中文AIGC检测模型
- `english`: 英文AIGC检测模型
- `desklib`: Desklib通用检测模型

## 模型原理

本工具使用了多种预训练模型进行AI生成内容检测：

1. **AIGC中文模型**：基于BERT的中文AI内容检测模型
2. **AIGC英文模型**：基于RoBERTa的英文AI内容检测模型
3. **Desklib模型**：通用AI文本检测模型

多模型检测模式下，工具会同时使用多个模型进行评估，并通过加权投票算法综合各模型的检测结果，提高检测的准确性和鲁棒性。对于不同语言类型的内容，系统会自动调整不同模型的权重，优先考虑针对该语言优化的模型结果。

## 依赖项

- Python 3.8+
- transformers
- torch
- python-docx
- typer
- rich
- huggingface-hub
- numpy

## 首次运行

首次运行时，工具会自动从Hugging Face下载预训练模型，并缓存到本地。这可能需要一些时间，请耐心等待。后续运行将使用本地缓存的模型，启动更快。

## 许可证

MIT

## 贡献指南

欢迎提交问题和合并请求，共同改进这个项目！