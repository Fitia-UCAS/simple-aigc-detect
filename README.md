# Simple AIGC Detect

一个简单易用的AI生成内容检测工具，可以帮助您识别文档中可能由AI生成的内容。

## 环境管理

本项目使用 [uv](https://github.com/astral-sh/uv) 作为Python包管理和运行工具。uv是一个快速的Python包安装器和解析器，由Rust编写，可大幅提升依赖项安装速度。

## 功能特点

- 支持中文和英文内容的AIGC检测
- 支持多种文件格式（.txt和.docx）
- 提供友好的命令行界面和彩色输出
- 支持多种预训练模型选择
- 可自定义AI内容判定阈值

## 使用方法

基本用法：

```bash
uv run .\src\main.py 文件路径 [选项]
```

### 选项参数

- `-m, --model`: 选择检测模型类型，可选值：`chinese`（默认）、`english`、`desklib`
- `-t, --threshold`: AI内容判定阈值，默认为0.7（大于此值则判定为AI生成）

### 使用示例

检测单个文件：

```bash
uv run .\src\main.py .\example.txt
```

检测多个文件：

```bash
uv run .\src\main.py file1.txt file2.docx
```

使用英文检测模型：

```bash
uv run .\src\main.py .\example.txt --model english
```

自定义判定阈值：

```bash
uv run .\src\main.py .\example.txt --threshold 0.8
```

## 支持的模型

- `chinese`: 中文AIGC检测模型
- `english`: 英文AIGC检测模型
- `desklib`: Desklib通用检测模型

## 依赖项

- Python 3.8+
- transformers
- torch
- python-docx
- typer
- rich
- huggingface-hub

## 许可证

MIT

## 贡献指南

欢迎提交问题和合并请求，共同改进这个项目！