import os
import typer
from typing import Optional, List, Tuple
from enum import Enum
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from docx import Document
from rich.console import Console
from rich.table import Table
from rich import print as rprint
from pathlib import Path

# 可用的AIGC检测模型列表
class ModelType(str, Enum):
    ENGLISH = "english"
    CHINESE = "chinese"
    DESKLIB = "desklib"

# 模型配置
MODEL_CONFIGS = {
    ModelType.ENGLISH: ("yuchuantian/AIGC_detector_env2", "AIGC_text_detector/AIGC_en_model"),
    ModelType.CHINESE: ("yuchuantian/AIGC_detector_zhv2", "AIGC_text_detector/AIGC_zh_model"),
    ModelType.DESKLIB: ("desklib/ai-text-detector-v1.01", "desklib_detector/desklib_model"),
}

app = typer.Typer()
console = Console()

def load_model(model_type: ModelType):
    """加载指定类型的AIGC检测模型"""
    model_id, _ = MODEL_CONFIGS[model_type]
    try:
        console.print(f"正在加载模型 [bold]{model_id}[/bold]...")
        return pipeline("text-classification", model=model_id)
    except Exception as e:
        console.print(f"[bold red]模型加载失败: {e}[/bold red]")
        raise typer.Exit(code=1)

def read_text_file(file_path: str) -> str:
    """读取文本文件内容"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        # 尝试其他编码
        with open(file_path, "r", encoding="gbk") as f:
            return f.read()

def read_docx_file(file_path: str) -> str:
    """读取Word文档内容"""
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

def read_file_content(file_path: str) -> str:
    """根据文件扩展名读取文件内容"""
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == ".txt":
        return read_text_file(file_path)
    elif file_extension == ".docx":
        return read_docx_file(file_path)
    else:
        console.print(f"[bold red]不支持的文件格式: {file_extension}[/bold red]")
        raise typer.Exit(code=1)

def analyze_text(text: str, pipe) -> dict:
    """分析文本是否由AI生成"""
    if not text.strip():
        return {"label": "UNKNOWN", "score": 0.0}
    
    # 如果文本太长，分段处理
    if len(text) > 500:
        chunks = [text[i:i+500] for i in range(0, len(text), 500)]
        results = [pipe(chunk)[0] for chunk in chunks]
        
        # 计算平均得分
        human_score = sum(r['score'] for r in results if r['label'] == 'Human') / len(results)
        ai_score = sum(r['score'] for r in results if r['label'] == 'AI') / len(results)
        
        if human_score > ai_score:
            return {"label": "Human", "score": human_score}
        else:
            return {"label": "AI", "score": ai_score}
    else:
        return pipe(text)[0]

@app.command()
def detect(
    file_path: List[str] = typer.Argument(..., help="文件路径，支持.txt和.docx格式"),
    model_type: ModelType = typer.Option(ModelType.CHINESE, "--model", "-m", help="AIGC检测模型类型"),
    threshold: float = typer.Option(0.7, "--threshold", "-t", help="AI内容判定阈值，大于此值则判定为AI生成")
):
    """检测文档中的内容是否由AI生成"""
    
    # 加载模型
    pipe = load_model(model_type)
    
    # 创建结果表格
    table = Table(title=f"AIGC检测结果 (使用模型: {MODEL_CONFIGS[model_type][0]})")
    table.add_column("文件名", style="cyan")
    table.add_column("判定结果", style="green")
    table.add_column("置信度", style="yellow")
    table.add_column("状态", style="magenta")
    
    for path in file_path:
        try:
            file_name = os.path.basename(path)
            console.print(f"正在分析文件: [bold]{file_name}[/bold]")
            
            # 读取文件内容
            content = read_file_content(path)
            
            # 分析内容
            result = analyze_text(content, pipe)
            
            # 判定结果
            if result["label"] == "AI" and result["score"] >= threshold:
                status = "[bold red]AI生成[/bold red]"
            else:
                status = "[bold green]人类创作[/bold green]"
                
            table.add_row(
                file_name,
                result["label"],
                f"{result['score']:.4f}",
                status
            )
            
        except Exception as e:
            table.add_row(file_name, "错误", "0", f"[bold red]处理失败: {str(e)}[/bold red]")
    
    console.print(table)

def main():
    app()

if __name__ == "__main__":
    main()