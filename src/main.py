# %% src/main.py

# 内置库
import os
import typer
import numpy as np
from typing import List, Dict
from enum import Enum
import torch
import torch.nn as nn
from transformers import (
    pipeline,
    AutoTokenizer,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    BertForSequenceClassification,
    BertTokenizer,
    PreTrainedModel,
    AutoModel,
    AutoConfig,
)
from docx import Document
from rich.console import Console
from rich.table import Table
import logging
import glob
import sys
import re

# 配置日志记录，用于跟踪程序运行状态和错误信息
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# 定义支持的AIGC检测模型类型
class ModelType(str, Enum):
    ENGLISH = "english"  # 英文AIGC检测模型
    CHINESE = "chinese"  # 中文AIGC检测模型
    DESKLIB = "desklib"  # Desklib特定检测模型
    MULTI = "multi"  # 多语言检测模型


# 模型配置字典，映射模型类型到对应的模型路径
MODEL_CONFIGS = {
    ModelType.ENGLISH: ("yuchuantian/AIGC_detector_env2", "AIGC_text_detector/AIGC_en_model"),
    ModelType.CHINESE: ("yuchuantian/AIGC_detector_zhv2", "AIGC_text_detector/AIGC_zh_model"),
    ModelType.DESKLIB: ("desklib/ai-text-detector-v1.01", "desklib_detector/desklib_model"),
}

# 初始化Typer应用，用于命令行接口
app = typer.Typer()
# 初始化Rich控制台，用于美化输出
console = Console()


# Desklib AI检测模型的自定义实现
class DesklibAIDetectionModel(PreTrainedModel):
    """基于Transformers的Desklib AI检测模型"""

    config_class = AutoConfig  # 配置类，用于模型初始化

    def __init__(self, config):
        super().__init__(config)
        self.model = AutoModel.from_config(config)  # 加载预训练模型
        self.classifier = nn.Linear(config.hidden_size, 1)  # 分类头，输出单一概率
        self.init_weights()  # 初始化模型权重

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        前向传播，处理输入并返回预测结果
        :param input_ids: 输入的token ID序列
        :param attention_mask: 输入的注意力掩码
        :param labels: 可选的标签，用于计算损失
        :return: 包含logits和可选损失的字典
        """
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        # 池化操作：根据注意力掩码对隐藏状态加权平均
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        logits = self.classifier(pooled_output)  # 分类预测
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()  # 二分类损失函数
            loss = loss_fct(logits.view(-1), labels.float())
        output = {"logits": logits}
        if loss is not None:
            output["loss"] = loss
        return output


# AIGC检测器类，用于加载和运行多种AI生成内容检测模型
class AIGCDetector:
    """支持多语言和多模型的AIGC检测器"""

    def __init__(self, use_gpu=True):
        """
        初始化AIGC检测器，设置设备并加载模型
        :param use_gpu: 是否使用GPU加速（默认True）
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        self.models = {}  # 存储加载的模型
        self.tokenizers = {}  # 存储对应的分词器
        self.id2labels = {}  # 模型标签映射
        self.thresholds = {
            "en": [0.60867064, 0.85145506],  # 英文模型阈值
            "zh-cn": [0.60867064, 0.85145506],  # 中文模型阈值
        }
        self._init_models()  # 初始化所有模型

    def _init_models(self):
        """加载中英文及Desklib检测模型，处理加载过程中的异常"""
        try:
            console.print("正在加载中文AIGC检测模型...")
            self._init_zh_model()
        except Exception as e:
            console.print(f"[yellow]中文AIGC模型加载失败: {str(e)}[/yellow]")
        try:
            console.print("正在加载英文AIGC检测模型...")
            self._init_en_model()
        except Exception as e:
            console.print(f"[yellow]英文AIGC模型加载失败: {str(e)}[/yellow]")
        try:
            console.print("正在加载Desklib检测模型...")
            self._init_desklib_model()
        except Exception as e:
            console.print(f"[yellow]Desklib模型加载失败: {str(e)}[/yellow]")

    def _init_zh_model(self):
        """
        初始化中文AIGC检测模型，从本地或远程加载
        """
        model_id, local_path = MODEL_CONFIGS[ModelType.CHINESE]
        try:
            if os.path.exists(local_path):
                self.models["zh"] = BertForSequenceClassification.from_pretrained(local_path)
                self.tokenizers["zh"] = BertTokenizer.from_pretrained(local_path)
            else:
                self.models["zh"] = BertForSequenceClassification.from_pretrained(model_id)
                self.tokenizers["zh"] = BertTokenizer.from_pretrained(model_id)
                os.makedirs(local_path, exist_ok=True)
                self.models["zh"].save_pretrained(local_path)
                self.tokenizers["zh"].save_pretrained(local_path)
            self.models["zh"].to(self.device)
            self.id2labels["zh"] = ["Human", "AI"]
            console.print("[green]中文AIGC模型加载成功[/green]")
        except Exception as e:
            console.print(f"[red]中文AIGC模型加载失败: {str(e)}[/red]")
            raise

    def _init_en_model(self):
        """
        初始化英文AIGC检测模型，从本地或远程加载
        """
        model_id, local_path = MODEL_CONFIGS[ModelType.ENGLISH]
        try:
            if os.path.exists(local_path):
                self.models["en"] = RobertaForSequenceClassification.from_pretrained(local_path)
                self.tokenizers["en"] = RobertaTokenizer.from_pretrained(local_path)
            else:
                self.models["en"] = RobertaForSequenceClassification.from_pretrained(model_id)
                self.tokenizers["en"] = RobertaTokenizer.from_pretrained(model_id)
                os.makedirs(local_path, exist_ok=True)
                self.models["en"].save_pretrained(local_path)
                self.tokenizers["en"].save_pretrained(local_path)
            self.models["en"].to(self.device)
            self.id2labels["en"] = ["Human", "AI"]
            console.print("[green]英文AIGC模型加载成功[/green]")
        except Exception as e:
            console.print(f"[red]英文AIGC模型加载失败: {str(e)}[/red]")
            raise

    def _init_desklib_model(self):
        """
        初始化Desklib检测模型，从本地或远程加载
        """
        model_id, local_path = MODEL_CONFIGS[ModelType.DESKLIB]
        try:
            if os.path.exists(local_path):
                self.tokenizers["desklib"] = AutoTokenizer.from_pretrained(local_path)
                config = AutoConfig.from_pretrained(local_path)
                self.models["desklib"] = DesklibAIDetectionModel.from_pretrained(local_path)
            else:
                self.tokenizers["desklib"] = AutoTokenizer.from_pretrained(model_id)
                self.models["desklib"] = DesklibAIDetectionModel.from_pretrained(model_id)
                os.makedirs(local_path, exist_ok=True)
                self.models["desklib"].save_pretrained(local_path)
                self.tokenizers["desklib"].save_pretrained(local_path)
            self.models["desklib"].to(self.device)
            console.print("[green]Desklib模型加载成功[/green]")
        except Exception as e:
            console.print(f"[red]Desklib模型加载失败: {str(e)}[/red]")
            raise

    def _aigc_predict(self, text: str, lang: str) -> float:
        """
        使用指定语言的AIGC模型进行预测
        :param text: 输入文本
        :param lang: 语言标识（"en"或"zh"）
        :return: AI生成内容的概率
        """
        if lang not in self.models:
            return 0.5
        tokenizer = self.tokenizers[lang]
        model = self.models[lang]
        id2label = self.id2labels.get(lang, ["Human", "AI"])
        model.eval()
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = model(**inputs)
            scores = outputs.logits[0].softmax(0).cpu().numpy()
            res = {"label": scores.argmax().item(), "score": scores.max().item()}
            return res["score"] if id2label[res["label"]] == "AI" else 1 - res["score"]

    def _desklib_predict(self, text: str) -> float:
        """
        使用Desklib模型进行预测
        :param text: 输入文本
        :return: AI生成内容的概率
        """
        if "desklib" not in self.models:
            return 0.5
        tokenizer = self.tokenizers["desklib"]
        model = self.models["desklib"]
        encoded = tokenizer(
            text, padding="max_length", truncation=True, max_length=768, return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            return torch.sigmoid(outputs["logits"]).cpu().item()

    def _calculate_score(self, sample_probs: List[float], thresholds: List[float]) -> Dict:
        """
        根据模型预测概率和阈值计算最终得分
        :param sample_probs: 各模型的预测概率
        :param thresholds: 对应模型的阈值
        :return: 包含标签和概率的字典
        """
        sample_probs = np.asarray(sample_probs)
        thresholds = np.asarray(thresholds)
        exceeds = sample_probs >= thresholds
        if np.any(exceeds):
            label = "AI"
            exceed_indices = np.where(exceeds)[0]
            scores = []
            weights = []
            for i in exceed_indices:
                prob = sample_probs[i]
                thresh = thresholds[i]
                if thresh >= 1.0:
                    score = 1.0
                else:
                    score = (prob - thresh) / (1.0 - thresh)
                    score = np.clip(score, 0.0, 1.0)
                scores.append(score)
                weights.append(score)
            total_weight = np.sum(weights)
            if total_weight > 1e-6:
                weighted_scores = np.sum(np.array(scores) * np.array(weights)) / total_weight
            else:
                weighted_scores = 0.0
            ai_prob = 50.0 + 50.0 * weighted_scores
        else:
            label = "Human"
            sample_dist = np.linalg.norm(sample_probs)
            threshold_dist = np.linalg.norm(thresholds)
            if threshold_dist < 1e-6:
                ai_prob = 50.0
            else:
                ratio = sample_dist / threshold_dist
                ai_prob = 50.0 * np.clip(ratio, 0.0, 1.0)
        return {"label": label, "probability": float(round(ai_prob, 2)) / 100}

    def detect(self, text: str, language: str) -> Dict:
        """
        检测输入文本是否为AI生成内容
        :param text: 输入文本
        :param language: 语言类型（"en", "zh-cn", 或其他）
        :return: 包含预测标签和概率的字典
        """
        if language == "en":
            aigc_score = self._aigc_predict(text, "en")
            desklib_score = self._desklib_predict(text)
            thresholds = self.thresholds["en"]
            return self._calculate_score(
                [aigc_score, desklib_score], [thresholds[0], thresholds[1]]
            )
        elif language == "zh-cn":
            aigc_score = self._aigc_predict(text, "zh")
            logger.info(f"AIGC中文模型得分: {aigc_score}")
            desklib_score = self._desklib_predict(text)
            logger.info(f"Desklib模型得分: {desklib_score}")
            thresholds = self.thresholds["zh-cn"]
            return self._calculate_score([aigc_score, desklib_score], [thresholds[0], 0.95])
        else:
            aigc_en_score = self._aigc_predict(text, "en")
            aigc_zh_score = self._aigc_predict(text, "zh")
            desklib_score = self._desklib_predict(text)
            if aigc_en_score > aigc_zh_score:
                thresholds = self.thresholds["en"]
                return self._calculate_score(
                    [aigc_en_score, desklib_score], [thresholds[0], thresholds[1]]
                )
            else:
                thresholds = self.thresholds["zh-cn"]
                return self._calculate_score([aigc_zh_score, desklib_score], [thresholds[0], 0.95])


def load_model(model_type: ModelType):
    """
    加载指定类型的AIGC检测模型，支持在线和本地加载
    :param model_type: 模型类型（英文、中文、Desklib或多模型）
    :return: 加载的模型实例（pipeline或AIGCDetector）
    """
    if model_type == ModelType.MULTI:
        console.print("正在初始化多模型检测器...")
        try:
            return AIGCDetector()
        except Exception as e:
            console.print(f"[bold red]多模型检测器初始化失败: {str(e)}[/bold red]")
            console.print("[yellow]尝试使用单一模型回退...[/yellow]")
            return load_model(ModelType.CHINESE)
    model_id, local_path = MODEL_CONFIGS[model_type]
    local_model_exists = os.path.exists(local_path)
    try:
        console.print(f"正在加载模型 [bold]{model_id}[/bold]...")
        return pipeline("text-classification", model=model_id)
    except Exception as e:
        console.print(f"[yellow]在线模型加载失败: {str(e)}[/yellow]")
        if local_model_exists:
            try:
                console.print(f"尝试从本地加载模型 [bold]{local_path}[/bold]...")
                return pipeline("text-classification", model=local_path)
            except Exception as local_e:
                console.print(f"[bold red]本地模型加载失败: {str(local_e)}[/bold red]")
        if model_type == ModelType.DESKLIB:
            console.print("[yellow]尝试使用中文AIGC检测模型作为备选...[/yellow]")
            return load_model(ModelType.CHINESE)
        console.print(f"[bold red]所有模型加载尝试均失败，无法继续检测[/bold red]")
        raise typer.Exit(code=1)


def read_text_file(file_path: str) -> str:
    """
    读取文本文件内容，支持多种编码
    :param file_path: 文件路径
    :return: 文件内容字符串
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, "r", encoding="gbk") as f:
                return f.read()
        except UnicodeDecodeError:
            console.print(f"[bold red]无法解码文件: {file_path}，请检查编码[/bold red]")
            raise typer.Exit(code=1)


def read_docx_file(file_path: str) -> str:
    """
    读取Word文档内容
    :param file_path: Word文档路径
    :return: 文档内容字符串
    """
    try:
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    except Exception as e:
        console.print(f"[bold red]读取 .docx 文件失败: {str(e)}[/bold red]")
        raise typer.Exit(code=1)


def read_file_content(file_path: str) -> str:
    """
    根据文件扩展名选择适当的方法读取文件内容
    :param file_path: 文件路径
    :return: 文件内容字符串
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == ".txt":
        return read_text_file(file_path)
    elif file_extension == ".docx":
        return read_docx_file(file_path)
    else:
        console.print(
            f"[bold red]不支持的文件格式: {file_extension}，仅支持 .txt 和 .docx[/bold red]"
        )
        raise typer.Exit(code=1)


def analyze_text(text: str, detector) -> dict:
    """
    分析文本是否为AI生成内容，处理长文本分块和多语言检测
    :param text: 输入文本
    :param detector: AIGC检测器或pipeline实例
    :return: 包含标签和分数的字典
    """
    if not text.strip():
        return {"label": "UNKNOWN", "score": 0.0}
    try:
        # 判断文本语言（中文占比超过10%视为中文）
        chinese_chars = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
        is_chinese = chinese_chars / len(text) > 0.1 if text else False
        language = "zh-cn" if is_chinese else "en"
        is_aigc_detector = isinstance(detector, AIGCDetector)
        # 根据模型类型设置最大序列长度
        max_seq_len = 512
        if is_aigc_detector:
            if language == "zh-cn" and "zh" in detector.tokenizers:
                max_seq_len = 512
            elif language == "en" and "en" in detector.tokenizers:
                max_seq_len = 512
            elif "desklib" in detector.tokenizers:
                max_seq_len = 512
        else:
            model_id = detector.model.name_or_path
            if "AIGC_en_model" in model_id or "AIGC_detector_env2" in model_id:
                max_seq_len = 500
            else:
                max_seq_len = 500
        # 处理长文本分块
        if len(text) > max_seq_len * 0.7:
            paragraphs = text.split("\n")
            chunks = []
            current_chunk = ""
            for para in paragraphs:
                if not para.strip():
                    continue
                if len(current_chunk) + len(para) < max_seq_len * 0.7:
                    current_chunk += para + "\n"
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = ""
                    if len(para) > max_seq_len * 0.7:
                        sentences = para.split(". ")
                        temp_chunk = ""
                        for sentence in sentences:
                            if not sentence.strip():
                                continue
                            if len(temp_chunk) + len(sentence) < max_seq_len * 0.7:
                                temp_chunk += sentence + ". "
                            else:
                                if temp_chunk:
                                    chunks.append(temp_chunk.strip())
                                if len(sentence) > max_seq_len * 0.7:
                                    words = sentence.split()
                                    word_chunk = ""
                                    for word in words:
                                        if len(word_chunk) + len(word) < max_seq_len * 0.7:
                                            word_chunk += word + " "
                                        else:
                                            if word_chunk:
                                                chunks.append(word_chunk.strip())
                                            word_chunk = word + " "
                                    if word_chunk:
                                        chunks.append(word_chunk.strip())
                                else:
                                    temp_chunk = sentence + ". "
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                    else:
                        current_chunk = para + "\n"
            if current_chunk:
                chunks.append(current_chunk.strip())
            if not chunks:
                chunk_size = int(max_seq_len * 0.5)
                chunks = []
                for i in range(0, len(text), chunk_size):
                    chunk = text[i : i + chunk_size]
                    if chunk.strip():
                        chunks.append(chunk)
            # 确保分块大小安全
            safe_chunks = []
            for chunk in chunks:
                if len(chunk) > max_seq_len * 0.8:
                    sub_chunks = [
                        chunk[i : i + int(max_seq_len * 0.5)]
                        for i in range(0, len(chunk), int(max_seq_len * 0.5))
                    ]
                    safe_chunks.extend([sc for sc in sub_chunks if sc.strip()])
                else:
                    safe_chunks.append(chunk)
            chunks = safe_chunks
            results = []
            # 对每个分块进行预测
            for chunk in chunks:
                try:
                    if is_aigc_detector:
                        chunk_result = detector.detect(chunk, language)
                    else:
                        pipeline_result = detector(chunk)
                        label = pipeline_result[0]["label"]
                        score = pipeline_result[0]["score"]
                        chunk_result = {
                            "label": label,
                            "probability" if is_aigc_detector else "score": score,
                        }
                    results.append(chunk_result)
                except Exception as e:
                    logger.warning(f"处理文本块时发生错误: {str(e)}")
                    if len(chunk) > 200:
                        try:
                            shorter_chunk = chunk[:200]
                            if is_aigc_detector:
                                short_result = detector.detect(shorter_chunk, language)
                            else:
                                pipeline_result = detector(shorter_chunk)
                                label = pipeline_result[0]["label"]
                                score = pipeline_result[0]["score"]
                                short_result = {
                                    "label": label,
                                    "probability" if is_aigc_detector else "score": score,
                                }
                            results.append(short_result)
                            logger.info("使用缩短块成功处理")
                        except Exception as inner_e:
                            logger.warning(f"处理缩短块仍然失败: {str(inner_e)}")
                    continue
            if not results:
                return {"label": "UNKNOWN", "score": 0.0}
            # 汇总分块结果
            ai_count = sum(1 for r in results if r["label"] == "AI")
            human_count = len(results) - ai_count
            if ai_count >= human_count * 0.5:
                score_key = "probability" if is_aigc_detector else "score"
                avg_score = sum(r.get(score_key, 0.0) for r in results if r["label"] == "AI") / max(
                    ai_count, 1
                )
                return {"label": "AI", score_key: avg_score}
            else:
                score_key = "probability" if is_aigc_detector else "score"
                avg_score = 1 - sum(
                    r.get(score_key, 0.0) for r in results if r["label"] == "Human"
                ) / max(human_count, 1)
                return {"label": "Human", score_key: avg_score}
        else:
            # 直接处理短文本
            if is_aigc_detector:
                return detector.detect(text, language)
            else:
                result = detector(text)
                label = result[0]["label"]
                score = result[0]["score"]
                return {"label": label, "score": score}
    except Exception as e:
        logger.error(f"分析文本时发生错误: {str(e)}")
        return {"label": "ERROR", "score": 0.0}


def generate_report(file_name: str, result: dict, status: str, output_dir: str):
    """
    生成检测报告并保存到指定路径
    :param file_name: 文件名
    :param result: 检测结果字典
    :param status: 判定状态
    :param output_dir: 报告输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "report.txt")
    score_key = "probability" if result.get("probability") is not None else "score"
    score = result.get(score_key, 0.0)
    label = result.get("label", "UNKNOWN")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"文件名: {file_name}\n")
        f.write(f"判定结果: {label}\n")
        f.write(f"置信度: {score:.4f}\n")
        f.write(f"状态: {status}\n")
    console.print(f"[green]报告已保存到: {report_path}[/green]")


@app.command()
def detect(
    file_path: List[str] = typer.Argument(..., help="文件路径，支持.txt和.docx格式"),
    model_type: ModelType = typer.Option(ModelType.MULTI, "--model", "-m", help="AIGC检测模型类型"),
    threshold: float = typer.Option(
        0.7, "--threshold", "-t", help="AI内容判定阈值，大于此值则判定为AI生成"
    ),
):
    try:
        detector = load_model(model_type)
        console.print("[green]模型加载成功[/green]")
    except Exception as e:
        console.print(f"[bold red]模型加载失败: {str(e)}，无法继续[/bold red]")
        raise typer.Exit(code=1)

    # 初始化结果展示表格
    table = Table(title=f"AIGC检测结果 (使用模型: {model_type.value})", show_lines=True)
    table.add_column("文件名", style="cyan")
    table.add_column("判定结果", style="green")
    table.add_column("置信度", style="yellow")
    table.add_column("状态", style="magenta")

    # 创建 res_detect 目录
    res_detect_dir = os.path.join(os.getcwd(), "res_detect")
    os.makedirs(res_detect_dir, exist_ok=True)

    # 遍历输入文件进行检测
    for path in file_path:
        try:
            file_name = os.path.basename(path)
            # 获取文件名（不含扩展名）作为子目录
            file_base_name = os.path.splitext(file_name)[0]
            report_dir = os.path.join(res_detect_dir, file_base_name, "report")

            # 检查是否已有报告
            if os.path.exists(report_dir):
                console.print(f"[yellow]文件 {file_name} 的报告已存在，跳过检测[/yellow]")
                try:
                    with open(os.path.join(report_dir, "report.txt"), "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        if len(lines) < 4:
                            raise ValueError("报告文件格式不正确")
                        label = lines[1].split(": ")[1].strip()
                        score = float(lines[2].split(": ")[1].strip())
                        status = lines[3].split(": ")[1].strip()
                    # 清理 status 中的可能标记
                    status = re.sub(r"\[/?\w+\]", "", status).strip()
                    # 动态设置颜色，避免标记冲突
                    status_display = f"[{ 'red' if status == 'AI生成' else 'green' } bold]{status}[/{ 'red' if status == 'AI生成' else 'green' } bold]"
                    table.add_row(file_name, label, f"{score:.4f}", status_display)
                except Exception as e:
                    table.add_row(
                        file_name, "错误", "0", f"[bold red]读取报告失败: {str(e)}[/bold red]"
                    )
                continue

            console.print(f"正在分析文件: [bold]{file_name}[/bold]")
            content = read_file_content(path)
            if model_type == ModelType.MULTI:
                result = analyze_text(content, detector)
                score = result.get("probability", 0.0)
                label = result.get("label", "UNKNOWN")
            else:
                result = analyze_text(content, detector)
                score = result.get("score", 0.0)
                label = result.get("label", "UNKNOWN")

            # 根据阈值判断最终状态
            if label == "AI" and score >= threshold:
                status = "AI生成"
            else:
                status = "人类创作"

            # 生成并保存报告
            generate_report(file_name, result, status, report_dir)

            # 添加到结果表格，使用正确的标记语法
            status_display = f"[{ 'red' if status == 'AI生成' else 'green' } bold]{status}[/{ 'red' if status == 'AI生成' else 'green' } bold]"
            table.add_row(file_name, label, f"{score:.4f}", status_display)

        except Exception as e:
            table.add_row(
                os.path.basename(path), "错误", "0", f"[bold red]处理失败: {str(e)}[/bold red]"
            )

    console.print(table)


def main():
    """
    主函数，处理命令行调用或默认目录检测逻辑
    """
    # 检查是否通过命令行运行detect命令
    if len(sys.argv) > 1 and sys.argv[1] == "detect":
        app()
    else:
        # 默认检测to_detect目录中的文件
        detect_dir = os.path.join(os.getcwd(), "to_detect")
        os.makedirs(detect_dir, exist_ok=True)
        default_files = glob.glob(os.path.join(detect_dir, "*.txt")) + glob.glob(
            os.path.join(detect_dir, "*.docx")
        )
        default_files = [os.path.relpath(f, os.getcwd()) for f in default_files]
        if not default_files:
            console.print("[bold red]to_detect 目录中未找到任何 .txt 或 .docx 文件！[/bold red]")
            raise typer.Exit(code=1)
        default_model = ModelType.MULTI
        default_threshold = 0.7
        console.print("[bold blue]运行 AIGC 检测工具...[/bold blue]")
        console.print(f"检测文件: {default_files}")
        detect(file_path=default_files, model_type=default_model, threshold=default_threshold)


if __name__ == "__main__":
    main()
