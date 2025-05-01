import os
import typer
import numpy as np
from typing import List, Dict
from enum import Enum
import torch
import torch.nn as nn
from transformers import (
    pipeline, AutoTokenizer, 
    RobertaForSequenceClassification, RobertaTokenizer,
    BertForSequenceClassification, BertTokenizer,
    PreTrainedModel, AutoModel, AutoConfig
)
from docx import Document
from rich.console import Console
from rich.table import Table
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 可用的AIGC检测模型列表
class ModelType(str, Enum):
    ENGLISH = "english"
    CHINESE = "chinese"
    DESKLIB = "desklib"
    MULTI = "multi"  # 多模型组合检测

# 模型配置，每个模型包括在线模型ID和本地保存路径
MODEL_CONFIGS = {
    ModelType.ENGLISH: ("yuchuantian/AIGC_detector_env2", "AIGC_text_detector/AIGC_en_model"),
    ModelType.CHINESE: ("yuchuantian/AIGC_detector_zhv2", "AIGC_text_detector/AIGC_zh_model"),
    ModelType.DESKLIB: ("desklib/ai-text-detector-v1.01", "desklib_detector/desklib_model"),
}

app = typer.Typer()
console = Console()

class DesklibAIDetectionModel(PreTrainedModel):
    """Desklib AI检测模型实现"""
    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        # 初始化基础transformer模型
        self.model = AutoModel.from_config(config)
        # 定义分类头
        self.classifier = nn.Linear(config.hidden_size, 1)
        # 初始化权重
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None):
        # transformer前向传播
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        # 平均池化
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask

        # 分类器
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.float())

        output = {"logits": logits}
        if loss is not None:
            output["loss"] = loss
        return output

class AIGCDetector:
    """AIGC检测器，支持多模型检测"""
    def __init__(self, use_gpu=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        self.models = {}
        self.tokenizers = {}
        self.id2labels = {}
        
        # 默认阈值
        self.thresholds = {
            "en": [0.60867064, 0.85145506],
            "zh-cn": [0.60867064, 0.85145506]
        }
        
        # 初始化模型
        self._init_models()
    
    def _init_models(self):
        """初始化所有可用模型"""
        # 尝试加载中文AIGC模型
        try:
            console.print("正在加载中文AIGC检测模型...")
            self._init_zh_model()
        except Exception as e:
            console.print(f"[yellow]中文AIGC模型加载失败: {str(e)}[/yellow]")
        
        # 尝试加载英文AIGC模型
        try:
            console.print("正在加载英文AIGC检测模型...")
            self._init_en_model()
        except Exception as e:
            console.print(f"[yellow]英文AIGC模型加载失败: {str(e)}[/yellow]")
        
        # 尝试加载Desklib模型
        try:
            console.print("正在加载Desklib检测模型...")
            self._init_desklib_model()
        except Exception as e:
            console.print(f"[yellow]Desklib模型加载失败: {str(e)}[/yellow]")

    def _init_zh_model(self):
        """初始化中文模型"""
        model_id, local_path = MODEL_CONFIGS[ModelType.CHINESE]
        
        try:
            if os.path.exists(local_path):
                self.models["zh"] = BertForSequenceClassification.from_pretrained(local_path)
                self.tokenizers["zh"] = BertTokenizer.from_pretrained(local_path)
            else:
                self.models["zh"] = BertForSequenceClassification.from_pretrained(model_id)
                self.tokenizers["zh"] = BertTokenizer.from_pretrained(model_id)
                
                # 保存模型到本地
                os.makedirs(local_path, exist_ok=True)
                self.models["zh"].save_pretrained(local_path)
                self.tokenizers["zh"].save_pretrained(local_path)
                
            self.models["zh"].to(self.device)
            self.id2labels["zh"] = ['Human', 'AI']
            console.print("[green]中文AIGC模型加载成功[/green]")
        except Exception as e:
            console.print(f"[red]中文AIGC模型加载失败: {str(e)}[/red]")
            raise

    def _init_en_model(self):
        """初始化英文模型"""
        model_id, local_path = MODEL_CONFIGS[ModelType.ENGLISH]
        
        try:
            if os.path.exists(local_path):
                self.models["en"] = RobertaForSequenceClassification.from_pretrained(local_path)
                self.tokenizers["en"] = RobertaTokenizer.from_pretrained(local_path)
            else:
                self.models["en"] = RobertaForSequenceClassification.from_pretrained(model_id)
                self.tokenizers["en"] = RobertaTokenizer.from_pretrained(model_id)
                
                # 保存模型到本地
                os.makedirs(local_path, exist_ok=True)
                self.models["en"].save_pretrained(local_path)
                self.tokenizers["en"].save_pretrained(local_path)
                
            self.models["en"].to(self.device)
            self.id2labels["en"] = ['Human', 'AI']
            console.print("[green]英文AIGC模型加载成功[/green]")
        except Exception as e:
            console.print(f"[red]英文AIGC模型加载失败: {str(e)}[/red]")
            raise

    def _init_desklib_model(self):
        """初始化Desklib模型"""
        model_id, local_path = MODEL_CONFIGS[ModelType.DESKLIB]
        
        try:
            if os.path.exists(local_path):
                self.tokenizers["desklib"] = AutoTokenizer.from_pretrained(local_path)
                config = AutoConfig.from_pretrained(local_path)
                self.models["desklib"] = DesklibAIDetectionModel.from_pretrained(local_path)
            else:
                self.tokenizers["desklib"] = AutoTokenizer.from_pretrained(model_id)
                self.models["desklib"] = DesklibAIDetectionModel.from_pretrained(model_id)
                
                # 保存模型到本地
                os.makedirs(local_path, exist_ok=True)
                self.models["desklib"].save_pretrained(local_path)
                self.tokenizers["desklib"].save_pretrained(local_path)
                
            self.models["desklib"].to(self.device)
            console.print("[green]Desklib模型加载成功[/green]")
        except Exception as e:
            console.print(f"[red]Desklib模型加载失败: {str(e)}[/red]")
            raise
    
    def _aigc_predict(self, text: str, lang: str) -> float:
        """AIGC模型预测函数"""
        if lang not in self.models:
            return 0.5
            
        tokenizer = self.tokenizers[lang]
        model = self.models[lang]
        id2label = self.id2labels.get(lang, ['Human', 'AI'])
        
        model.eval()
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = model(**inputs)
            scores = outputs.logits[0].softmax(0).cpu().numpy()
            res = {"label": scores.argmax().item(), "score": scores.max().item()}
            return res["score"] if id2label[res['label']] == 'AI' else 1 - res["score"]

    def _desklib_predict(self, text: str) -> float:
        """Desklib模型预测"""
        if "desklib" not in self.models:
            return 0.5
            
        tokenizer = self.tokenizers["desklib"]
        model = self.models["desklib"]
        
        encoded = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=768,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)

        model.eval()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            return torch.sigmoid(outputs["logits"]).cpu().item()

    def _calculate_score(self, sample_probs: List[float], thresholds: List[float]) -> Dict:
        """
        根据样本概率和阈值向量判断标签并计算AI生成概率
        
        参数：
        sample_probs : 模型输出的概率值
        thresholds : 模型对应的阈值
        """
        sample_probs = np.asarray(sample_probs)
        thresholds = np.asarray(thresholds)
        
        # 判断标签
        exceeds = sample_probs >= thresholds
        if np.any(exceeds):
            label = 'AI'
            # 计算AI概率（存在超过阈值的维度）
            exceed_indices = np.where(exceeds)[0]
            scores = []
            weights = []
            
            for i in exceed_indices:
                prob = sample_probs[i]
                thresh = thresholds[i]
                
                # 处理阈值等于1的特殊情况
                if thresh >= 1.0:
                    score = 1.0
                else:
                    score = (prob - thresh) / (1.0 - thresh)
                    score = np.clip(score, 0.0, 1.0)
                
                scores.append(score)
                weights.append(score)
                
            # 计算加权平均
            total_weight = np.sum(weights)
            if total_weight > 1e-6:
                weighted_scores = np.sum(np.array(scores) * np.array(weights)) / total_weight
            else:
                weighted_scores = 0.0
            
            # 线性映射到50%-100%
            ai_prob = 50.0 + 50.0 * weighted_scores
            
        else:
            label = 'Human'
            # 计算AI概率（无超过阈值的维度）
            sample_dist = np.linalg.norm(sample_probs)
            threshold_dist = np.linalg.norm(thresholds)
            
            if threshold_dist < 1e-6:
                ai_prob = 50.0
            else:
                ratio = sample_dist / threshold_dist
                ai_prob = 50.0 * np.clip(ratio, 0.0, 1.0)
            
        return {
            "label": label,
            "probability": float(round(ai_prob, 2))/100
        }
    
    def detect(self, text: str, language: str) -> Dict:
        """
        统一检测接口
        :param text: 待检测文本
        :param language: 语言类型 (en/zh-cn)
        :return: 检测结果
        """
        if language == "en":
            aigc_score = self._aigc_predict(text, "en")
            desklib_score = self._desklib_predict(text)
            thresholds = self.thresholds["en"] 
            # 英文使用两个模型
            return self._calculate_score(
                [aigc_score, desklib_score],
                [thresholds[0], thresholds[1]]
            )
        elif language == "zh-cn":
            aigc_score = self._aigc_predict(text, "zh")
            logger.info(f"AIGC中文模型得分: {aigc_score}")
            # Desklib模型对中文支持有限，使用较高阈值
            desklib_score = self._desklib_predict(text)
            logger.info(f"Desklib模型得分: {desklib_score}")
            thresholds = self.thresholds["zh-cn"]
            return self._calculate_score(
                [aigc_score, desklib_score],
                [thresholds[0], 0.95]
            )
        else:
            # 未知语言，尝试双语模型
            aigc_en_score = self._aigc_predict(text, "en")
            aigc_zh_score = self._aigc_predict(text, "zh")
            desklib_score = self._desklib_predict(text)
            
            # 使用最高置信度的结果
            if aigc_en_score > aigc_zh_score:
                thresholds = self.thresholds["en"]
                return self._calculate_score(
                    [aigc_en_score, desklib_score],
                    [thresholds[0], thresholds[1]]
                )
            else:
                thresholds = self.thresholds["zh-cn"]
                return self._calculate_score(
                    [aigc_zh_score, desklib_score],
                    [thresholds[0], 0.95]
                )

def load_model(model_type: ModelType):
    """加载指定类型的AIGC检测模型，带有失败处理和本地回退机制"""
    if model_type == ModelType.MULTI:
        console.print("正在初始化多模型检测器...")
        try:
            return AIGCDetector()
        except Exception as e:
            console.print(f"[bold red]多模型检测器初始化失败: {str(e)}[/bold red]")
            console.print("[yellow]尝试使用单一模型回退...[/yellow]")
            return load_model(ModelType.CHINESE)
    
    model_id, local_path = MODEL_CONFIGS[model_type]
    
    # 检查是否存在本地模型
    local_model_exists = os.path.exists(local_path)
    
    # 尝试加载模型，优先从在线模型加载
    try:
        console.print(f"正在加载模型 [bold]{model_id}[/bold]...")
        return pipeline("text-classification", model=model_id)
    except Exception as e:
        console.print(f"[yellow]在线模型加载失败: {str(e)}[/yellow]")
        
        # 如果在线加载失败且本地模型存在，尝试从本地加载
        if local_model_exists:
            try:
                console.print(f"尝试从本地加载模型 [bold]{local_path}[/bold]...")
                return pipeline("text-classification", model=local_path)
            except Exception as local_e:
                console.print(f"[bold red]本地模型加载失败: {str(local_e)}[/bold red]")
        
        # 如果是desklib模型加载失败，尝试使用中文模型作为备选
        if model_type == ModelType.DESKLIB:
            console.print("[yellow]尝试使用中文AIGC检测模型作为备选...[/yellow]")
            return load_model(ModelType.CHINESE)
        
        console.print(f"[bold red]所有模型加载尝试均失败，无法继续检测[/bold red]")
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

def analyze_text(text: str, detector) -> dict:
    """分析文本是否由AI生成"""
    if not text.strip():
        return {"label": "UNKNOWN", "score": 0.0}
    
    try:
        # 识别文本语言
        # 这里用简单方法判断，实际项目中可以用更复杂的语言检测
        chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
        is_chinese = chinese_chars / len(text) > 0.1 if text else False
        language = "zh-cn" if is_chinese else "en"
        
        # 检查是否是AIGCDetector类型或pipeline类型
        is_aigc_detector = isinstance(detector, AIGCDetector)
        
        # 获取模型的最大序列长度
        max_seq_len = 512  # 默认值
        if is_aigc_detector:
            # 针对不同语言和模型设定合适的最大序列长度
            if language == "zh-cn" and "zh" in detector.tokenizers:
                # 中文BERT模型
                max_seq_len = 512  # 中文BERT标准长度
            elif language == "en" and "en" in detector.tokenizers:
                # 英文RoBERTa模型
                max_seq_len = 512  # 略小于配置的514以确保安全
            elif "desklib" in detector.tokenizers:
                # Desklib模型
                max_seq_len = 512  # 根据配置限制
        else:
            # 对于pipeline类型，从模型配置判断
            # 判断模型类型
            model_id = detector.model.name_or_path
            if "AIGC_en_model" in model_id or "AIGC_detector_env2" in model_id:
                # 英文RoBERTa模型限制为小于514
                max_seq_len = 500
            else:
                # 中文或其他模型
                max_seq_len = 500  # 安全值
        
        # 如果文本太长，分段处理，考虑模型最大序列长度
        if len(text) > max_seq_len * 0.7:  # 仅使用70%以确保安全
            # 更智能的分块策略，根据段落和句子进行分块
            paragraphs = text.split('\n')
            chunks = []
            current_chunk = ""
            
            for para in paragraphs:
                if not para.strip():
                    continue
                    
                if len(current_chunk) + len(para) < max_seq_len * 0.7:
                    current_chunk += para + "\n"
                else:
                    # 如果当前段落太长，按句子分割
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = ""
                        
                    if len(para) > max_seq_len * 0.7:
                        sentences = para.split('. ')
                        temp_chunk = ""
                        for sentence in sentences:
                            if not sentence.strip():
                                continue
                                
                            if len(temp_chunk) + len(sentence) < max_seq_len * 0.7:
                                temp_chunk += sentence + ". "
                            else:
                                if temp_chunk:
                                    chunks.append(temp_chunk.strip())
                                # 如果单句超长，进一步分割
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
                
            # 如果分块失败或块数为0，使用更安全的固定长度分块
            if not chunks:
                chunk_size = int(max_seq_len * 0.5)  # 使用50%长度确保安全
                chunks = []
                for i in range(0, len(text), chunk_size):
                    chunk = text[i:i+chunk_size]
                    if chunk.strip():
                        chunks.append(chunk)
            
            # 验证所有分块长度
            safe_chunks = []
            for chunk in chunks:
                if len(chunk) > max_seq_len * 0.8:
                    # 太长的块再次分割
                    sub_chunks = [chunk[i:i+int(max_seq_len*0.5)] for i in range(0, len(chunk), int(max_seq_len*0.5))]
                    safe_chunks.extend([sc for sc in sub_chunks if sc.strip()])
                else:
                    safe_chunks.append(chunk)
            
            chunks = safe_chunks
            results = []
            
            for chunk in chunks:
                try:
                    if is_aigc_detector:
                        # 使用AIGCDetector的detect方法
                        chunk_result = detector.detect(chunk, language)
                    else:
                        # 使用pipeline的__call__方法
                        pipeline_result = detector(chunk)
                        label = pipeline_result[0]['label']
                        score = pipeline_result[0]['score']
                        chunk_result = {"label": label, "probability" if is_aigc_detector else "score": score}
                    results.append(chunk_result)
                except Exception as e:
                    logger.warning(f"处理文本块时发生错误: {str(e)}")
                    # 尝试进一步缩短
                    if len(chunk) > 200:
                        try:
                            shorter_chunk = chunk[:200]  # 使用更短的块再试一次
                            if is_aigc_detector:
                                short_result = detector.detect(shorter_chunk, language)
                            else:
                                pipeline_result = detector(shorter_chunk)
                                label = pipeline_result[0]['label']
                                score = pipeline_result[0]['score']
                                short_result = {"label": label, "probability" if is_aigc_detector else "score": score}
                            results.append(short_result)
                            logger.info("使用缩短块成功处理")
                        except Exception as inner_e:
                            logger.warning(f"处理缩短块仍然失败: {str(inner_e)}")
                    continue
            
            if not results:
                return {"label": "UNKNOWN", "score": 0.0}
            
            # 计算加权平均得分
            ai_count = sum(1 for r in results if r['label'] == 'AI')
            human_count = len(results) - ai_count
            
            # AI优先判断策略
            if ai_count >= human_count * 0.5:  # 如果有超过一半的块被判断为AI，则整体判为AI
                score_key = "probability" if is_aigc_detector else "score"
                avg_score = sum(r.get(score_key, 0.0) for r in results if r['label'] == 'AI') / max(ai_count, 1)
                return {"label": "AI", score_key: avg_score}
            else:
                score_key = "probability" if is_aigc_detector else "score"
                avg_score = 1 - sum(r.get(score_key, 0.0) for r in results if r['label'] == 'Human') / max(human_count, 1)
                return {"label": "Human", score_key: avg_score}
        else:
            # 短文本直接分析
            if is_aigc_detector:
                # 使用AIGCDetector的detect方法
                return detector.detect(text, language)
            else:
                # 使用pipeline的__call__方法
                result = detector(text)
                label = result[0]['label']
                score = result[0]['score']
                return {"label": label, "score": score}
            
    except Exception as e:
        logger.error(f"分析文本时发生错误: {str(e)}")
        return {"label": "ERROR", "score": 0.0}

@app.command()
def detect(
    file_path: List[str] = typer.Argument(..., help="文件路径，支持.txt和.docx格式"),
    model_type: ModelType = typer.Option(ModelType.MULTI, "--model", "-m", help="AIGC检测模型类型"),
    threshold: float = typer.Option(0.7, "--threshold", "-t", help="AI内容判定阈值，大于此值则判定为AI生成")
):
    """检测文档中的内容是否由AI生成"""
    
    # 加载模型
    try:
        detector = load_model(model_type)
        console.print("[green]模型加载成功[/green]")
    except Exception as e:
        console.print(f"[bold red]模型加载失败: {str(e)}，无法继续[/bold red]")
        raise typer.Exit(code=1)
    
    # 创建结果表格
    table = Table(title=f"AIGC检测结果 (使用模型: {model_type.value})")
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
            if model_type == ModelType.MULTI:
                # 使用AIGCDetector
                result = analyze_text(content, detector)
                score = result.get("probability", 0.0)
                label = result.get("label", "UNKNOWN")
            else:
                # 使用pipeline
                result = analyze_text(content, detector)
                score = result.get("score", 0.0)
                label = result.get("label", "UNKNOWN")
            
            # 判定结果
            if label == "AI" and score >= threshold:
                status = "[bold red]AI生成[/bold red]"
            else:
                status = "[bold green]人类创作[/bold green]"
                
            table.add_row(
                file_name,
                label,
                f"{score:.4f}",
                status
            )
            
        except Exception as e:
            table.add_row(
                os.path.basename(path), 
                "错误", 
                "0", 
                f"[bold red]处理失败: {str(e)}[/bold red]"
            )
    
    console.print(table)

def main():
    app()

if __name__ == "__main__":
    main()