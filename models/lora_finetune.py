# -*- coding: utf-8 -*-
"""
NLLB-200 LoRA 微调训练脚本
专注于专业词汇和多义词的精准翻译
"""

import os
import json
import torch
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset
import evaluate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LoraFineTuneConfig:
    """LoRA 微调配置"""
    # 模型配置
    base_model: str = "facebook/nllb-200-distilled-600M"
    lora_output_dir: str = "models/lora_checkpoints"
    
    # LoRA 参数（核心）
    lora_r: int = 16  # LoRA 秩（越大效果越好，但训练越慢）
    lora_alpha: int = 32  # LoRA 缩放系数（通常是 r 的 2 倍）
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj",  # Query 投影层
        "v_proj",  # Value 投影层
        "k_proj",  # Key 投影层
        "out_proj",  # 输出投影层
        "fc1",  # FFN 第一层
        "fc2",  # FFN 第二层
    ])
    
    # 训练超参数
    num_epochs: int = 10
    learning_rate: float = 3e-4
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    
    # 早停策略
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    
    # 数据路径
    train_data_path: str = "data/finetune/train/data.json"
    eval_data_path: str = "data/finetune/eval/data.json"
    
    # 设备配置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    fp16: bool = torch.cuda.is_available()


class NLLBLoraTrainer:
    """NLLB LoRA 微调训练器"""
    
    def __init__(self, config: LoraFineTuneConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.bleu_metric = evaluate.load("sacrebleu")
        
    def load_base_model(self):
        """加载基础 NLLB 模型"""
        logger.info(f"加载基础模型: {self.config.base_model}")
        
        # 加载 tokenizer 并设置源语言和目标语言
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            src_lang="eng_Latn",  # 英语作为源语言
            tgt_lang="zho_Hans"   # 中文作为目标语言
        )
        
        # 加载模型（使用 8-bit 量化节省显存）
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.base_model,
            load_in_8bit=True if self.config.device == "cuda" else False,
            device_map="auto" if self.config.device == "cuda" else None,
            torch_dtype=torch.float16 if self.config.fp16 else torch.float32
        )
        
        logger.info("✅ 基础模型加载完成")
    
    def apply_lora(self):
        """应用 LoRA 适配器"""
        logger.info("应用 LoRA 配置...")
        
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        logger.info("✅ LoRA 配置应用完成")
    
    def load_datasets(self) -> Dict[str, Dataset]:
        """加载训练和验证数据集"""
        logger.info("加载数据集...")
        
        # 加载 JSON 数据
        with open(self.config.train_data_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        
        with open(self.config.eval_data_path, 'r', encoding='utf-8') as f:
            eval_data = json.load(f)
        
        # 转换为 Hugging Face Dataset
        train_dataset = Dataset.from_list(train_data)
        eval_dataset = Dataset.from_list(eval_data)
        
        # Tokenize
        def preprocess_function(examples):
            # 源语言编码
            inputs = self.tokenizer(
                examples["src"],
                max_length=88,
                truncation=True,
                padding="max_length"
            )
            
            # 目标语言编码（修复：直接使用 tokenizer，不使用 as_target_tokenizer）
            labels = self.tokenizer(
                examples["tgt"],
                max_length=88,
                truncation=True,
                padding="max_length"
            )
            
            inputs["labels"] = labels["input_ids"]
            return inputs
        
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=eval_dataset.column_names
        )
        
        logger.info(f"✅ 数据集加载完成: Train={len(train_dataset)}, Eval={len(eval_dataset)}")
        
        return {"train": train_dataset, "eval": eval_dataset}
    
    def compute_metrics(self, eval_preds):
        """计算 BLEU 分数"""
        preds, labels = eval_preds
        
        # 解码预测
        if isinstance(preds, tuple):
            preds = preds[0]
        
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        
        # 处理标签（替换 -100 为 pad_token_id）
        labels = [[l if l != -100 else self.tokenizer.pad_token_id for l in label] for label in labels]
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # 计算 BLEU
        result = self.bleu_metric.compute(
            predictions=decoded_preds,
            references=[[label] for label in decoded_labels]
        )
        
        return {"bleu": result["score"]}
    
    def train(self):
        """执行微调训练"""
        # 1. 加载模型和数据
        self.load_base_model()
        self.apply_lora()
        datasets = self.load_datasets()
        
        # 2. 训练参数
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.config.lora_output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            max_grad_norm=self.config.max_grad_norm,
            fp16=self.config.fp16,
            
            # 评估策略
            eval_strategy="steps",
            eval_steps=750,
            save_strategy="steps",
            save_steps=750,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="bleu",
            greater_is_better=True,
            
            # 日志
            logging_steps=300,
            logging_dir=f"{self.config.lora_output_dir}/logs",
            report_to="none",
            
            # 生成配置
            predict_with_generate=True,
            generation_max_length=88,
            generation_num_beams=4,
        )
        
        # 3. 数据整理器
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            padding=True
        )
        
        # 4. 早停回调
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=self.config.early_stopping_patience,
            early_stopping_threshold=self.config.early_stopping_threshold
        )
        
        # 5. 初始化 Trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["eval"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[early_stopping]
        )
        
        # 6. 开始训练
        logger.info("🚀 开始 LoRA 微调训练...")
        train_result = trainer.train()
        
        # 7. 保存最终模型
        logger.info("保存微调后的 LoRA 模型...")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.lora_output_dir)
        
        # 8. 保存训练指标
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        logger.info("✅ LoRA 微调训练完成！")
        logger.info(f"   模型保存路径: {self.config.lora_output_dir}")
        
        return metrics


# ============================================================
# 使用示例
# ============================================================
if __name__ == "__main__":
    # 配置
    config = LoraFineTuneConfig(
        base_model="facebook/nllb-200-1.3B",
        lora_output_dir="models/lora_nllb_terminology",
        num_epochs=8,
        batch_size=8,  # 如果显存不足，降低到 2
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        lora_r=16,
        lora_alpha=32,
        warmup_steps=50,  # 热身步数减少
    )
    
    # 训练
    trainer = NLLBLoraTrainer(config)
    metrics = trainer.train()
    
    print("\n" + "="*50)
    print("🎉 训练完成！最终指标:")
    print(f"   Loss: {metrics.get('train_loss', 'N/A'):.4f}")
    print("="*50)