# -*- coding: utf-8 -*-
"""
LoRA 模型质量对比测试
用于验证微调效果是否符合预期
"""

from translator_with_lora import LoraEnhancedTranslator
from models.translator import NeuralTranslator  # 原始模型
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LoraQualityTester:
    """LoRA 模型质量测试器"""
    
    def __init__(self):
        # 加载两个模型进行对比
        logger.info("加载基础模型...")
        self.base_translator = NeuralTranslator(
            nmt_model_id="facebook/nllb-200-distilled-600M",
            device='cuda'
        )
        
        logger.info("加载 LoRA 微调模型...")
        self.lora_translator = LoraEnhancedTranslator(
            base_model_id="facebook/nllb-200-distilled-600M",
            lora_model_path="models/lora_nllb_terminology",
            device='cuda'
        )
    
    def get_test_cases(self) -> List[Tuple[str, str]]:
        """返回测试用例 (原文, 期望译文)"""
        return [
            # === 鸟类术语 ===
            ("A Sandhill Crane flew over the lake.", "一只沙丘鹤飞过湖面。"),
            ("The Whooping Crane is endangered.", "美洲鹤濒临灭绝。"),
            
            # === 多义词：Night (人名 vs 时间) ===
            ("Night won the championship.", "Night 赢得了冠军。"),
            ("It was a dark night.", "那是一个黑暗的夜晚。"),
            
            # === 多义词：Crane (鹤 vs 起重机) ===
            ("A crane lifted the cargo.", "起重机吊起了货物。"),
            ("We saw a crane in the wetlands.", "我们在湿地看到了一只鹤。"),
            
            # === 专业术语 ===
            ("Myocardial infarction is serious.", "心肌梗死很严重。"),
            ("The plaintiff filed a lawsuit.", "原告提起了诉讼。"),
        ]
    
    def compare_models(self):
        """对比两个模型的翻译质量"""
        test_cases = self.get_test_cases()
        
        print("\n" + "="*80)
        print("📊 LoRA 微调效果对比测试")
        print("="*80)
        
        for i, (source, expected) in enumerate(test_cases, 1):
            # 基础模型翻译
            base_segments = [{"start": 0.0, "end": 3.0, "text": source}]
            base_result = self.base_translator.translate_segments(
                base_segments, target_lang="zh-cn", source_lang="en"
            )
            base_trans = base_result[0]["text"]
            
            # LoRA 模型翻译
            lora_result = self.lora_translator.translate_segments(
                base_segments, target_lang="zh-cn", source_lang="en"
            )
            lora_trans = lora_result[0]["text"]
            
            # 输出对比
            print(f"\n测试用例 {i}:")
            print(f"原文:     {source}")
            print(f"期望译文: {expected}")
            print(f"基础模型: {base_trans}")
            print(f"LoRA模型: {lora_trans}")
            
            # 简单匹配检测（生产环境应使用 BLEU 等指标）
            base_match = self._fuzzy_match(base_trans, expected)
            lora_match = self._fuzzy_match(lora_trans, expected)
            
            print(f"基础模型匹配度: {'✓' if base_match > 0.7 else '✗'} ({base_match:.2f})")
            print(f"LoRA模型匹配度: {'✓' if lora_match > 0.7 else '✗'} ({lora_match:.2f})")
            
            if lora_match > base_match:
                print("🎉 LoRA 模型表现更好！")
            elif lora_match < base_match:
                print("⚠️ LoRA 模型未改进，需要调整训练数据")
            else:
                print("➖ 两个模型效果相当")
        
        print("\n" + "="*80)
    
    def _fuzzy_match(self, text1: str, text2: str) -> float:
        """简单的模糊匹配（字符级相似度）"""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, text1, text2).ratio()


# ============================================================
# 运行测试
# ============================================================
if __name__ == "__main__":
    tester = LoraQualityTester()
    tester.compare_models()