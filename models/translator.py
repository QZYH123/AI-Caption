import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from peft import PeftModel  # <--- 新增引入
import logging
import gc
from typing import Optional, Dict, Any, List
from sentence_transformers import SentenceTransformer, util
import re
import os

# 确保 offload 文件夹存在
os.makedirs("offload_nllb", exist_ok=True)
logger = logging.getLogger(__name__)


class NeuralTranslator:
    def __init__(self, nmt_model_id="facebook/nllb-200-distilled-600M", 
                 reflection_model_id=None, 
                 lora_model_id=None,  # <--- 新增参数：LoRA 模型路径
                 device='cpu'):
        
        # 自动检测并设置 GPU 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.nmt_tokenizer = None
        self.nmt_model = None
        self.reflector = None
        self.qe_model = None
        self.qe_threshold = 0.7
        
        # 保存 LoRA 路径
        self.lora_model_id = lora_model_id

        self.nmt_max_length = 150
        self.nmt_max_input_length = 128

        self._load_models(nmt_model_id, reflection_model_id)

    def _load_models(self, nmt_model_id: str, reflection_model_id: Optional[str]):
        try:
            # 1. 加载 NMT 模型（NLLB，支持多语言翻译）
            logger.info(f"Loading NMT Base model: {nmt_model_id} (Device: {self.device})")
        
            # 基础模型和分词器
            self.nmt_tokenizer = AutoTokenizer.from_pretrained(nmt_model_id)
            self.nmt_model = AutoModelForSeq2SeqLM.from_pretrained(
                nmt_model_id,
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                load_in_8bit=False  # 保持 False
            ).to(self.device)

            # 🚀 新增 LoRA 挂载逻辑 🚀
            if self.lora_model_id and os.path.exists(self.lora_model_id):
                logger.info(f"Loading LoRA Adapter from: {self.lora_model_id}")
            
                # 加载 LoRA 权重
                lora_model = PeftModel.from_pretrained(
                    self.nmt_model, 
                    self.lora_model_id, 
                    adapter_name="nllb_lora" # 可以自定义一个名称
                ).to(self.device)
            
                # 切换到 LoRA 适配器（可选，但通常需要）
                lora_model.set_adapter("nllb_lora") 
            
                # 将 PEFT 模型设置为新的 NMT 模型
                self.nmt_model = lora_model 
            
                logger.info("✅ LoRA Adapter loaded and merged successfully. Model is ready for inference.")
        
            else:
                logger.info("✅ NMT Base Model loaded successfully (No LoRA adapter found or used).")
                

            # 2. 加载反思模型（可选）
            if reflection_model_id:
                logger.info(f"Loading reflection model: {reflection_model_id}")
                
                self.reflector = pipeline(
                    "text-generation",
                    model=reflection_model_id,
                    dtype=torch.float32, # Reflection model usually keeps float32 or bfloat16
                    device_map="auto",
                    model_kwargs={
                        "low_cpu_mem_usage": True,
                        "use_safetensors": True
                    }
                )

                # 确保 reflector 的 pad_token 设置稳定
                if self.reflector.tokenizer.pad_token is None and self.reflector.tokenizer.eos_token is not None:
                    self.reflector.tokenizer.pad_token = self.reflector.tokenizer.eos_token
                    self.reflector.model.config.pad_token_id = self.reflector.tokenizer.eos_token_id
                    logger.warning(f"Set pad_token/id to eos_token/id for reflection model.")

                logger.info("✅ Reflection model loaded successfully.")
            else:
                logger.warning("No reflection model specified, skipping optimization.")

            # 3. 加载 QE 模型
            self._load_qe_model()

        except Exception as e:
            logger.error(f"Model load failed: {e}", exc_info=True)
            self._cleanup_vram()
            raise Exception(f"Translator init error: {str(e)}")

    def _load_qe_model(self):
        """加载翻译质量评估（QE）模型"""
        try:
            self.qe_model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device.type)
            logger.info("✅ QE model (sentence-transformers) loaded successfully.")
        except Exception as e:
            logger.warning(f"QE model load failed: {e}", exc_info=True)
            self.qe_model = None

    def translate_segments(self, segments: List[Dict[str, Any]], target_lang: str, source_lang: str = 'auto',
                           use_reflection: bool = False, av_context: Optional[Dict[str, Any]] = None) -> List[
        Dict[str, Any]]:
        """
        翻译字幕片段（支持片段级 AV 上下文优化）
        """
        for idx, seg in enumerate(segments):
            if not all(key in seg for key in ["start", "end", "text"]):
                raise ValueError(f"Segment {idx + 1} missing required fields (start/end/text).")
            if "av_context" not in seg:
                seg["av_context"] = av_context or {}

        source_texts = [seg["text"].strip() for seg in segments]
        logger.info(
            f"Starting translation: {len(source_texts)} segments -> Target lang: {target_lang} (Reflection: {use_reflection})")

        # 第一步：批量翻译（基础翻译结果，包含 LoRA 影响）
        translated_texts = self._translate_batch(source_texts, source_lang, target_lang)
        logger.info(f"Batch translation completed.")

        # 第二步：反思优化
        if use_reflection and self.reflector:
            logger.info("Starting reflection optimization with segment-level AV context...")
            optimized_texts = []

            for idx, (seg, src_text, trans_text) in enumerate(zip(segments, source_texts, translated_texts)):
                segment_av_ctx = seg["av_context"] or av_context or {}
                optimized = self._reflect_and_improve(src_text, trans_text, target_lang, segment_av_ctx, idx + 1)
                optimized_texts.append(optimized)
            translated_texts = optimized_texts
            logger.info("Reflection optimization completed.")

        # 第三步：计算 QE 分数
        qe_scores = self._calculate_batch_qe_scores(source_texts, translated_texts) if self.qe_model else [0.0] * len(
            source_texts)

        # 第四步：组装最终结果
        result = []
        for idx, (seg, trans_text, qe_score) in enumerate(zip(segments, translated_texts, qe_scores)):
            result.append({
                "start": round(seg["start"], 2),
                "end": round(seg["end"], 2),
                "text": trans_text,
                "original_text": seg["text"],
                "qe_score": round(qe_score, 2),
                "av_context": seg["av_context"],
                "is_optimized": use_reflection and self.reflector is not None
            })

        logger.info(f"Translation process finished: {len(result)} segments processed.")

        return result

    def _translate_batch(self, texts: List[str], src_lang_code: str, tgt_lang_code: str) -> List[str]:
        """
        批量翻译文本（NLLB 模型核心翻译逻辑）
        """
        lang_map = {
            'auto': 'eng_Latn', 'en': 'eng_Latn', 'zh': 'zho_Hans', 'zh-cn': 'zho_Hans',
            'ja': 'jpn_Jpan', 'ko': 'kor_Hang', 'fr': 'fra_Latn', 'de': 'deu_Latn',
            'es': 'spa_Latn', 'ru': 'rus_Cyrl', 'ar': 'ara_Arab', 'hi': 'hin_Deva',
            'pt': 'por_Latn', 'it': 'ita_Latn', 'nl': 'nld_Latn', 'pl': 'pol_Latn'
        }

        src_code = lang_map.get(src_lang_code.lower(), 'eng_Latn')
        tgt_code = lang_map.get(tgt_lang_code.lower(), 'zho_Hans')
        
        inputs = self.nmt_tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.nmt_max_input_length
        ).to(self.device)

        forced_bos_token_id = self.nmt_tokenizer.convert_tokens_to_ids(tgt_code)

        with torch.no_grad():
            generated_tokens = self.nmt_model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=self.nmt_max_length,
                num_beams=4,
                do_sample=False,
                early_stopping=True,
                no_repeat_ngram_size=2
            )

        translations = self.nmt_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        translations = [trans.strip() for trans in translations]

        return translations

    def _reflect_and_improve(self, source_text: str, initial_translation: str, tgt_lang: str,
                             segment_av_ctx: Dict[str, Any], segment_idx: int) -> str:
        """
        结合片段级 AV 上下文优化翻译结果（反思机制）
        """
        scene_type = segment_av_ctx.get("scene_type", "无/未知").strip()
        environment = segment_av_ctx.get("environment", "无/未知").strip()
        emotion = segment_av_ctx.get("emotion", "无/未知").strip()
        activity = segment_av_ctx.get("activity", "无/未知").strip()
        scene_desc = segment_av_ctx.get("description", "无详细描述").strip()

        prompt = f"""你是一位专业的、场景感知的字幕翻译助手。你的任务是根据提供的**所有**场景信息和视觉描述，优化给定的翻译结果，以确保翻译的词汇、风格和情感与场景高度匹配。

=== 当前场景信息 (来自 VLM 的提取结果) ===
- 场景类型: {scene_type}
- 具体环境: {environment}
- 人物情感: {emotion}
- 活动状态: {activity}
- 详细视觉描述: {scene_desc}

=== 翻译守则 ===
1. 【准确性】翻译必须严格忠于原文含义，不得增删语义。
2. 【场景适配】请根据**以上所有信息**，选择最符合场景（例如：游戏、直播、医疗、法律、日常生活等）的专业术语和口语化风格。
3. 【情感匹配】翻译结果应能反映人物的情感状态（例如：兴奋、平静、严肃）。
4. 【简洁性】字幕需简短精炼，易于观众快速阅读。
5. 【目标语言】请翻译成{self._get_lang_name(tgt_lang)}。

=== 需要优化的内容 ===
- 源文本: "{source_text}"
- 基础翻译: "{initial_translation}"

请根据以上所有信息，输出优化后的最终翻译结果。
**🔴 核心指令: 你的回答中，必须且只能包含最终优化后的纯净中文字幕文本，不允许包含任何解释、分析、打招呼或额外的文字。请立即开始输出翻译文本。**
""" 

        try:
            prompt_text = prompt.strip()
            pad_id = self.reflector.model.config.pad_token_id
            eos_id = self.reflector.model.config.eos_token_id

            response = self.reflector(
                prompt_text,
                max_new_tokens=150,
                do_sample=False,
                num_return_sequences=1,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )[0]["generated_text"]

            optimized = initial_translation 

            if response.startswith(prompt_text):
                optimized = response[len(prompt_text):].strip()
            else:
                optimized = response.strip()

            optimized = re.sub(r'(Human:|Assistant:|\n\n).*', '', optimized, flags=re.IGNORECASE | re.DOTALL).strip()
            optimized = optimized.replace("<|endoftext|>", "").strip()
            optimized = optimized.strip('"').strip("'").strip()
            optimized = optimized.replace('\n', ' ').strip()

            if not optimized or len(optimized) < 1:
                return initial_translation

            return optimized

        except Exception as e:
            logger.error(f"Segment {segment_idx} reflection failed: {e}", exc_info=True)
            return initial_translation

    def _calculate_batch_qe_scores(self, source_texts: List[str], translated_texts: List[str]) -> List[float]:
        if not source_texts or not translated_texts or len(source_texts) != len(translated_texts):
            return [0.0] * len(source_texts)

        try:
            src_embeddings = self.qe_model.encode(source_texts, convert_to_tensor=True, show_progress_bar=False)
            trans_embeddings = self.qe_model.encode(translated_texts, convert_to_tensor=True, show_progress_bar=False)
            similarities = util.cos_sim(src_embeddings, trans_embeddings).diag().cpu().numpy()
            qe_scores = [float(max(0, sim)) for sim in similarities]
            return qe_scores
        except Exception as e:
            logger.error(f"QE score calculation failed: {e}", exc_info=True)
            return [0.0] * len(source_texts)

    def _get_lang_name(self, lang_code: str) -> str:
        lang_names = {
            'zh': '中文', 'zh-cn': '中文', 'en': '英文', 'ja': '日文', 'ko': '韩文',
            'fr': '法文', 'de': '德文', 'es': '西班牙文', 'ru': '俄文', 'ar': '阿拉伯文',
            'pt': '葡萄牙文', 'it': '意大利文', 'nl': '荷兰文', 'pl': '波兰文'
        }
        return lang_names.get(lang_code.lower(), lang_code)

    def _cleanup_vram(self, nmt_only: bool = False):
        if self.nmt_model:
            del self.nmt_model
            self.nmt_model = None

        if not nmt_only:
            if self.reflector:
                del self.reflector
                self.reflector = None
            if self.qe_model:
                del self.qe_model
                self.qe_model = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_supported_languages(self) -> Dict[str, List[str]]:
        return {
            "whisper": ["auto", "en", "zh", "ja", "ko", "fr", "de", "es", "ru", "ar", "hi", "pt", "it", "nl", "pl"],
            "nmt": ["en", "zh", "zh-cn", "ja", "ko", "fr", "de", "es", "ru", "ar", "hi", "pt", "it", "nl", "pl"]
        }

    def __del__(self):
        self._cleanup_vram()