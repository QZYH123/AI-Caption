import logging
import requests
from typing import List, Dict, Any, Optional
# from googletrans import Translator as GoogleTranslator
import asyncio
import edge_tts
import os

logger = logging.getLogger(__name__)

# 尝试导入OpenAI，如果不可用则使用基础翻译
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI库未安装，将使用基础翻译功能")

class Translator:
    def __init__(self, service: str = "google", api_key: str = "", use_reflection: bool = False):
        """
        初始化翻译器
        
        Args:
            service: 翻译服务 (google, openai)
            api_key: API密钥（如果需要）
            use_reflection: 是否使用翻译-反思-翻译方法（需要OpenAI）
        """
        self.service = service
        self.api_key = api_key or os.getenv('OPENAI_API_KEY', '')
        self.use_reflection = use_reflection and OPENAI_AVAILABLE
        
        # 初始化OpenAI客户端
        if self.use_reflection and self.api_key:
            try:
                self.openai_client = OpenAI(api_key=self.api_key)
                logger.info("OpenAI翻译反思模式已启用")
            except Exception as e:
                logger.warning(f"OpenAI初始化失败: {e}，将使用基础翻译")
                self.use_reflection = False
        else:
            self.openai_client = None
        
    def translate_text(self, text: str, target_lang: str, source_lang: str = "auto") -> str:
        """
        翻译文本
        
        Args:
            text: 要翻译的文本
            target_lang: 目标语言代码
            source_lang: 源语言代码
            
        Returns:
            翻译后的文本
        """
        try:
            # 如果启用了反思模式，使用翻译-反思-翻译方法
            if self.use_reflection:
                return self._translate_with_reflection(text, target_lang, source_lang)
            
            # 否则使用基础翻译
            if self.service == "google":
                return self._translate_google(text, target_lang, source_lang)
            elif self.service == "openai":
                return self._translate_openai(text, target_lang, source_lang)
            else:
                raise ValueError(f"不支持的翻译服务: {self.service}")
                
        except Exception as e:
            logger.error(f"文本翻译失败: {e}")
            raise Exception(f"翻译失败: {e}")
    
    def _translate_google(self, text: str, target_lang: str, source_lang: str = "auto") -> str:
        """使用Google翻译API"""
        try:
            # 使用Google Translate API
            url = "https://translate.googleapis.com/translate_a/single"
            params = {
                'client': 'gtx',
                'sl': source_lang if source_lang != 'auto' else 'auto',
                'tl': target_lang,
                'dt': 't',
                'q': text
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            result = response.json()
            translated_text = ''.join([item[0] for item in result[0]])
            return translated_text
            
        except Exception as e:
            logger.error(f"Google翻译失败: {e}")
            # 如果API失败，返回原文
            return text
    
    def _translate_openai(self, text: str, target_lang: str, source_lang: str = "auto") -> str:
        """使用OpenAI进行翻译"""
        try:
            lang_map = {
                'zh': '中文', 'zh-cn': '简体中文', 'zh-tw': '繁体中文',
                'en': 'English', 'ja': '日本語', 'ko': '한국어',
                'fr': 'Français', 'de': 'Deutsch', 'es': 'Español',
                'ru': 'Русский', 'ar': 'العربية', 'pt': 'Português'
            }
            target_language = lang_map.get(target_lang, target_lang)
            
            prompt = f"""请将以下文本翻译成{target_language}。
要求：
1. 保持原文的语气和风格
2. 确保翻译自然流畅
3. 只返回翻译结果，不要添加任何解释

原文：
{text}"""
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI翻译失败: {e}")
            return text
    
    def _translate_with_reflection(self, text: str, target_lang: str, source_lang: str = "auto") -> str:
        """使用翻译-反思-翻译方法提升翻译质量"""
        try:
            lang_map = {
                'zh': '中文', 'zh-cn': '简体中文', 'zh-tw': '繁体中文',
                'en': 'English', 'ja': '日本語', 'ko': '한국어',
                'fr': 'Français', 'de': 'Deutsch', 'es': 'Español',
                'ru': 'Русский', 'ar': 'العربية', 'pt': 'Português'
            }
            target_language = lang_map.get(target_lang, target_lang)
            source_language = lang_map.get(source_lang, '原语言') if source_lang != 'auto' else '原语言'
            
            # 第一步：初始翻译
            logger.info("步骤1/3: 初始翻译...")
            initial_prompt = f"""请将以下{source_language}文本翻译成{target_language}。
要求：
1. 准确传达原文含义
2. 保持原文的语气和风格
3. 确保翻译自然流畅
4. 只返回翻译结果，不要添加任何解释或标记

原文：
{text}"""
            
            initial_response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": initial_prompt}],
                temperature=0.3
            )
            initial_translation = initial_response.choices[0].message.content.strip()
            
            # 第二步：反思和改进建议
            logger.info("步骤2/3: 反思分析...")
            reflection_prompt = f"""作为一位专业的翻译评审专家，请评估以下翻译的质量，并提供改进建议。

原文（{source_language}）：
{text}

当前翻译（{target_language}）：
{initial_translation}

请分析：
1. 翻译的准确性（是否准确传达原文含义）
2. 语言的流畅性（是否符合目标语言的表达习惯）
3. 专业术语的处理（如有）
4. 语气和风格的保持
5. 可能存在的错误或不当之处

请提供具体的改进建议，说明哪些地方可以优化。"""
            
            reflection_response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": reflection_prompt}],
                temperature=0.5
            )
            reflection = reflection_response.choices[0].message.content.strip()
            logger.debug(f"反思结果: {reflection}")
            
            # 第三步：基于反思改进翻译
            logger.info("步骤3/3: 改进翻译...")
            improvement_prompt = f"""基于以下反思和建议，请提供改进后的翻译版本。

原文（{source_language}）：
{text}

初始翻译（{target_language}）：
{initial_translation}

反思和改进建议：
{reflection}

请提供改进后的最终翻译，只返回翻译结果，不要添加任何解释或标记。"""
            
            final_response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": improvement_prompt}],
                temperature=0.3
            )
            final_translation = final_response.choices[0].message.content.strip()
            
            logger.info("翻译-反思-翻译流程完成")
            return final_translation
            
        except Exception as e:
            logger.error(f"反思翻译失败: {e}，回退到基础翻译")
            # 如果反思翻译失败，回退到基础翻译
            return self._translate_google(text, target_lang, source_lang)
    
    def translate_segments(self, segments: List[Dict[str, Any]], 
                          target_lang: str, source_lang: str = "auto") -> List[Dict[str, Any]]:
        """
        翻译字幕段落
        
        Args:
            segments: 字幕段落列表
            target_lang: 目标语言
            source_lang: 源语言
            
        Returns:
            翻译后的字幕段落
        """
        translated_segments = []
        
        try:
            for segment in segments:
                translated_text = self.translate_text(
                    segment["text"], 
                    target_lang, 
                    source_lang
                )
                
                translated_segment = segment.copy()
                translated_segment["text"] = translated_text
                translated_segments.append(translated_segment)
                
            logger.info(f"翻译完成: {len(segments)} 个段落")
            return translated_segments
            
        except Exception as e:
            logger.error(f"字幕翻译失败: {e}")
            raise Exception(f"字幕翻译失败: {e}")
    
    def get_supported_languages(self) -> Dict[str, str]:
        """获取支持的语言列表"""
        if self.service == "google":
            return {
                'auto': '自动检测',
                'zh-cn': '中文（简体）',
                'zh-tw': '中文（繁体）',
                'en': '英语',
                'ja': '日语',
                'ko': '韩语',
                'fr': '法语',
                'de': '德语',
                'es': '西班牙语',
                'ru': '俄语',
                'ar': '阿拉伯语',
                'pt': '葡萄牙语',
                'it': '意大利语',
                'nl': '荷兰语',
                'pl': '波兰语',
                'tr': '土耳其语',
                'vi': '越南语',
                'th': '泰语',
                'id': '印尼语',
                'ms': '马来语'
            }
        else:
            return {}