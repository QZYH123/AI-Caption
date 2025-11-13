import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Flask配置
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    UPLOAD_FOLDER = 'static/uploads'
    OUTPUT_FOLDER = 'output'
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB max file size
    
    # Whisper配置
    WHISPER_MODEL = os.environ.get('WHISPER_MODEL', 'base')  # tiny, base, small, medium, large
    WHISPER_DEVICE = os.environ.get('WHISPER_DEVICE', 'cpu')  # cpu, cuda
    WHISPER_LANGUAGE = os.environ.get('WHISPER_LANGUAGE', 'auto')  # auto or specific language code
    
    # 翻译配置
    TRANSLATOR_SERVICE = os.environ.get('TRANSLATOR_SERVICE', 'google')  # google, openai
    TRANSLATOR_USE_REFLECTION = os.environ.get('TRANSLATOR_USE_REFLECTION', 'false').lower() == 'true'
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
    GOOGLE_TRANSLATE_API_KEY = os.environ.get('GOOGLE_TRANSLATE_API_KEY', '')
    
    # 音频处理配置
    SUPPORTED_AUDIO_FORMATS = ['.mp3', '.wav', '.m4a', '.flac', '.aac']
    SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    SUPPORTED_FORMATS = SUPPORTED_AUDIO_FORMATS + SUPPORTED_VIDEO_FORMATS
    
    # 字幕配置
    SUBTITLE_FORMATS = ['srt', 'vtt']
    DEFAULT_SUBTITLE_FORMAT = 'srt'
    
    # 临时文件配置
    TEMP_FOLDER = 'temp'
    
    @staticmethod
    def init_app(app):
        # 创建必要的文件夹
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(Config.OUTPUT_FOLDER, exist_ok=True)
        os.makedirs(Config.TEMP_FOLDER, exist_ok=True)