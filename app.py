from flask import Flask, render_template, request, jsonify, send_file, url_for
import os
import json
import logging
import uuid
from datetime import datetime
from werkzeug.utils import secure_filename

from config import Config
from models.whisper_model_fixed import WhisperModel
from models.translator import Translator
from utils.audio_processor import AudioProcessor
from utils.subtitle_generator import SubtitleGenerator
from utils.file_handler import FileHandler

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建Flask应用
app = Flask(__name__)
app.config.from_object(Config)
Config.init_app(app)

# 初始化组件
try:
    whisper_model = WhisperModel(
        model_name=Config.WHISPER_MODEL,
        device=Config.WHISPER_DEVICE
    )
    translator = Translator(
        service=Config.TRANSLATOR_SERVICE,
        api_key=Config.OPENAI_API_KEY,
        use_reflection=Config.TRANSLATOR_USE_REFLECTION
    )
    audio_processor = AudioProcessor()
    subtitle_generator = SubtitleGenerator()
    file_handler = FileHandler()
    
    logger.info("所有组件初始化成功")
    if Config.TRANSLATOR_USE_REFLECTION:
        logger.info("翻译反思模式已启用")
except Exception as e:
    logger.error(f"组件初始化失败: {e}")
    raise

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/debug')
def debug():
    """调试页面"""
    return render_template('debug_test.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """上传文件API"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': '没有上传文件'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
        
        # 保存文件
        file_path, filename = file_handler.save_uploaded_file(file)
        
        return jsonify({
            'success': True,
            'file_path': file_path,
            'filename': filename,
            'original_name': file.filename
        })
        
    except Exception as e:
        logger.error(f"文件上传失败: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/transcribe', methods=['POST'])
def transcribe_audio():
    """音频转录API"""
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        language = data.get('language', 'auto')
        
        if not file_path or not os.path.exists(file_path):
            return jsonify({'error': '文件不存在'}), 400
        
        # 创建临时目录
        temp_dir = file_handler.create_temp_directory()
        
        try:
            # 处理音频文件
            processed_audio_path = audio_processor.process_audio_for_transcription(
                file_path, temp_dir
            )
            
            # 音频转录
            result = whisper_model.transcribe(processed_audio_path, language=language)
            
            return jsonify({
                'success': True,
                'text': result['text'],
                'segments': result['segments'],
                'language': result['language'],
                'duration': result['duration']
            })
            
        finally:
            # 清理临时文件
            file_handler.cleanup_temp_files(temp_dir)
            
    except Exception as e:
        logger.error(f"音频转录失败: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/translate', methods=['POST'])
def translate_subtitle():
    """字幕翻译API"""
    try:
        data = request.get_json()
        segments = data.get('segments', [])
        target_lang = data.get('target_language', 'en')
        source_lang = data.get('source_language', 'auto')
        
        if not segments:
            return jsonify({'error': '没有字幕内容'}), 400
        
        # 翻译字幕
        translated_segments = translator.translate_segments(
            segments, target_lang, source_lang
        )
        
        return jsonify({
            'success': True,
            'segments': translated_segments
        })
        
    except Exception as e:
        logger.error(f"字幕翻译失败: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-subtitle', methods=['POST'])
def generate_subtitle():
    """生成字幕文件API"""
    try:
        data = request.get_json()
        segments = data.get('segments', [])
        format_type = data.get('format', 'srt')
        filename = data.get('filename', 'subtitle')
        
        if not segments:
            return jsonify({'error': '没有字幕内容'}), 400
        
        # 生成输出文件名
        output_filename = file_handler.generate_output_filename(
            filename, "translated", f".{format_type}"
        )
        output_path = os.path.join(Config.OUTPUT_FOLDER, output_filename)
        
        # 创建字幕文件
        subtitle_path = subtitle_generator.create_subtitle(
            segments, output_path, format_type
        )
        
        return jsonify({
            'success': True,
            'download_url': url_for('download_file', filename=output_filename),
            'filename': output_filename
        })
        
    except Exception as e:
        logger.error(f"字幕文件生成失败: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/languages')
def get_languages():
    """获取支持的语言列表"""
    try:
        whisper_languages = whisper_model.get_supported_languages()
        translator_languages = translator.get_supported_languages()
        
        return jsonify({
            'whisper_languages': whisper_languages,
            'translator_languages': translator_languages
        })
        
    except Exception as e:
        logger.error(f"获取语言列表失败: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    """下载文件"""
    try:
        file_path = os.path.join(Config.OUTPUT_FOLDER, filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({'error': '文件不存在'}), 404
            
    except Exception as e:
        logger.error(f"文件下载失败: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/process-complete', methods=['POST'])
def process_complete():
    """处理完成，清理文件"""
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        
        if file_path and os.path.exists(file_path):
            file_handler.delete_file(file_path)
            
        return jsonify({'success': True})
        
    except Exception as e:
        logger.error(f"清理文件失败: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': '文件过大'}), 413

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': '页面不存在'}), 404

@app.route('/api/log-error', methods=['POST'])
def log_frontend_error():
    """记录前端错误日志"""
    try:
        data = request.get_json()
        function_name = data.get('function', 'unknown')
        error_message = data.get('error', 'unknown error')
        details = data.get('details', {})
        timestamp = data.get('timestamp', datetime.now().isoformat())
        
        logger.error(f"前端错误 [{function_name}]: {error_message}")
        if details:
            logger.error(f"详细信息: {json.dumps(details, ensure_ascii=False)}")
        
        return jsonify({'success': True})
        
    except Exception as e:
        logger.error(f"记录前端错误失败: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("启动AI字幕翻译应用...")
    
    # 检查FFmpeg
    if not audio_processor.check_ffmpeg():
        logger.warning("FFmpeg未安装，某些功能可能无法使用")
        print("警告: FFmpeg未安装，视频处理功能可能无法使用")
        print("请安装FFmpeg: https://ffmpeg.org/download.html")
    
    app.run(debug=True, host='0.0.0.0', port=5000)