# AI字幕生成与翻译系统

这是一个基于深度学习的AI字幕生成和翻译项目，可以自动将音频/视频文件转录为文字并翻译成多种语言。

## 功能特性

- 🎤 音频转录：使用OpenAI Whisper模型进行高精度语音识别
- 🌐 多语言翻译：支持多种语言之间的字幕翻译
- 📁 多格式支持：支持MP4、MP3、WAV等常见音视频格式
- 💾 多种输出格式：支持SRT、VTT等字幕格式导出
- 🌐 Web界面：提供友好的用户界面
- ⚡ 快速处理：使用优化的模型加速处理速度

## 技术栈

- **后端**：Python Flask
- **AI模型**：OpenAI Whisper、Google Translate
- **音频处理**：PyDub、FFmpeg
- **前端**：HTML/CSS/JavaScript
- **字幕格式**：SRT、WebVTT

## 项目结构

```
ai-caption-translator/
├── app.py                    # Flask主应用
├── config.py                 # 配置文件
├── requirements.txt          # Python依赖
├── static/                   # 静态文件
│   ├── css/
│   ├── js/
│   └── uploads/
├── templates/                # HTML模板
├── models/                   # AI模型相关
│   ├── whisper_model.py     # Whisper模型封装
│   └── translator.py        # 翻译模块
├── utils/                    # 工具模块
│   ├── audio_processor.py   # 音频处理
│   ├── subtitle_generator.py # 字幕生成
│   └── file_handler.py      # 文件处理
└── output/                   # 输出目录
```

## 安装和运行

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 运行应用
```bash
python app.py
```

### 3. 访问应用
打开浏览器访问 `http://localhost:5000`

## 使用说明

1. 上传音频或视频文件
2. 选择源语言和目标语言
3. 点击生成字幕按钮
4. 等待处理完成
5. 下载生成的字幕文件

## API接口

- `POST /upload` - 上传文件
- `POST /transcribe` - 音频转录
- `POST /translate` - 字幕翻译
- `GET /download/<filename>` - 下载字幕文件

## 配置说明

在 `config.py` 中可以配置：
- Whisper模型大小（tiny/base/small/medium/large）
- 翻译服务选择
- 文件上传限制
- 输出格式设置

## 注意事项

- 首次运行会下载Whisper模型，可能需要一些时间
- 大文件处理需要较多内存和CPU资源
- 建议使用GPU加速（如果可用）