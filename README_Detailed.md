# AI字幕生成与翻译系统

这是一个基于深度学习的AI字幕生成和翻译项目，可以自动将音频/视频文件转录为文字并翻译成多种语言。

## 🌟 功能特性

- **🎤 音频转录**：使用OpenAI Whisper模型进行高精度语音识别
- **🌐 多语言翻译**：支持多种语言之间的字幕翻译
- **📁 多格式支持**：支持MP4、MP3、WAV等常见音视频格式
- **💾 多种输出格式**：支持SRT、VTT、JSON等字幕格式导出
- **🌐 Web界面**：提供友好的用户界面
- **⚡ 快速处理**：使用优化的模型加速处理速度
- **🖥️ 命令行工具**：支持批量处理音频文件

## 🛠️ 技术栈

- **后端**：Python Flask
- **AI模型**：OpenAI Whisper、Google Translate
- **音频处理**：PyDub、FFmpeg
- **前端**：HTML/CSS/JavaScript + Bootstrap
- **字幕格式**：SRT、WebVTT、JSON

## 📁 项目结构

```
ai-caption-translator/
├── app.py                    # Flask主应用
├── cli.py                    # 命令行工具
├── config.py                 # 配置文件
├── requirements.txt          # Python依赖
├── static/                   # 静态文件
│   ├── css/
│   ├── js/
│   └── uploads/             # 上传文件存储
├── templates/                # HTML模板
│   └── index.html           # Web界面
├── models/                   # AI模型相关
│   ├── __init__.py
│   ├── whisper_model.py     # Whisper模型封装
│   └── translator.py        # 翻译模块
├── utils/                    # 工具模块
│   ├── __init__.py
│   ├── audio_processor.py   # 音频处理
│   ├── subtitle_generator.py # 字幕生成
│   └── file_handler.py      # 文件处理
└── output/                   # 输出目录
```

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 运行Web应用
```bash
python app.py
```
打开浏览器访问 `http://localhost:5000`

### 3. 使用命令行工具（可选）
```bash
# 处理单个文件
python cli.py video.mp4 -t zh-cn

# 批量处理目录
python cli.py ./videos/ --recursive -t zh-cn

# 指定输出格式和模型
python cli.py audio.mp3 -f vtt -m large -t en
```

## 📋 使用说明

### Web界面使用

1. **上传文件**：拖拽或点击上传音频/视频文件
2. **选择语言**：选择音频语言和目标翻译语言
3. **开始转录**：点击开始转录按钮，等待处理完成
4. **字幕翻译**：选择目标语言进行翻译（可选）
5. **下载字幕**：预览并下载生成的字幕文件

### 命令行使用

```bash
# 基本用法
python cli.py input_file -t target_language

# 高级选项
python cli.py input_file \
    -o output_dir \
    -l source_language \
    -t target_language \
    -f subtitle_format \
    -m model_size \
    -d device \
    --save-transcript
```

#### 参数说明

- `input`：输入文件或目录路径
- `-o, --output-dir`：输出目录（默认：output）
- `-l, --source-lang`：源语言，auto表示自动检测（默认：auto）
- `-t, --target-lang`：目标语言，none表示不翻译（默认：zh-cn）
- `-f, --format`：输出格式：srt, vtt, json（默认：srt）
- `-m, --model-size`：Whisper模型大小：tiny, base, small, medium, large（默认：base）
- `-d, --device`：运行设备：cpu, cuda（默认：cpu）
- `--translator`：翻译服务（默认：google）
- `--save-transcript`：保存转录文本
- `--recursive`：递归处理子目录
- `--extensions`：处理的文件扩展名

## 🌍 支持的语言

### Whisper支持的语言
支持99种语言的语音识别，包括：
- 中文（zh）
- 英语（en）
- 日语（ja）
- 韩语（ko）
- 法语（fr）
- 德语（de）
- 西班牙语（es）
- 俄语（ru）
- 阿拉伯语（ar）
- 葡萄牙语（pt）
- 等等...

### 翻译支持的语言
- 中文（简体/繁体）
- 英语
- 日语
- 韩语
- 法语
- 德语
- 西班牙语
- 俄语
- 阿拉伯语
- 葡萄牙语
- 意大利语
- 荷兰语
- 波兰语
- 土耳其语
- 越南语
- 泰语
- 印尼语
- 马来语

## ⚙️ 配置说明

在 `config.py` 中可以配置：

- **Whisper模型**：选择模型大小（tiny/base/small/medium/large）
- **运行设备**：CPU或CUDA（GPU）
- **翻译服务**：Google翻译等
- **文件限制**：最大上传文件大小（默认500MB）
- **输出格式**：默认字幕格式
- **临时文件**：临时文件存储位置

## 🔧 环境要求

### 系统要求
- Python 3.7+
- FFmpeg（用于视频处理）
- 足够的磁盘空间（模型文件较大）

### 安装FFmpeg
```bash
# Windows
# 下载地址: https://ffmpeg.org/download.html

# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt update
sudo apt install ffmpeg
```

### Python依赖
```bash
pip install -r requirements.txt
```

主要依赖包括：
- `openai-whisper`：语音识别模型
- `flask`：Web框架
- `pydub`：音频处理
- `googletrans`：翻译服务
- `srt`：字幕格式处理
- `webvtt-py`：WebVTT格式支持

## 🎯 使用场景

- **视频制作**：为视频自动生成字幕
- **语言学习**：制作双语字幕
- **会议记录**：转录会议录音
- **内容创作**：为播客、视频添加字幕
- **无障碍访问**：为听障人士提供字幕
- **批量处理**：大量音视频文件的字幕生成

## 💡 注意事项

1. **首次运行**：首次运行会下载Whisper模型，可能需要一些时间
2. **大文件处理**：大文件处理需要较多内存和CPU资源
3. **GPU加速**：建议使用GPU加速处理（如果可用）
4. **文件格式**：确保输入文件格式受支持
5. **网络连接**：翻译功能需要网络连接

## 🔒 安全提示

- 不要在生产环境中使用调试模式
- 定期清理上传的临时文件
- 限制文件上传大小
- 验证上传的文件类型
- 使用安全的文件名

## 🐛 常见问题

### Q: 模型下载很慢怎么办？
A: 可以手动下载模型文件放到缓存目录，或使用代理。

### Q: 处理大文件时内存不足？
A: 尝试使用较小的模型（tiny/base），或增加系统内存。

### Q: 视频处理失败？
A: 确保FFmpeg已正确安装并添加到系统路径。

### Q: 翻译功能无法使用？
A: 检查网络连接，或尝试使用不同的翻译服务。

## 📞 支持与反馈

如有问题或建议，请通过以下方式联系：
- 提交Issue
- 发送邮件
- 技术社区讨论

## 📄 许可证

本项目采用MIT许可证，详见LICENSE文件。

---

**🎉 祝您使用愉快！**