# AI Caption (NeuralSub 智能字幕系统)

## 项目简介
本项目是一个基于 AI 的视频字幕生成与翻译系统，旨在为视频内容提供高质量的自动字幕生成及多语言翻译服务。系统集成了 OpenAI Whisper 进行语音转写，并结合神经机器翻译（NLLB）与大语言模型（Qwen）实现高精度的字幕翻译，同时引入了“反思模式”以提升翻译的自然度和准确性。

## 核心功能
- **自动语音转写 (ASR)**: 使用 Whisper 模型将视频/音频转换为文本。
- **智能翻译**: 
  - 支持 NLLB (No Language Left Behind) 进行基础翻译。
  - 挂载针对专业词汇与歧义语义微调的LoRA模型。
  - 集成 Qwen 大模型进行“反思式”翻译优化。
  - 添加VLM视觉模型进行场景/物品标签识别，辅助翻译。
- **质量评估 (QE)**: 使用 SentenceTransformers (Bi-Encoder) 对翻译结果进行语义相似度评分。
- **多端支持**: 提供 Web 可视化界面 (Flask) 和命令行工具 (CLI)。
- **多格式输出**: 支持导出 SRT, VTT, JSON 等多种字幕格式。

## 项目结构

```
AI-Caption/
├── README.md
├── app.py                  # Web 应用入口 (Flask)
├── cli.py                  # 命令行工具入口
├── config.py               # 项目配置文件
├── data_preparation.py     # 微调数据集构建器
├── install.sh              # 安装脚本
├── monitor_api.py
├── requirements.txt        # 项目依赖清单
├── run_app_production.py
├── test_lora_quality.py
├── test_qe_manual.py
├── translator_with_lora.py
├── utils/                  # 通用工具库
│   ├── __init__.py
│   ├── __pycache__/
│   ├── audio_processor.py  # 音频处理与提取 (FFmpeg)
│   ├── file_handler.py     # 文件 I/O 管理
│   ├── srt_utils.py
│   ├── subtitle_generator.py  # 字幕文件生成逻辑
│   └── video_processor.py
├── data/
│   ├── finetune/
│   ├── reference.srt
│   └── terminology.json
├── static/                 # Web 静态资源
│   └── js/
│       ├── app.js
│       └── app_debug.js
├── templates/              # Web 页面模板 (HTML)
│   ├── debug_test.html
│   └── index.html
└── models/                 # AI 模型核心组件
    ├── __init__.py
    ├── vlm_scene_analyzer.py
    ├── whisper_model_fixed.py  # Whisper ASR 模型封装
    ├── translator.py           # 神经翻译器 (NLLB + Qwen)
    ├── quality_estimator.py    # 翻译质量评估模块
    ├── finetune.py             # 模型微调脚本 (实验性)
    ├── lora_finetune.py		# LoRA模型训练脚本
    └── lora_nllb_terminology/  # LoRA模型

```

## 快速开始

### 环境要求
- Python 3.11
- FFmpeg (必须安装并配置环境变量)
- CUDA (推荐，用于 GPU 加速)
- torch版本：2.6.0+cu124

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行 Web 应用
```bash
python app.py
# 访问 http://localhost:5000
```


## 待办事项 (TODO)
- [ ] 升级至更大参数的模型以提升精度。

- [ ] 丰富视觉模块，利用画面信息辅助字幕生成。

- [ ] 建立合理的翻译性能评估体系。

  
