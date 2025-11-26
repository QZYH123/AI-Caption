class VLMSceneAnalyzer:
    def __init__(self):
        pass

    def analyze(self, video_path, timestamp):
        """
        返回一个空的分析结果，保证系统正常运行
        """
        return {
            "scene": None,
            "faces": [],
            "objects": [],
            "emotion": None,
        }
# models/vlm_scene_analyzer.py