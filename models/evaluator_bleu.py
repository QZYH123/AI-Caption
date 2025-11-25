# models/evaluator_bleu.py

import os
import re
from sacrebleu.metrics import BLEU

class SacreBLEUEvaluator:
    def __init__(self):
        self.metric = BLEU()

    def load_srt(self, path):
        """
        读取 SRT 文件，按句子列表返回，兼容 sacrebleu 格式
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"文件不存在: {path}")

        lines = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()

                # 跳过序号（独占一行）
                if re.fullmatch(r"\d+", line):
                    continue
                
                # 跳过时间轴
                if "-->" in line:
                    continue

                if line:
                    lines.append(line)

        return lines    # 返回列表而不是一个长字符串

    def evaluate(self, reference_path, candidate_path):
        """
        计算 BLEU 分数
        """
        ref_lines = self.load_srt(reference_path)
        cand_lines = self.load_srt(candidate_path)

        # sacrebleu 的格式要求：
        # hypotheses: List[str]
        # references: List[List[str]]
        score_obj = self.metric.corpus_score(cand_lines, [ref_lines])

        return {
            "score": score_obj.score,
            "precisions": score_obj.precisions,
            "bp": score_obj.bp,
            "sys_len": score_obj.sys_len,
            "ref_len": score_obj.ref_len,
        }
