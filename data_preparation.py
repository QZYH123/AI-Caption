# -*- coding: UTF-8 -*-
"""
NLLB LoRA 微调数据准备工具
用于构建专业词汇和多义词的训练数据集
"""

import json
import os
from typing import List, Dict
import random


class FinetuneDatasetBuilder:
    """微调数据集构建器"""
    
    def __init__(self, output_dir: str = "data/finetune"):
        self.output_dir = output_dir
        os.makedirs(f"{output_dir}/train", exist_ok=True)
        os.makedirs(f"{output_dir}/eval", exist_ok=True)
        
        # ==========================================
        # 1. 扩充后的专业术语知识库 
        # ==========================================
        self.terminology_pairs = [
            # === 鸟类与自然  ===
            ("Sandhill Crane", "沙丘鹤"),
            ("Whooping Crane", "美洲鹤"),
            ("Great Blue Heron", "大蓝鹭"),
            ("Bald Eagle", "白头海雕"),
            ("Peregrine Falcon", "游隼"),
            ("Snowy Owl", "雪鸮"),
            ("Kingfisher", "翠鸟"),
            ("Golden Retriever", "金毛寻回犬"),
            ("Siberian Husky", "哈士奇"),
            ("Golden Retriever", "金毛寻回犬"),
            ("Labrador Retriever", "拉布拉多猎犬"),
            ("German Shepherd", "德国牧羊犬"),
            ("Poodle", "贵宾犬"),
            ("Shih Tzu", "西施犬"),
            ("Bulldog", "斗牛犬"),
            ("Beagle", "比格犬"),
            ("Pug", "哈巴狗"),
            ("Chihuahua", "吉娃娃"),
            ("Dachshund", "短腿猎犬"),
            ("Corgi", "柯基犬"),
            ("Pomeranian", "博美犬"),
            ("Persian Cat", "波斯猫"),
            ("Siamese Cat", "暹罗猫"),
            ("Maine Coon Cat", "缅因猫"),
            ("Ragdoll Cat", "布偶猫"),
            ("Sphynx Cat", "无毛猫"),
            ("Abyssinian Cat", "阿比西尼亚猫"),
            ("Savannah Cat", "萨凡纳猫"),
            ("Bengal Cat", "孟加拉猫"),
            ("Oriental Short-haired Cat", "东方短毛猫"),
            ("American Shorthair Cat", "美国短毛猫"),
            ("Giant Panda", "大熊猫"),
            ("Red Panda", "小熊猫"),
            ("Grizzly Bear", "灰熊"),
            ("Polar Bear", "北极熊"),
            ("Black Bear", "黑熊"),
            ("Brown Bear", "棕熊"),
            ("Sloth", "树懒"),
            ("White-tailed Deer", "白尾鹿"),
            ("Monarch Butterfly", "帝王蝶"),

            
    

            # === 电竞/游戏/动漫 (针对 NLLB 容易直译的人名/术语) ===
            ("Night (the professional player)", "Night（职业选手）"),
            ("Faker dominates the mid lane", "Faker 统治中路"),
            ("Uzi's positioning is perfect", "Uzi 的站位完美"),
            ("TheShy", "TheShy"),
            ("Rookie", "Rookie"), 
            ("ShowMaker", "ShowMaker"),
            ("Jungler", "打野"),
            ("Gank", "抓人"),
            ("Nerf", "削弱"),
            ("Buff", "增益"),
            ("Carry", "核心输出"),
            ("Support", "辅助"),
            ("Ult / Ultimate", "大招"),
            ("Cooldown", "冷却时间"),
            ("Respawn", "重生"),
            
            # === AI 与 计算机科学 (针对你的项目领域) ===
            ("Transformer architecture", "Transformer 架构"),
            ("Generative AI", "生成式人工智能"),
            ("Large Language Model", "大语言模型"),
            ("Neural Network", "神经网络"),
            ("Backpropagation", "反向传播"),
            ("Epoch", "轮次"),
            ("Batch Size", "批大小"),
            ("Inference", "推理"),
            ("Latency", "延迟"),
            ("Fine-tuning", "微调"),
            ("Zero-shot learning", "零样本学习"),
            ("Hallucination", "幻觉"),
            ("Tokenization", "分词"),
            ("Embedding", "嵌入"),
            ("Gradient Descent", "梯度下降"),
            ("Python script", "Python 脚本"),
            ("Driver", "驱动程序"), # 易错译为“司机”

            # === 医学术语 (高精度要求) ===
            ("myocardial infarction", "心肌梗死"),
            ("hypertension", "高血压"),
            ("pneumonia", "肺炎"),
            ("CPR", "心肺复苏"),
            ("MRI", "核磁共振"),
            ("Placebo effect", "安慰剂效应"),
            ("Chronic disease", "慢性病"),
            ("Anesthesia", "麻醉"),
            ("ICU", "重症监护室"),
            ("Trauma", "创伤"),

            # === 法律与罪案 (美剧常用) ===
            ("plaintiff", "原告"),
            ("defendant", "被告"),
            ("jurisdiction", "司法管辖权"),
            ("Parole", "假释"),
            ("Subpoena", "传票"),
            ("Affidavit", "宣誓书"),
            ("Objection", "反对"),
            ("Sustained", "（法官）维持异议"),
            ("Overruled", "（法官）驳回异议"),
            ("Felony", "重罪"),
            ("Misdemeanor", "轻罪"),
            ("Warrant", "搜查令/逮捕令"),

            # === 常见多义词/易错词 (影视字幕重灾区) ===
            ("Apple (the company)", "苹果公司"),
            ("Windows (the OS)", "Windows 系统"),
            ("China (porcelain)", "瓷器"),
            ("Turkey (the country)", "土耳其"),
            ("Turkey (the bird)", "火鸡"),
            ("Date (calendar)", "日期"),
            ("Date (romantic)", "约会"),
            ("Date (fruit)", "海枣"),
            ("Lead (metal)", "铅"),
            ("Lead (verb)", "领导"),

            
            # === AI/计算机科学专业术语 ===
            ("Transformer architecture", "Transformer架构"),
            ("LoRA / Low-Rank Adaptation", "低秩适配"),
            ("QLoRA", "量化低秩适配"),
            ("Parameter Efficient Fine-tuning", "参数高效微调"),
            ("PEFT", "参数高效微调技术"),
            ("Retrieval-Augmented Generation", "检索增强生成"),
            ("RAG", "检索增强生成"),
            ("Chain of Thought", "思维链"),
            ("In-context Learning", "上下文学习"),
            ("Prompt Engineering", "提示工程"),
            ("Instruction Tuning", "指令调优"),
            ("RLHF / Reinforcement Learning from Human Feedback", "基于人类反馈的强化学习"),
            ("Constitutional AI", "宪法AI"),
            ("Multi-modal AI", "多模态人工智能"),
            ("Vision-Language Model", "视觉语言模型"),
            ("VLM", "视觉语言模型"),
            ("CLIP", "CLIP模型"),
            ("Mixture of Experts", "混合专家模型"),
            ("MoE", "混合专家模型"),
            ("Sparse Activation", "稀疏激活"),
            ("FlashAttention", "Flash注意力机制"),
            ("Grouped-Query Attention", "分组查询注意力"),
            ("GQA", "分组查询注意力"),
            ("Rotary Position Embedding", "旋转位置编码"),
            ("RoPE", "旋转位置编码"),
            ("Swish activation", "Swish激活函数"),
            ("GELU activation", "GELU激活函数"),
            ("Layer Normalization", "层归一化"),
            ("RMSNorm", "RMS归一化"),
            ("Gradient Checkpointing", "梯度检查点"),
            ("ZeRO optimization", "ZeRO优化"),
            ("LoRA adapter", "LoRA适配器"),
            ("Adapter fusion", "适配器融合"),
            ("Prefix Tuning", "前缀调优"),
            ("P-tuning", "P调优"),
            ("Bit quantization", "比特量化"),
            ("4-bit quantization", "4比特量化"),
            ("8-bit quantization", "8比特量化"),
            ("GPTQ", "GPTQ量化"),
            ("AWQ", "AWQ量化"),
            ("SmoothQuant", "平滑量化"),
            ("KV cache", "键值缓存"),
            ("Context window extension", "上下文窗口扩展"),
            ("Positional encoding", "位置编码"),
            ("Byte Pair Encoding", "字节对编码"),
            ("BPE", "字节对编码"),
            ("SentencePiece", "SentencePiece分词器"),
            ("Unigram language model", "一元语言模型"),
            ("Perplexity", "困惑度"),
            ("BLEU score", "BLEU分数"),
            ("ROUGE score", "ROUGE分数"),
            ("METEOR score", "METEOR分数"),
            ("ChrF score", "ChrF分数"),
            ("Contrastive Learning", "对比学习"),
            ("Self-supervised Learning", "自监督学习"),
            ("Contrastive Loss", "对比损失"),
            ("Triplet Loss", "三元组损失"),
            ("InfoNCE loss", "InfoNCE损失"),
            ("Knowledge Distillation", "知识蒸馏"),
            ("Model Pruning", "模型剪枝"),
            ("Model Quantization", "模型量化"),
            ("Model Compression", "模型压缩"),
            ("ONNX Runtime", "ONNX运行时"),
            ("TensorRT", "TensorRT推理引擎"),
            ("vLLM", "vLLM推理框架"),
            ("Text Generation Inference", "文本生成推理"),
            ("TGI", "文本生成推理"),
            ("StreamingLLM", "流式LLM"),
            ("Speculative Decoding", "推测解码"),
            ("Draft model", "草稿模型"),
            ("Verification model", "验证模型"),
            ("Mamba architecture", "Mamba架构"),
            ("State Space Model", "状态空间模型"),
            ("SSM", "状态空间模型"),
            ("Hyena Hierarchy", "Hyena层级结构"),
            ("RWKV", "RWKV模型"),
            ("Transformer-XL", "Transformer-XL"),
            ("Longformer", "Longformer模型"),
            ("Reformer", "Reformer模型"),
            ("Linformer", "Linformer模型"),
            ("Perceiver", "Perceiver模型"),
            ("RetNet", "RetNet模型"),
    
            # === 医学专业术语 ===
            ("myocardial infarction", "心肌梗死"),
            ("acute coronary syndrome", "急性冠脉综合征"),
            ("ACS", "急性冠脉综合征"),
            ("ST-elevation myocardial infarction", "ST段抬高型心肌梗死"),
            ("STEMI", "ST段抬高型心肌梗死"),
            ("non-ST-elevation myocardial infarction", "非ST段抬高型心肌梗死"),
            ("NSTEMI", "非ST段抬高型心肌梗死"),
            ("percutaneous coronary intervention", "经皮冠状动脉介入治疗"),
            ("PCI", "经皮冠状动脉介入治疗"),
            ("coronary artery bypass graft", "冠状动脉旁路移植术"),
            ("CABG", "冠状动脉旁路移植术"),
            ("ventricular fibrillation", "心室颤动"),
            ("VFib", "心室颤动"),
            ("atrial fibrillation", "心房颤动"),
            ("AFib", "心房颤动"),
            ("pulmonary embolism", "肺栓塞"),
            ("PE", "肺栓塞"),
            ("deep vein thrombosis", "深静脉血栓"),
            ("DVT", "深静脉血栓"),
            ("non-alcoholic steatohepatitis", "非酒精性脂肪性肝炎"),
            ("NASH", "非酒精性脂肪性肝炎"),
            ("non-alcoholic fatty liver disease", "非酒精性脂肪肝"),
            ("NAFLD", "非酒精性脂肪肝"),
            ("gastroesophageal reflux disease", "胃食管反流病"),
            ("GERD", "胃食管反流病"),
            ("inflammatory bowel disease", "炎症性肠病"),
            ("IBD", "炎症性肠病"),
            ("irritable bowel syndrome", "肠易激综合征"),
            ("IBS", "肠易激综合征"),
            ("Clostridioides difficile infection", "艰难梭菌感染"),
            ("C. diff", "艰难梭菌"),
            ("hepatocellular carcinoma", "肝细胞癌"),
            ("HCC", "肝细胞癌"),
            ("pancreatic ductal adenocarcinoma", "胰腺导管腺癌"),
            ("PDAC", "胰腺导管腺癌"),
            ("chronic obstructive pulmonary disease", "慢性阻塞性肺疾病"),
            ("COPD", "慢性阻塞性肺疾病"),
            ("community-acquired pneumonia", "社区获得性肺炎"),
            ("CAP", "社区获得性肺炎"),
            ("hospital-acquired pneumonia", "医院获得性肺炎"),
            ("HAP", "医院获得性肺炎"),
            ("ventilator-associated pneumonia", "呼吸机相关性肺炎"),
            ("VAP", "呼吸机相关性肺炎"),
            ("acute respiratory distress syndrome", "急性呼吸窘迫综合征"),
            ("ARDS", "急性呼吸窘迫综合征"),
            ("idiopathic pulmonary fibrosis", "特发性肺纤维化"),
            ("IPF", "特发性肺纤维化"),
            ("systemic lupus erythematosus", "系统性红斑狼疮"),
            ("SLE", "系统性红斑狼疮"),
            ("rheumatoid arthritis", "类风湿关节炎"),
            ("RA", "类风湿关节炎"),
            ("ankylosing spondylitis", "强直性脊柱炎"),
            ("AS", "强直性脊柱炎"),
            ("psoriatic arthritis", "银屑病关节炎"),
            ("PsA", "银屑病关节炎"),
            ("systemic sclerosis", "系统性硬化症"),
            ("multiple sclerosis", "多发性硬化"),
            ("MS", "多发性硬化"),
            ("amyotrophic lateral sclerosis", "肌萎缩侧索硬化"),
            ("ALS", "肌萎缩侧索硬化"),
            ("Guillain-Barré syndrome", "吉兰-巴雷综合征"),
            ("chronic kidney disease", "慢性肾脏病"),
            ("CKD", "慢性肾脏病"),
            ("end-stage renal disease", "终末期肾病"),
            ("ESRD", "终末期肾病"),
            ("glomerulonephritis", "肾小球肾炎"),
            ("immunoglobulin A nephropathy", "IgA肾病"),
            ("IgAN", "IgA肾病"),
            ("membranous nephropathy", "膜性肾病"),
            ("minimal change disease", "微小病变病"),
            ("focal segmental glomerulosclerosis", "局灶节段性肾小球硬化"),
            ("FSGS", "局灶节段性肾小球硬化"),
            ("hepatorenal syndrome", "肝肾综合征"),
            ("HRS", "肝肾综合征"),
            ("thrombotic thrombocytopenic purpura", "血栓性血小板减少性紫癜"),
            ("TTP", "血栓性血小板减少性紫癜"),
            ("hemolytic uremic syndrome", "溶血尿毒综合征"),
            ("HUS", "溶血尿毒综合征"),
            ("immune thrombocytopenic purpura", "免疫性血小板减少性紫癜"),
            ("ITP", "免疫性血小板减少性紫癜"),
            ("hemophilia A", "血友病A"),
            ("hemophilia B", "血友病B"),
            ("von Willebrand disease", "血管性血友病"),
            ("VWD", "血管性血友病"),
            ("chronic lymphocytic leukemia", "慢性淋巴细胞白血病"),
            ("CLL", "慢性淋巴细胞白血病"),
            ("chronic myeloid leukemia", "慢性髓系白血病"),
            ("CML", "慢性髓系白血病"),
            ("acute lymphoblastic leukemia", "急性淋巴细胞白血病"),
            ("ALL", "急性淋巴细胞白血病"),
            ("acute myeloid leukemia", "急性髓系白血病"),
            ("AML", "急性髓系白血病"),
            ("diffuse large B-cell lymphoma", "弥漫性大B细胞淋巴瘤"),
            ("DLBCL", "弥漫性大B细胞淋巴瘤"),
            ("follicular lymphoma", "滤泡性淋巴瘤"),
            ("FL", "滤泡性淋巴瘤"),
            ("Hodgkin lymphoma", "霍奇金淋巴瘤"),
            ("HL", "霍奇金淋巴瘤"),
            ("multiple myeloma", "多发性骨髓瘤"),
            ("MM", "多发性骨髓瘤"),
            ("myelodysplastic syndrome", "骨髓增生异常综合征"),
            ("MDS", "骨髓增生异常综合征"),
            ("polycythemia vera", "真性红细胞增多症"),
            ("PV", "真性红细胞增多症"),
            ("essential thrombocythemia", "原发性血小板增多症"),
            ("ET", "原发性血小板增多症"),
            ("myelofibrosis", "骨髓纤维化"),
            ("MF", "骨髓纤维化"),
            
            # === 法律专业术语 ===
            ("voir dire", "陪审团遴选"),
            ("habeas corpus", "人身保护令"),
            ("subpoena duces tecum", "书证传票"),
            ("prima facie", "表面证据"),
            ("res ipsa loquitur", "事实自证"),
            ("stare decisis", "遵循先例"),
            ("mens rea", "犯罪意图"),
            ("actus reus", "犯罪行为"),
            ("in pari delicto", "双方均有过错"),
            ("de facto", "事实上的"),
            ("de jure", "法律上的"),
            ("ad hoc", "特别的/专门的"),
            ("amicus curiae", "法庭之友"),
            ("certiorari", "调卷令"),
            ("sub judice", "审理中"),
            ("corpus delicti", "犯罪事实"),
            ("ex parte", "单方面的"),
            ("habeas data", "数据保护令"),
            ("in camera", "不公开审理"),
            ("in rem", "对物诉讼"),
            ("in personam", "对人诉讼"),
            ("lex loci", "行为地法"),
            ("locus standi", "诉讼资格"),
            ("mala in se", "本质邪恶的犯罪"),
            ("mala prohibita", "法律禁止的犯罪"),
            ("nolle prosequi", "不起诉"),
            ("onus probandi", "举证责任"),
            ("pro bono", "无偿服务"),
            ("pro se", "自诉"),
            ("qui tam", "公益诉讼"),
            ("res judicata", "既判事项"),
            ("sine qua non", "必要条件"),
            ("sub silentio", "默认"),
            ("ultra vires", "越权"),
            ("vexatious litigation", "恶意诉讼"),
            ("voir dire examination", "资格审查"),
            ("writ of mandamus", "履行职责令"),
            ("writ of prohibition", "禁止令"),
            ("writ of quo warranto", "问责令"),
            ("ab initio", "从一开始"),
            ("actus reus non facit reum nisi mens sit rea", "无犯意则无犯人"),
            ("audi alteram partem", "听取对方意见"),
            ("bona fide", "善意的"),
            ("caveat emptor", "买者自负"),
            ("caveat venditor", "卖者自负"),
            ("de minimis non curat lex", "法律不计较琐事"),
            ("damnum absque injuria", "损害但未违法"),
            ("expressio unius est exclusio alterius", "明示其一即排除其他"),
            ("fiat justitia ruat caelum", "天塌下来也要伸张正义"),
            ("ignorantia juris non excusat", "不知法不免责"),
            ("injuria sine damno", "违法但未造成损害"),
            ("ipsa loquitur", "本身说明问题"),
            ("jus cogens", "强行法"),
            ("lex posterior derogat priori", "后法优于前法"),
            ("lex specialis derogat generali", "特别法优于普通法"),
            ("nemo dat quod non habet", "无权处分"),
            ("nemo judex in causa sua", "任何人不得做自己案件的法官"),
            ("nulla poena sine lege", "法无明文规定不处罚"),
            ("onus probandi incumbit actori", "举证责任在原告"),
            ("parens patriae", "国家亲权"),
            ("pari passu", "同等比例"),
            ("per se", "本身"),
            ("prima facie case", "表面证据案件"),
            ("ratio decidendi", "判决理由"),
            ("rebus sic stantibus", "情势变更"),
            ("sui generis", "自成一类"),
            ("ubi jus ibi remedium", "有权利必有救济"),
            ("alibi", "不在场证明"),
            ("battery", "殴打"),
            ("assault", "威胁伤害"),
            ("burglary", "入室盗窃"),
            ("larceny", "盗窃罪"),
            ("embezzlement", "侵占罪"),
            ("false pretenses", "诈骗罪"),
            ("forgery", "伪造罪"),
            ("extortion", "敲诈勒索"),
            ("bribery", "贿赂罪"),
            ("racketeering", "敲诈勒索集团犯罪"),
            (" RICO", "反 Racketeer 影响和腐败组织法"),
            ("money laundering", "洗钱罪"),
            ("obstruction of justice", "妨碍司法公正"),
            ("contempt of court", "藐视法庭"),
            ("perjury", "伪证罪"),
            ("subornation of perjury", "唆使伪证"),
            ("misprision of felony", "包庇重罪"),
            ("accessory before the fact", "事前从犯"),
            ("accessory after the fact", "事后从犯"),
            ("accomplice", "共犯"),
            ("conspiracy", "共谋罪"),
            ("attempt", "未遂罪"),
            ("solicitation", "教唆罪"),
            ("duress", "胁迫"),
            ("entrapment", "诱捕"),
            ("necessity", "紧急避险"),
            ("self-defense", "正当防卫"),
            ("defense of others", "防卫他人"),
            ("defense of property", "防卫财产"),
            ("insanity defense", "精神失常辩护"),
            ("statute of limitations", "诉讼时效"),
            ("double jeopardy", "双重危险"),
            ("due process", "正当程序"),
            ("equal protection", "平等保护"),
            ("probable cause", "合理根据"),
            ("reasonable suspicion", "合理怀疑"),
            ("beyond a reasonable doubt", "排除合理怀疑"),
            ("preponderance of the evidence", "优势证据"),
            ("clear and convincing evidence", "清楚且有说服力的证据"),
            ("voir dire", "陪审团遴选"),
            
            # === 电竞/游戏专业术语 ===
            ("Jungler", "打野"),
            ("Gank", "抓人"),
            ("Nerf", "削弱"),
            ("Buff", "增益"),
            ("Carry", "核心输出"),
            ("Support", "辅助"),
            ("Ult / Ultimate", "大招"),
            ("Cooldown", "冷却时间"),
            ("Respawn", "重生"),
            ("CS / Creep Score", "补刀数"),
            ("Farm", "发育"),
            ("Push", "推进"),
            ("Rotate", "转线"),
            ("Split Push", "分带"),
            ("Team Fight", "团战"),
            ("Solo Kill", "单杀"),
            ("Counter Gank", "反蹲"),
            ("Invade", "入侵"),
            ("Objective", "目标"),
            ("Dragon", "小龙"),
            ("Baron Nashor", "纳什男爵/大龙"),
            ("Herald", "峡谷先锋"),
            ("Vision", "视野"),
            ("Ward", "眼位"),
            ("Control Ward", "控制守卫/真眼"),
            ("Smite", "惩戒"),
            ("Flash", "闪现"),
            ("Ignite", "点燃"),
            ("Teleport", "传送"),
            ("Marksman", "射手"),
            ("Mage", "法师"),
            ("Assassin", "刺客"),
            ("Tank", "坦克"),
            ("Bruiser", "战士"),
            ("Meta", "版本环境"),
            ("Patch", "版本更新"),
            ("OP / Overpowered", "过强"),
            ("CC / Crowd Control", "控制效果"),
            ("Stun", "眩晕"),
            ("Silence", "沉默"),
            ("Slow", "减速"),
            ("Knockup", "击飞"),
            ("Grandmaster", "傲世宗师"),
            ("Master", "超凡大师"),
            ("Diamond", "璀璨钻石"),
            ("Platinum", "华贵铂金"),
            ("Gold", "荣耀黄金"),
            ("Silver", "不屈白银"),
            ("Bronze", "英勇黄铜"),
            ("Iron", "坚韧黑铁"),
            ("Promotion", "晋级"),
            ("Demotion", "降级"),
            ("Series", "晋级赛"),
            ("BO5 / Best of Five", "五局三胜"),
            ("BO3 / Best of Three", "三局两胜"),
            ("Match", "比赛"),
            ("Tournament", "锦标赛"),
            ("League", "联赛"),
            ("Championship", "冠军赛"),
            ("Worlds", "全球总决赛"),
            ("MSI / Mid-Season Invitational", "季中邀请赛"),
            ("All-Star", "全明星赛"),
            ("Regional Finals", "区域决赛"),
            ("Qualifiers", "资格赛"),
            ("Play-ins", "入围赛"),
            ("AoE / Area of Effect", "范围伤害"),
            ("Ban", "禁用"),
            ("Pick", "选用"),
            ("True Damage", "真实伤害"),
            ("Physical Damage", "物理伤害"),
            ("Magic Damage", "魔法伤害"),
            ("Armor Penetration", "护甲穿透"),
            ("Magic Penetration", "法术穿透"),
            ("Life Steal", "生命偷取"),
            ("Spell Vamp", "法术吸血"),
            ("Critical Strike", "暴击"),
            ("Cooldown Reduction", "冷却缩减"),
            ("Wave Clear", "清线"),
            ("Wave Management", "兵线管理"),
            ("Freeze", "控线"),
            ("All-in", "全力进攻"),
            ("Trade", "换血"),
            ("Poke", "消耗"),
            ("Dive", "越塔"),
            ("Backdoor", "偷家"),
            ("Roam", "游走"),
            ("Engage", "开团"),
            ("Disengage", "撤退"),
            ("Peel", "保护"),
            ("Flank", "侧翼包抄"),
            ("Zone", "区域控制"),
            ("Macro", "大局观"),
            ("Micro", "操作细节"),
            ("Mechanics", "操作技巧"),
            ("Flick", "甩枪"),
            ("Tracking", "跟枪"),
            ("Spray Control", "压枪"),
            ("Recoil", "后坐力"),
            ("Crosshair", "准星"),
            ("Sensitivity", "灵敏度"),
            ("DPI", "鼠标灵敏度"),
            ("AFK / Away From Keyboard", "挂机"),
            ("KDA / Kill Death Assist", "击杀死亡助攻比"),
            ("MMR / Match Making Rating", "匹配分"),
            ("LP / League Points", "联赛积分"),
            ("BO5 / Best of Five", "五局三胜"),
            ("ARAM / All Random All Mid", "极地大乱斗"),
            ("URF / Ultra Rapid Fire", "无限火力"),
            ("One for All", "克隆大作战"),
            ("Teamfight Tactics", "云顶之弈"),
            ("Auto Chess", "自走棋"),
            ("LoR / Legends of Runeterra", "符文大地传说"),
            ("Wild Rift", "英雄联盟手游"),
    
            # === 多义词/歧义表达 ===
            ("counting stars", "数星星"),
            ("across the lines", "越过这条线"),
            ("across the lines", "跨越边界"),
            ("every turn", "每次"),
            ("every turn", "次次"),
            ("every turn", "总是"),
            ("every turn", "每个转弯处"),
            ("Driver", "驱动程序"),
            ("Driver", "司机"),
            ("Apple (the company)", "苹果公司"),
            ("Apple (fruit)", "苹果"),
            ("Windows (OS)", "Windows系统"),
            ("Windows (glass)", "窗户"),
            ("China (country)", "中国"),
            ("China (porcelain)", "瓷器"),
            ("Turkey (country)", "土耳其"),
            ("Turkey (bird)", "火鸡"),
            ("Lead (metal)", "铅"),
            ("Lead (verb)", "领导"),
            ("Date (calendar)", "日期"),
            ("Date (romantic)", "约会"),
            ("Date (fruit)", "海枣"),
            ("Spring (season)", "春天"),
            ("Spring (metal)", "弹簧"),
            ("Spring (water)", "泉水"),
            ("Bank (financial)", "银行"),
            ("Bank (river)", "河岸"),
            ("Charge (electric)", "充电"),
            ("Charge (attack)", "冲锋"),
            ("Charge (fee)", "费用"),
            ("Rock (music)", "摇滚"),
            ("Rock (stone)", "岩石"),
            ("Rock (sugar)", "冰糖"),
            ("Mouse (animal)", "老鼠"),
            ("Mouse (computer)", "鼠标"),
            ("Deck (card)", "卡组"),
            ("Deck (ship)", "甲板"),
            ("Deck (yard)", "露台"),
            ("Chip (computer)", "芯片"),
            ("Chip (snack)", "薯片"),
            ("Chip (golf)", "切杆"),
            ("Bug (insect)", "虫子"),
            ("Bug (computer)", "漏洞"),
            ("Feature (movie)", "正片"),
            ("Feature (characteristic)", "特性"),
            ("Feature (software)", "功能"),
            ("Frame (picture)", "相框"),
            ("Frame (video)", "帧"),
            ("Frame (structure)", "框架"),
            ("Frame (accuse)", "陷害"),
            ("Table (furniture)", "桌子"),
            ("Table (data)", "表格"),
            ("Table (postpone)", "搁置"),
            ("Line (phone)", "电话线路"),
            ("Line (queue)", "队伍"),
            ("Line (text)", "行"),
            ("Line (border)", "边界"),
            ("Check (verify)", "检查"),
            ("Check (payment)", "支票"),
            ("Check (chess)", "将军"),
            ("Check (mark)", "勾号"),
            ("Current (electric)", "电流"),
            ("Current (recent)", "当前的"),
            ("Current (water)", "水流"),
            ("Key (lock)", "钥匙"),
            ("Key (music)", "调"),
            ("Key (important)", "关键的"),
            ("Key (keyboard)", "按键"),
            ("Left (direction)", "左边"),
            ("Left (remaining)", "剩下的"),
            ("Left (departed)", "离开"),
            ("Right (direction)", "右边"),
            ("Right (correct)", "正确的"),
            ("Right (entitlement)", "权利"),
            ("Right (politics)", "右翼"),
            ("Screen (display)", "屏幕"),
            ("Screen (filter)", "筛选"),
            ("Screen (hide)", "掩护"),
            ("Screen (test)", "筛查"),
            ("Server (computer)", "服务器"),
            ("Server (waiter)", "服务员"),
            ("Server (tennis)", "发球者"),
            ("Station (train)", "车站"),
            ("Station (position)", "岗位"),
            ("Station (police)", "警局"),
            ("Station (TV)", "电视台"),
            ("Test (exam)", "考试"),
            ("Test (trial)", "测试"),
            ("Test (cup)", "世界杯"),
            ("Test (chemical)", "化验"),
            ("Trip (journey)", "旅行"),
            ("Trip (fall)", "绊倒"),
            ("Trip (drug)", "幻觉"),
            ("Trip (switch)", "跳闸"),
            ("Type (write)", "打字"),
            ("Type (category)", "类型"),
            ("Type (blood)", "血型"),
            ("Type (keyboard)", "键盘类型"),
            ("Well (water)", "井"),
            ("Well (healthy)", "健康的"),
            ("Well (done)", "做得好"),
            ("Well (oil)", "油井"),
            ("Yard (unit)", "码"),
            ("Yard (garden)", "院子"),
            ("Yard (train)", "调车场"),
            ("Yard (ship)", "船坞"),
        ]
        
        # ==========================================
        # 2. 扩充后的多义词上下文模板
        # 用于生成更多样的训练语料
        # ==========================================
        self.context_templates = {
            # --- 人名 vs 时间 ---
            "night_person": [
                "{name} is streaming on Twitch tonight.",
                "{name} won the championship last year.",
                "The commentator praised {name}'s performance.",
                "Did you see {name}'s play?",
                "{name} is known for his aggressive style."
            ],
            "night_time": [
                "It's a beautiful night outside.",
                "The stars are bright at night.",
                "We stayed up all night studying.",
                "The shop is closed at night.",
                "Good night, sleep tight."
            ],
            # --- 鹤 vs 起重机 ---
            "crane_bird": [
                "A {species} flew over the lake.",
                "We observed a {species} in the wetlands.",
                "The {species} is an endangered species.",
                "Look at that beautiful crane flying.",
                "The crane spreads its wings."
            ],
            "crane_machine": [
                "The construction crane lifted the steel beams.",
                "They used a crane to move the cargo.",
                "A tower crane dominates the skyline.",
                "The operator controls the crane.",
                "Safety around the crane is mandatory."
            ],
            # --- 苹果公司 vs 水果 (新增) ---
            "apple_tech": [
                "Apple released a new iPhone yesterday.",
                "I bought shares in Apple stock.",
                "Apple's headquarters is in Cupertino.",
                "The new Apple update fixed the bug.",
                "He works as an engineer at Apple."
            ],
            "apple_fruit": [
                "I ate a red apple for lunch.",
                "The apple pie smells delicious.",
                "An apple a day keeps the doctor away.",
                "She picked an apple from the tree.",
                "This apple is very sweet."
            ],
            # --- 驱动程序 vs 司机 (新增) ---
            "driver_tech": [
                "I need to update my graphics driver.",
                "The printer driver is missing.",
                "Download the latest driver from the website.",
                "Driver conflict caused the crash.",
                "Install the audio driver first."
            ],
            "driver_human": [
                "The taxi driver was very polite.",
                "He is a formula one driver.",
                "Call a designated driver.",
                "The bus driver stopped at the station.",
                "She is a safe driver."
            ],

            # --- 其他 ---
            "across_lines_literal": [
                "The athlete sprinted across the lines.",
                "The child drew a line and jumped across the lines.",
                "The car crossed across the lines on the road.",
                "Make sure you don't step across the lines.",
                "The finish line is clear—across the lines now!"
            ],
            "across_lines_abstract": [
                "The team aims to across the lines of industry limits.",
                "Her research across the lines of science and humanities.",
                "To innovate, we must across the lines of conventional wisdom.",
                "The policy helps across the lines of regional differences.",
                "Collaboration can across the lines of organizational silos."
            ],
             # every turn 各含义的上下文模板
            "every_turn_each_time": [
                "We learn something new at every turn.",
                "At every turn, we face new challenges.",
                "She surprises us at every turn.",
                "Problems arise at every turn.",
                "Opportunities appear at every turn."
            ],
            "every_turn_always": [
                "He finds fault at every turn.",
                "Something goes wrong at every turn.",
                "She complains at every turn.",
                "They change plans at every turn.",
                "Difficulties come up at every turn."
            ],
            "every_turn_location": [
                "There are beautiful views at every turn of the river.",
                "We saw wildlife at every turn in the forest.",
                "The path offers surprises at every turn.",
                "Historical markers stand at every turn of the trail.",
                "Flowers bloom at every turn along the road."
            ]
        }
    
    def generate_augmented_samples(self, src: str, tgt: str, num_variants: int = 5) -> List[Dict]:
        """为每个术语对生成多样化的上下文样本"""
        samples = []
        
        # 基础样本
        samples.append({"src": src, "tgt": tgt})
        
        # 生成带上下文的变体
        context_prefixes = [
            "In the field of biology, ",
            "According to experts, ",
            "The research shows that ",
            "It is important to note that ",
            "Scientists discovered that ",
        ]
        
        context_suffixes = [
            " is a critical topic.",
            " requires further study.",
            " has significant implications.",
            " is widely recognized.",
            " attracts global attention.",
        ]
        
        for i in range(min(num_variants, len(context_prefixes))):
            augmented_src = f"{context_prefixes[i]}{src}{context_suffixes[i]}"
            augmented_tgt = f"{context_prefixes[i]}{tgt}{context_suffixes[i]}"
            samples.append({"src": augmented_src, "tgt": augmented_tgt})
        
        return samples
    
    def build_professional_dataset(self, train_ratio: float = 0.9) -> Dict[str, List]:
        """构建专业词汇数据集"""
        all_samples = []
        
        # 1. 从术语库生成增强样本
        for src, tgt in self.terminology_pairs:
            variants = self.generate_augmented_samples(src, tgt, num_variants=8)
            all_samples.extend(variants)
        
        # 2. 添加多义词辨析样本
        # Night (人名 vs 时间)
        all_samples.extend([
            {"src": "Night is the best player on the team.", 
             "tgt": "Night 是队伍中最好的选手。"},
            {"src": "Night won the MVP award.", 
             "tgt": "Night 获得了 MVP 奖。"},
            {"src": "It was a cold night.", 
             "tgt": "这是一个寒冷的夜晚。"},
            {"src": "Good night, everyone!", 
             "tgt": "大家晚安！"},
        ])
        
        # Crane (鹤 vs 起重机)
        all_samples.extend([
            {"src": "The Sandhill Crane is migrating south.", 
             "tgt": "沙丘鹤正在向南迁徙。"},
            {"src": "A Whooping Crane was spotted in the reserve.", 
             "tgt": "在保护区发现了一只美洲鹤。"},
            {"src": "The crane lifted the container onto the ship.", 
             "tgt": "起重机将集装箱吊上了船。"},
            {"src": "They need a mobile crane for this project.", 
             "tgt": "这个项目需要一台移动起重机。"},
        ])
        
        # Night (人名 vs 时间)
        all_samples.extend([
            {"src": "Night is the best player on the team.", "tgt": "Night 是队伍中最好的选手。"},
            {"src": "Night won the MVP award.", "tgt": "Night 获得了 MVP 奖。"},
            {"src": "It was a cold night.", "tgt": "这是一个寒冷的夜晚。"},
        ])
        
        # Apple (公司 vs 水果)
        all_samples.extend([
            {"src": "Apple announced new earnings.", "tgt": "苹果公司公布了新收益。"},
            {"src": "Apple stock is rising.", "tgt": "苹果公司的股票正在上涨。"},
            {"src": "She is eating an apple.", "tgt": "她正在吃苹果。"},
            {"src": "Rotten apple.", "tgt": "烂苹果。"},
        ])

        # Driver (驱动 vs 司机)
        all_samples.extend([
            {"src": "Please install the USB driver.", "tgt": "请安装 USB 驱动程序。"},
            {"src": "NVIDIA driver update.", "tgt": "英伟达驱动程序更新。"},
            {"src": "The taxi driver is late.", "tgt": "出租车司机迟到了。"},
        ])

        # across the lines (字面：越过具体的线 vs 引申：跨越抽象边界)
        all_samples.extend([
            # 字面含义：具体场景（物理线、规则线等）
            {"src": "The runner across the lines and won the race.", "tgt": "跑步选手越过这条线，赢得了比赛。"},
            {"src": "Don't step across the lines on the floor.", "tgt": "不要越过地板上的这条线。"},
            {"src": "The ball rolled across the lines of the court.", "tgt": "球滚过了球场的界线。"},
    
            # 引申含义：抽象场景（领域、立场、边界等）
            {"src": "They decided to across the lines of traditional thinking.", "tgt": "他们决定跨越传统思维的边界。"},
            {"src": "The project across the lines between technology and art.", "tgt": "这个项目跨越了技术与艺术的边界。"},
            {"src": "We need to across the lines of communication barriers.", "tgt": "我们需要跨越沟通障碍的边界。"},
        ])

        all_samples.extend([
            # 每次/次次含义
            {"src": "We meet new people at every turn.", "tgt": "我们每次都遇到新朋友。"},
            {"src": "Success comes at every turn.", "tgt": "成功次次降临。"},
            {"src": "Chances come at every turn.", "tgt": "机会每次都有。"},
    
            # 总是/事事含义
            {"src": "He argues at every turn.", "tgt": "他总是在争辩。"},
            {"src": "Something goes wrong at every turn.", "tgt": "事事都出问题。"},
            {"src": "She finds mistakes at every turn.", "tgt": "她总是挑错。"},
    
            # 每个转弯处/到处含义（地理空间）
            {"src": "There are shops at every turn in the city.", "tgt": "城市里到处都是商店。"},
            {"src": "The path has benches at every turn.", "tgt": "小路的每个转弯处都有长椅。"},
            {"src": "We saw waterfalls at every turn on the hike.", "tgt": "徒步路上每个转弯处都能看到瀑布。"},
        ])


        
        # 3. 打乱并划分训练/验证集
        random.shuffle(all_samples)
        split_idx = int(len(all_samples) * train_ratio)
        
        return {
            "train": all_samples[:split_idx],
            "eval": all_samples[split_idx:]
        }
    
    def save_datasets(self, datasets: Dict[str, List], format: str = "json"):
        """保存数据集到文件"""
        for split, data in datasets.items():
            output_path = f"{self.output_dir}/{split}/data.json"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ {split.upper()} 数据集已保存: {output_path} ({len(data)} 样本)")
    
    def add_custom_terminology(self, term_pairs: List[tuple]):
        """添加自定义术语对"""
        self.terminology_pairs.extend(term_pairs)
        print(f"✅ 已添加 {len(term_pairs)} 个自定义术语")
    
    def export_terminology_dict(self, output_path: str = "data/terminology.json"):
        """导出术语词典供后续使用"""
        terminology_dict = {src: tgt for src, tgt in self.terminology_pairs}
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(terminology_dict, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 术语词典已导出: {output_path}")


# ============================================================
# 使用示例
# ============================================================
if __name__ == "__main__":
    builder = FinetuneDatasetBuilder()
    
    # 1. 添加你的自定义术语（根据实际需求扩展）
    custom_terms = [
        ("Barack Obama", "巴拉克·奥巴马"),  # 示例：确保人名不被误译
        ("neural network", "神经网络"),
        ("machine learning", "机器学习")
        # 在这里添加更多你遇到的错误翻译案例
    ]
    builder.add_custom_terminology(custom_terms)
    
    # 2. 构建数据集
    datasets = builder.build_professional_dataset(train_ratio=0.85)
    
    # 3. 保存到文件
    builder.save_datasets(datasets)
    
    # 4. 导出术语词典
    builder.export_terminology_dict()
    
    print("\n" + "="*50)
    print("📊 数据集统计:")
    print(f"   训练样本: {len(datasets['train'])}")
    print(f"   验证样本: {len(datasets['eval'])}")
    print(f"   总计: {len(datasets['train']) + len(datasets['eval'])}")
    print("="*50)