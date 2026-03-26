"""
context_compressor.py — 上下⽂压缩层
⽬的： 减少 token 消耗， 去除噪声， 让 LLM 专注于核⼼内容
⽀持两种模式：
- rule: 规则压缩（ 关键词过滤 + token 截断）
- llm: 模型压缩（ ⼩模型抽取相关句⼦）
"""

import re
from typing import List

from rich.console import Console

from rag_config import RAGConfig, DEFAULT_CONFIG
from document_processor import Document

console = Console()

# ─────────────────────────────────────────────
# 规则压缩（ 快速、 ⽆额外模型）
# ─────────────────────────────────────────────
class RuleCompressor:
    """
    基于关键词匹配的规则压缩
    策略：
    1. 按句⼦分割
    2. 保留包含查询关键词的句⼦
    3. 截断到最⼤ token 数
    """
    def __init__(self, config: RAGConfig = DEFAULT_CONFIG):
        self.max_tokens = config.compression.max_tokens_per_chunk
        self.lang = config.bm25.language

    def _sentence_split(self, text: str) -> List[str]:
        """按中英⽂标点切句"""
        sentences = re.split(r"(?<=[。 ！ ？ .!?\n])", text)
        return [s.strip() for s in sentences if s.strip()]

    def _extract_keywords(self, query: str) -> List[str]:
        """提取查询关键词"""
        if self.lang == "zh":
            import jieba
            words = list(jieba.cut(query))
            # 过滤停⽤词（ 简单版）
            stopwords = {"的", "了", "在", "是", "我", "有", "和", "就", "不", "⼈"}
            return [w for w in words if len(w) > 1 and w not in stopwords]
        else:
            return [w.lower() for w in query.split() if len(w) > 3]

    def _token_count(self, text: str) -> int:
        return len(text) // 2 if self.lang == "zh" else len(text.split())

    def compress(self, query: str, doc: Document) -> str:
        """压缩单个⽂档块"""
        keywords = self._extract_keywords(query)
        if not keywords:
            return doc.content[: self.max_tokens * 2]  # fallback

        sentences = self._sentence_split(doc.content)
        # 按关键词命中数打分
        scored = []
        for sent in sentences:
            hits = sum(1 for kw in keywords if kw in sent)
            scored.append((sent, hits))

        # 保留有命中或位置靠前的句⼦（ 保留语序）
        selected = []
        token_count = 0
        for sent, hits in scored:
            if hits > 0 or len(selected) < 2:  # 保留前两句保持上下⽂
                tc = self._token_count(sent)
                if token_count + tc > self.max_tokens:
                    break
                selected.append(sent)
                token_count += tc

        return "".join(selected) if selected else doc.content[: self.max_tokens * 2]

# ─────────────────────────────────────────────
# LLM 抽取式压缩
# ─────────────────────────────────────────────
class LLMCompressor:
    """
    使⽤⼩模型从 chunk 中抽取与问题最相关的句⼦
    推荐： Qwen3-7B（ 在 V100 32G 上可运⾏）
    """
    COMPRESS_PROMPT = """你的任务是从以下⽂本中， 仅抽取与问题直接相关的句⼦。不要改写或总结， 直接抽取原⽂句⼦。
如果没有相关内容，  输出"⽆相关内容"。
问题： {query}

⽂本：
{content}

请仅输出抽取的句⼦， 不要任何解释： """

    def __init__(self, config: RAGConfig = DEFAULT_CONFIG):
        self.cfg = config.compression
        self.tokenizer = None
        self.model = None

    def _load(self):
        if self.model is not None:
            return
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        console.print(f"[cyan]加载压缩模型: {self.cfg.local_model}[/cyan]")
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.local_model, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.local_model,
            device_map="cuda",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

    def compress(self, query: str, doc: Document) -> str:
        self._load()
        import torch
        prompt = self.COMPRESS_PROMPT.format(query=query, content=doc.content)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = inputs.to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=256, temperature=0.1)
        result = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return result.strip() if result.strip() else doc.content

# ─────────────────────────────────────────────
# 统⼀接⼝
# ─────────────────────────────────────────────
class ContextCompressor:
    """
    上下⽂压缩主类
    对 Rerank 后的 Top K ⽂档块进⾏压缩， 返回最终上下⽂字符串
    """
    def __init__(self, config: RAGConfig = DEFAULT_CONFIG):
        self.cfg = config.compression
        if self.cfg.mode == "llm":
            self._compressor = LLMCompressor(config)
        else:
            self._compressor = RuleCompressor(config)

    def compress(self, query: str, docs: List[Document]) -> List[dict]:
        """
        压缩并格式化⽂档块， 返回 list， 每项含 images 字段（ 来⾃ chunk 绑定的图⽚路径）
        """
        result = []
        for i, doc in enumerate(docs):
            compressed = self._compressor.compress(query, doc)
            result.append({
                "index": i + 1,
                "source": doc.source,
                "chunk_id": doc.chunk_index,
                "content": compressed,
                "images": doc.metadata.get("images", []),
            })
        img_total = sum(len(r["images"]) for r in result)
        msg = f"\n[bold cyan]✂ Context Compression[/bold cyan]: {len(docs)} 块 → {len(result)} 块"
        if img_total:
            msg += f"， 关联 [yellow]{img_total}[/yellow] 张图⽚"
        console.print(msg)
        return result

    def format_context(self, compressed_chunks: List[dict]) -> str:
        """将压缩后的块格式化为 LLM 可⽤的上下⽂字符串"""
        lines = []
        for chunk in compressed_chunks:
            lines.append(f"【 来源 {chunk['index']}】 {chunk['source']}")
            lines.append(chunk['content'])
            lines.append("")
        return "\n".join(lines)