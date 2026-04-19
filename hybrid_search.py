"""
hybrid_search.py — 混合检索模块
= 向量检索（FAISS）+ BM25 关键词检索
最终通过 RRF (Reciprocal Rank Fusion) 合并排分

Embedding 后端：
  bge    → 硅基流动 SiliconFlow  BAAI/bge-m3（默认，中文优秀）
  openai → OpenAI text-embedding-3-large（备用）
  local  → 本地 BGE 模型（备用）
"""

import os
import json
import pickle
import time
from typing import List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from rich.console import Console

from rag_config import RAGConfig, DEFAULT_CONFIG
from document_processor import Document

console = Console()


# ─────────────────────────────────────────────
# 中文 / 英文分词
# ─────────────────────────────────────────────
def tokenize(text: str, lang: str = "zh") -> List[str]:
    if lang == "zh":
        import jieba
        return list(jieba.cut(text))
    else:
        return text.lower().split()


# ─────────────────────────────────────────────
# Embedding 后端：BGE API（硅基流动）
# ─────────────────────────────────────────────
class BGEEmbedder:
    """
    通过硅基流动调用 BAAI/bge-m3
    接口与 OpenAI Embedding 完全兼容，直接复用 openai 客户端
    bge-m3 支持中英日韩等 100+ 语言，1024 维向量
    """

    def __init__(self, config: RAGConfig):
        import httpx
        from openai import OpenAI
        self.client = OpenAI(
            api_key=config.siliconflow_api_key,
            base_url=config.embedding.bge_base_url,
            http_client=httpx.Client(verify=False),
        )
        self.cfg = config.embedding

    def _encode_batch(self, texts: List[str], retry: int = 5) -> List[List[float]]:
        """带指数退避重试，自动处理 429 限速错误"""
        import openai
        wait = 10  # 初始等待秒数
        for attempt in range(retry):
            try:
                response = self.client.embeddings.create(
                    model=self.cfg.bge_model,
                    input=texts,
                    encoding_format="float",
                )
                return [item.embedding for item in response.data]
            except openai.RateLimitError:
                if attempt == retry - 1:
                    raise
                console.print(f"  [yellow]⚠ 触发限速，等待 {wait}s 后重试 ({attempt+1}/{retry})...[/yellow]")
                time.sleep(wait)
                wait = min(wait * 2, 120)  # 最长等待 120s
            except Exception as e:
                if attempt == retry - 1:
                    raise
                console.print(f"  [yellow]⚠ 请求失败({e})，等待 {wait}s 重试...[/yellow]")
                time.sleep(wait)
                wait = min(wait * 2, 120)

    # 每批之间的最小间隔（秒），仅建库时使用，查询时不限速
    BATCH_INTERVAL = 2.0

    def _encode_batch_direct(self, texts: List[str]) -> List[List[float]]:
        """查询时用：不限速，直接单次请求所有文本（数量少）"""
        return self._encode_batch(texts)

    def encode(self, texts: List[str]) -> np.ndarray:
        all_embeddings = []
        batch_size = self.cfg.batch_size
        total = (len(texts) - 1) // batch_size + 1
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            batch_num = i // batch_size + 1
            console.print(f"  [dim]BGE Embedding batch {batch_num}/{total}[/dim]")
            t0 = time.time()
            all_embeddings.extend(self._encode_batch(batch))
            # 控制请求速率：每批至少间隔 BATCH_INTERVAL 秒
            elapsed = time.time() - t0
            sleep_time = max(0, self.BATCH_INTERVAL - elapsed)
            if sleep_time > 0 and i + batch_size < len(texts):
                time.sleep(sleep_time)
        arr = np.array(all_embeddings, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.maximum(norms, 1e-10)

    def encode_query(self, query: str) -> np.ndarray:
        """查询时不需要限速，直接请求不等待"""
        response = self.client.embeddings.create(
            model=self.cfg.bge_model,
            input=[query],
            encoding_format="float",
        )
        arr = np.array([response.data[0].embedding], dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.maximum(norms, 1e-10)


# ─────────────────────────────────────────────
# Embedding 后端：OpenAI API（备用）
# ─────────────────────────────────────────────
class OpenAIEmbedder:
    def __init__(self, config: RAGConfig):
        from openai import OpenAI
        self.client = OpenAI(api_key=config.openai_api_key)
        self.cfg = config.embedding

    def _encode_batch(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(
            model=self.cfg.openai_model,
            input=texts,
            dimensions=self.cfg.openai_dimensions,
        )
        return [item.embedding for item in response.data]

    def encode(self, texts: List[str]) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), 100):
            batch = texts[i: i + 100]
            all_embeddings.extend(self._encode_batch(batch))
            if i + 100 < len(texts):
                time.sleep(0.1)
        arr = np.array(all_embeddings, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.maximum(norms, 1e-10)

    def encode_query(self, query: str) -> np.ndarray:
        """查询时不需要限速，直接请求不等待"""
        response = self.client.embeddings.create(
            model=self.cfg.bge_model,
            input=[query],
            encoding_format="float",
        )
        arr = np.array([response.data[0].embedding], dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.maximum(norms, 1e-10)


# ─────────────────────────────────────────────
# 向量检索器（FAISS）
# ─────────────────────────────────────────────
class VectorRetriever:
    """
    根据 config.embedding.api_provider 自动选择 Embedding 后端：
      "bge"    → BGE API（硅基流动）—— 默认，中文最优
      "openai" → OpenAI text-embedding-3-large
      "local"  → 本地 BGE 模型
    """

    def __init__(self, config: RAGConfig = DEFAULT_CONFIG):
        self.cfg = config.embedding
        self._config = config
        self.embedder = None
        self.index = None
        self.doc_ids: List[str] = []

    def _load_embedder(self):
        if self.embedder is not None:
            return
        provider = self.cfg.api_provider if self.cfg.mode == "api" else "local"
        if provider == "bge":
            console.print(f"[cyan]Embedding: BGE API ({self.cfg.bge_model}) via 硅基流动[/cyan]")
            self.embedder = BGEEmbedder(self._config)
        elif provider == "openai":
            console.print(f"[cyan]Embedding: OpenAI API ({self.cfg.openai_model})[/cyan]")
            self.embedder = OpenAIEmbedder(self._config)
        else:
            console.print(f"[cyan]Embedding: 本地模型 ({self.cfg.local_model})[/cyan]")
            from FlagEmbedding import FlagModel

            class _LocalEmb:
                def __init__(self, cfg):
                    self.model = FlagModel(
                        cfg.local_model,
                        query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                        use_fp16=(cfg.local_device == "cuda"),
                    )
                def encode(self, texts):
                    return np.array(self.model.encode(texts, batch_size=64, normalize_embeddings=True), dtype=np.float32)
                def encode_query(self, query):
                    return np.array(self.model.encode_queries([query], normalize_embeddings=True), dtype=np.float32)

            self.embedder = _LocalEmb(self.cfg)

    def build_index(self, docs: List[Document]):
        import faiss
        self._load_embedder()
        texts = [d.content for d in docs]
        self.doc_ids = [d.doc_id for d in docs]
        console.print(f"[cyan]编码 {len(texts)} 个文档块...[/cyan]")
        embeddings = self.embedder.encode(texts)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        console.print(f"[green]向量索引构建完成，维度={dim}，共 {self.index.ntotal} 条[/green]")

    def search(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        self._load_embedder()
        q_emb = self.embedder.encode_query(query)
        return self.search_by_vector(q_emb, top_k)

    def search_by_vector(self, q_emb: np.ndarray, top_k: int = 20) -> List[Tuple[str, float]]:
        """用已算好的向量直接检索，避免重复调用 embedding API"""
        scores, indices = self.index.search(q_emb, top_k)
        return [(self.doc_ids[idx], float(scores[0][i]))
                for i, idx in enumerate(indices[0]) if idx >= 0]

    def save(self):
        import faiss
        os.makedirs(os.path.dirname(self.cfg.index_path), exist_ok=True)
        faiss.write_index(self.index, self.cfg.index_path)
        with open(self.cfg.meta_path, "w") as f:
            json.dump({"doc_ids": self.doc_ids}, f)
        console.print(f"[green]向量索引已保存: {self.cfg.index_path}[/green]")

    def load(self):
        import faiss
        self.index = faiss.read_index(self.cfg.index_path)
        with open(self.cfg.meta_path) as f:
            self.doc_ids = json.load(f)["doc_ids"]
        console.print(f"[green]向量索引已加载，共 {self.index.ntotal} 条[/green]")


# ─────────────────────────────────────────────
# BM25 检索器（纯本地，无需 API）
# ─────────────────────────────────────────────
class BM25Retriever:
    """
    基于 rank_bm25 的关键词检索器
    中文用 jieba 分词，英文按空格分词
    """

    def __init__(self, config: RAGConfig = DEFAULT_CONFIG):
        self.cfg = config.bm25
        self.bm25 = None
        self.doc_ids: List[str] = []

    def build_index(self, docs: List[Document]):
        self.doc_ids = [d.doc_id for d in docs]
        console.print("[cyan]构建 BM25 索引（本地，无需 API）...[/cyan]")
        tokenized = [tokenize(d.content, self.cfg.language) for d in docs]
        self.bm25 = BM25Okapi(tokenized)
        console.print(f"[green]BM25 索引构建完成，共 {len(self.doc_ids)} 条[/green]")

    def search(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        tokens = tokenize(query, self.cfg.language)
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.doc_ids[i], float(scores[i])) for i in top_indices if scores[i] > 0]

    def save(self):
        os.makedirs(os.path.dirname(self.cfg.index_path), exist_ok=True)
        with open(self.cfg.index_path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "doc_ids": self.doc_ids}, f)
        console.print(f"[green]BM25 索引已保存: {self.cfg.index_path}[/green]")

    def load(self):
        with open(self.cfg.index_path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.doc_ids = data["doc_ids"]
        console.print(f"[green]BM25 索引已加载，共 {len(self.doc_ids)} 条[/green]")


# ─────────────────────────────────────────────
# 混合检索器（RRF 融合）
# ─────────────────────────────────────────────
class HybridRetriever:
    """
    融合向量检索 + BM25 检索
    使用 RRF (Reciprocal Rank Fusion) 合并排名
    Final_score = Σ 1/(k + rank_i)，k=60 是经验值，比加权更鲁棒
    """

    def __init__(self, config: RAGConfig = DEFAULT_CONFIG):
        self.cfg = config.search
        self.vector_retriever = VectorRetriever(config)
        self.bm25_retriever = BM25Retriever(config)
        self._doc_map: dict = {}

    def build(self, docs: List[Document]):
        self._doc_map = {d.doc_id: d for d in docs}
        self.vector_retriever.build_index(docs)
        self.bm25_retriever.build_index(docs)

    def save(self):
        self.vector_retriever.save()
        self.bm25_retriever.save()

    def load(self, docs: List[Document]):
        self._doc_map = {d.doc_id: d for d in docs}
        self.vector_retriever.load()
        self.bm25_retriever.load()

    def _rrf_merge(
        self,
        vector_results: List[Tuple[str, float]],
        bm25_results: List[Tuple[str, float]],
        k: int = 60,
    ) -> List[Tuple[str, float]]:
        scores: dict = {}
        for rank, (doc_id, _) in enumerate(vector_results):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        for rank, (doc_id, _) in enumerate(bm25_results):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[: self.cfg.top_k_merged]

    def search(self, query: str) -> List[Document]:
        vec_results = self.vector_retriever.search(query, self.cfg.top_k_vector)
        bm25_results = self.bm25_retriever.search(query, self.cfg.top_k_bm25)
        merged = self._rrf_merge(vec_results, bm25_results)
        return [self._doc_map[doc_id] for doc_id, _ in merged if doc_id in self._doc_map]

    def multi_query_search(self, queries: List[str]) -> List[Document]:
        """
        多查询检索，合并去重。
        优化：把所有查询的 embedding 合并成一次 API 调用，减少网络往返。
        """
        if not queries:
            return []

        # 确保 embedder 已加载（懒加载机制）
        self.vector_retriever._load_embedder()

        # 一次性 encode 所有查询（单次 API 调用）
        all_vecs = self.vector_retriever.embedder._encode_batch_direct(queries)

        seen_ids: dict = {}
        all_docs = []
        for i, query in enumerate(queries):
            # 向量检索（用已算好的向量）
            vec = np.array([all_vecs[i]], dtype=np.float32)
            norms = np.linalg.norm(vec, axis=1, keepdims=True)
            vec = vec / np.maximum(norms, 1e-10)
            vec_results = self.vector_retriever.search_by_vector(vec, self.cfg.top_k_vector)
            # BM25 检索
            bm25_results = self.bm25_retriever.search(query, self.cfg.top_k_bm25)
            # RRF 合并
            merged = self._rrf_merge(vec_results, bm25_results)
            for doc_id, _ in merged:
                if doc_id not in seen_ids and doc_id in self._doc_map:
                    seen_ids[doc_id] = True
                    all_docs.append(self._doc_map[doc_id])

        return all_docs[: self.cfg.top_k_merged * 2]
