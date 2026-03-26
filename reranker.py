"""
reranker.py — 重排序层
这是精准度提升最关键的一层。

Reranker 后端：
bge    → 硅基流动 SiliconFlow  BAAI/bge-reranker-v2-m3（默认，中文最优）
cohere → Cohere rerank-multilingual-v3.0（备用）
local  → 本地 BGE-reranker-large（备用）

注意：硅基流动的 Rerank API 与 Cohere API 格式完全兼容
"""

import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from typing import List, Tuple

from rich.console import Console

from rag_config import RAGConfig, DEFAULT_CONFIG
from document_processor import Document

console = Console()

# ─────────────────────────────────────────────
# BGE Reranker API 后端（硅基流动）
# ─────────────────────────────────────────────

class BGEReranker:
    """
    调用硅基流动的 BGE Reranker API
    端点：https://api.siliconflow.cn/v1/rerank
    请求格式与 Cohere Rerank 完全一致
    """
    def __init__(self, config: RAGConfig):
        self.cfg = config.reranker
        self.api_key = config.siliconflow_api_key
        self.url = self.cfg.bge_base_url
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def rerank(self, query: str, docs: List[Document]) -> List[Tuple[Document, float]]:
        payload = {
            "model": self.cfg.bge_model,
            "query": query,
            "documents": [doc.content for doc in docs],
            "top_n": self.cfg.top_k,
            "return_documents": False,
        }
        resp = requests.post(self.url, json=payload, headers=self.headers, timeout=30, verify=False)
        resp.raise_for_status()
        data = resp.json()

        ranked = []
        for result in data["results"]:
            doc = docs[result["index"]]
            score = result["relevance_score"]
            ranked.append((doc, score))
        # results 已按 relevance_score 降序排列
        return ranked

# ─────────────────────────────────────────────
# Cohere Reranker API 后端（备用）
# ─────────────────────────────────────────────

class CohereReranker:
    def __init__(self, config: RAGConfig):
        import cohere
        self.client = cohere.ClientV2(api_key=config.cohere_api_key)
        self.cfg = config.reranker

    def rerank(self, query: str, docs: List[Document]) -> List[Tuple[Document, float]]:
        response = self.client.rerank(
            model=self.cfg.cohere_model,
            query=query,
            documents=[doc.content for doc in docs],
            top_n=self.cfg.top_k,
        )
        return [(docs[r.index], r.relevance_score) for r in response.results]

# ─────────────────────────────────────────────
# 本地 BGE Reranker 后端（备用）
# ─────────────────────────────────────────────

class LocalReranker:
    def __init__(self, config: RAGConfig):
        from FlagEmbedding import FlagReranker
        self.cfg = config.reranker
        console.print(f"[cyan]加载本地 Reranker: {self.cfg.local_model}[/cyan]")
        self.model = FlagReranker(self.cfg.local_model,
                                  use_fp16=(self.cfg.local_device == "cuda"))

    def rerank(self, query: str, docs: List[Document]) -> List[Tuple[Document, float]]:
        pairs = [[query, doc.content] for doc in docs]
        scores = self.model.compute_score(pairs, batch_size=32)
        if not isinstance(scores, list):
            scores = [scores]
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return ranked[: self.cfg.top_k]

# ─────────────────────────────────────────────
# 统一 Reranker 接口
# ─────────────────────────────────────────────

class Reranker:
    """
    统一入口，根据 config.reranker.api_provider 自动路由：
    "bge"    → BGE Reranker API（硅基流动）—— 默认
    "cohere" → Cohere Rerank API
    "local"  → 本地 BGE 模型
    """
    def __init__(self, config: RAGConfig = DEFAULT_CONFIG):
        self.cfg = config.reranker
        provider = self.cfg.api_provider if self.cfg.mode == "api" else "local"
        if provider == "bge":
            console.print(f"[cyan]Reranker: BGE API ({self.cfg.bge_model}) via 硅基流动[/cyan]")
            self._backend = BGEReranker(config)
        elif provider == "cohere":
            console.print(f"[cyan]Reranker: Cohere API ({self.cfg.cohere_model})[/cyan]")
            self._backend = CohereReranker(config)
        else:
            self._backend = LocalReranker(config)

    def rerank(self, query: str, docs: List[Document]) -> List[Tuple[Document, float]]:
        if not docs:
            return []
        ranked = self._backend.rerank(query, docs)
        console.print(f"\n[bold cyan]📊 Rerank[/bold cyan]: {len(docs)} → Top {len(ranked)}")
        for rank, (doc, score) in enumerate(ranked):
            console.print(
                f"  [{rank+1}] score={score:.4f} | {doc.source.split('/')[-1]} | "
                f"{doc.content[:60].replace(chr(10), ' ')}..."
            )
        return ranked

    def get_top_docs(self, query: str, docs: List[Document]) -> List[Document]:
        return [doc for doc, _ in self.rerank(query, docs)]