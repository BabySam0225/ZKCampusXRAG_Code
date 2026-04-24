"""
graph_retriever.py — 图谱增强检索
=====================================
负责：
  1. 从用户问题中提取关键实体
  2. 用图谱找到相关 chunk_id
  3. 与 hybrid_search 结果合并（RRF 融合）
"""

import os
import re
import httpx
import json
from typing import List, Dict, Optional
from openai import OpenAI
from rich.console import Console
from dotenv import load_dotenv

from graph_store import KnowledgeGraph
from document_processor import Document

load_dotenv()
console = Console()

ENTITY_EXTRACT_PROMPT = """从以下问题中提取关键实体，用于知识图谱检索。

【问题】
{question}

提取规则：
1. 只提取名词性实体（人名、机构、政策名、专业名、岗位、地区等）
2. 不要提取动词、形容词
3. 最多提取 5 个最关键的实体

只返回 JSON：{{"entities": ["实体1", "实体2", ...]}}"""


class GraphRetriever:
    def __init__(self, graph: KnowledgeGraph,
                 doc_map: Dict[str, Document]):
        """
        graph:   已加载的知识图谱
        doc_map: chunk_id → Document 的映射（来自 HybridRetriever._doc_map）
        """
        self.graph = graph
        self.doc_map = doc_map
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            base_url="https://api.deepseek.com",
            http_client=httpx.Client(verify=False),
        )

    def extract_query_entities(self, question: str) -> List[str]:
        """
        从问题中提取实体。
        先用规则快速匹配图谱中已有的实体，不够再调 LLM。
        """
        # 规则匹配：直接在图谱实体名中查找问题包含的词
        rule_matched = self.graph.search_entities_by_keyword(question)
        if len(rule_matched) >= 2:
            return rule_matched[:5]

        # 调 LLM 提取
        try:
            resp = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content":
                           ENTITY_EXTRACT_PROMPT.format(question=question)}],
                max_tokens=150,
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or "{}"
            # 清理可能的 markdown
            raw = re.sub(r"```(?:json)?\s*", "", raw).strip()
            data = json.loads(raw)
            llm_entities = data.get("entities", [])
        except Exception:
            llm_entities = []

        # 合并两种结果，去重
        combined = list(dict.fromkeys(rule_matched + llm_entities))
        return combined[:5]

    def retrieve(self, question: str,
                 hops: int = 2,
                 max_chunks: int = 10) -> List[Document]:
        """
        图谱检索主入口：
          问题 → 实体 → 图谱扩展 → chunk_id → Document
        """
        entities = self.extract_query_entities(question)
        if not entities:
            return []

        console.print(
            f"  [dim]图谱实体：{', '.join(entities[:5])}[/dim]"
        )

        chunk_ids = self.graph.expand_chunks(
            entities, hops=hops, max_chunks=max_chunks
        )

        docs = []
        for cid in chunk_ids:
            doc = self.doc_map.get(cid)
            if doc:
                docs.append(doc)

        console.print(
            f"  [dim]图谱命中 {len(docs)} 个 chunk[/dim]"
        )
        return docs


def rrf_merge_with_graph(
    hybrid_docs: List[Document],
    graph_docs: List[Document],
    k: int = 60,
    graph_weight: float = 0.4,
) -> List[Document]:
    """
    把 hybrid search 和 graph retrieval 的结果用 RRF 合并。
    graph_weight: 图谱结果的权重系数（0~1，越高越偏重图谱）
    k: RRF 平滑参数
    """
    scores: Dict[str, float] = {}

    # Hybrid search 结果（权重 1.0）
    for rank, doc in enumerate(hybrid_docs):
        scores[doc.doc_id] = scores.get(doc.doc_id, 0) + 1.0 / (k + rank + 1)

    # Graph retrieval 结果（权重 graph_weight）
    for rank, doc in enumerate(graph_docs):
        scores[doc.doc_id] = scores.get(doc.doc_id, 0) + \
                              graph_weight / (k + rank + 1)

    # 合并所有 doc，按 RRF 分数排序
    all_docs = {d.doc_id: d for d in hybrid_docs + graph_docs}
    sorted_ids = sorted(scores.keys(), key=lambda x: -scores[x])
    return [all_docs[did] for did in sorted_ids if did in all_docs]
