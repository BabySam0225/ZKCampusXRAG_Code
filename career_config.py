"""
career_config.py  —  就业指导应⽤专属配置
知识库存储在 ./career_storage/
"""
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv
from rag_config import (
    ChunkConfig, EmbeddingConfig, BM25Config, SearchConfig, 
    RerankerConfig, QueryRewriterConfig, CompressionConfig, GeneratorConfig
)

load_dotenv()

# 专业前缀 → 全称映射
MAJOR_MAP = {
    "网络": "计算机网络技术",
    "电子": "电子信息技术",
    "通信": "通信技术",
    "大数据": "大数据技术",
    "计算机": "计算机应用技术",
    "物联网": "物联网应用技术",
    "信管": "信息管理",
}

@dataclass
class CareerRAGConfig:
    chunk: ChunkConfig = field(default_factory=lambda: ChunkConfig(
        chunk_size=400,
        chunk_overlap=80,
    ))
    embedding: EmbeddingConfig = field(default_factory=lambda: EmbeddingConfig(
        api_provider="bge",
        index_path="./career_storage/faiss.index",
        meta_path="./career_storage/faiss_meta.json",
    ))
    bm25: BM25Config = field(default_factory=lambda: BM25Config(
        index_path="./career_storage/bm25.pkl",
    ))
    search: SearchConfig = field(default_factory=lambda: SearchConfig(
        top_k_vector=15,
        top_k_bm25=15,
        top_k_merged=15,
    ))
    reranker: RerankerConfig = field(default_factory=lambda: RerankerConfig(
        api_provider="bge",
        top_k=5,
    ))
    query_rewriter: QueryRewriterConfig = field(default_factory=QueryRewriterConfig)
    compression: CompressionConfig = field(default_factory=lambda: CompressionConfig(
        mode="rule",
        max_tokens_per_chunk=500,
    ))
    generator: GeneratorConfig = field(default_factory=lambda: GeneratorConfig(
        api_provider="deepseek",
        deepseek_model="deepseek-chat",
        max_tokens=1500,
        temperature=0.3,
    ))
    
    # API Keys
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    cohere_api_key: str = field(default_factory=lambda: os.getenv("COHERE_API_KEY"))
    deepseek_api_key: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY"))
    siliconflow_api_key: str = field(default_factory=lambda: os.getenv("SILICONFLOW_API_KEY"))
    
    storage_dir: str = "./career_storage"

CAREER_CONFIG = CareerRAGConfig()