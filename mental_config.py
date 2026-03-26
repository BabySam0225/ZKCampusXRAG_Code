import os

from dataclasses import dataclass, field
from dotenv import load_dotenv

from rag_config import (
    ChunkConfig,
    EmbeddingConfig,
    BM25Config,
    SearchConfig,
    RerankerConfig,
    QueryRewriterConfig,
    CompressionConfig,
    GeneratorConfig
)

load_dotenv()


@dataclass
class MentalRAGConfig:
    chunk: ChunkConfig = field(default_factory=lambda: ChunkConfig(
        chunk_size=250,  # ⼼理类⽂本段落较短， 适当缩⼩
        chunk_overlap=80,
    ))
    embedding: EmbeddingConfig = field(default_factory=lambda: EmbeddingConfig(
        api_provider="bge",
        index_path="./mental_storage/faiss.index",
        meta_path="./mental_storage/faiss_meta.json",
    ))
    bm25: BM25Config = field(default_factory=lambda: BM25Config(
        index_path="./mental_storage/bm25.pkl",
    ))
    search: SearchConfig = field(default_factory=lambda: SearchConfig(
        top_k_vector=15,
        top_k_bm25=15,
        top_k_merged=15,
    ))
    reranker: RerankerConfig = field(default_factory=lambda: RerankerConfig(
        api_provider="bge",
        top_k=4,  # ⼼理回答不需要太多来源，  精⽽准
    ))
    query_rewriter: QueryRewriterConfig = field(default_factory=QueryRewriterConfig)
    compression: CompressionConfig = field(default_factory=lambda: CompressionConfig(
        mode="rule",
        max_tokens_per_chunk=350,
    ))
    generator: GeneratorConfig = field(default_factory=lambda: GeneratorConfig(
        api_provider="deepseek",
        deepseek_model="deepseek-chat",
        max_tokens=1200,  # ⼼理回答需要更完整的表达
        temperature=0.5,  # 适当提⾼创造性，  让回答不那么机械
    ))
    # API Keys
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHRoPIC_API_KEY"))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    cohere_api_key: str = field(default_factory=lambda: os.getenv("COHERE_API_KEY"))
    deepseek_api_key: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY"))
    siliconflow_api_key: str = field(default_factory=lambda: os.getenv("SILICONFLOW_API_KEY"))
    storage_dir: str = "./mental_storage"


MENTAL_CoNFIG = MentalRAGConfig()