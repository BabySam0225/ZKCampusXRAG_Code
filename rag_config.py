"""
rag_config.py - 全局配置中心
注意: 此文件只能有 dataclass 配置类, 不能有任何业务逻辑代码
"""
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

# 文档分块配置
@dataclass
class ChunkConfig:
    chunk_size: int = 300
    chunk_overlap: int = 100
    separators: list = field(default_factory=lambda: ["\n\n", "\n", "｡", "!", "?"])

# Embedding 配置
@dataclass
class EmbeddingConfig:
    mode: str = "api"
    api_provider: str = "bge"
    bge_model: str = "BAAI/bge-m3"
    bge_base_url: str = "https://api.siliconflow.cn/v1"
    bge_dimensions: int = 1024
    openai_model: str = "text-embedding-3-large"
    openai_dimensions: int = 1536
    index_path: str = "./storage/faiss.index"
    meta_path: str = "./storage/faiss_meta.json"
    batch_size: int = 32
    local_model: str = "BAAI/bge-large-zh-v1.5"
    local_device: str = "cpu"

# BM25 配置
@dataclass
class BM25Config:
    index_path: str = "./storage/bm25.pkl"
    language: str = "zh"

# 混合检索配置
@dataclass
class SearchConfig:
    top_k_vector: int = 20
    top_k_bm25: int = 20
    top_k_merged: int = 20

# Reranker 配置
@dataclass
class RerankerConfig:
    mode: str = "api"
    api_provider: str = "bge"
    bge_model: str = "BAAI/bge-reranker-v2-m3"
    bge_base_url: str = "https://api.siliconflow.cn/v1/rerank"
    cohere_model: str = "rerank-multilingual-v3.0"
    local_model: str = "BAAI/bge-reranker-large"
    local_device: str = "cpu"
    top_k: int = 5

# Query Rewriter 配置
@dataclass
class QueryRewriterConfig:
    mode: str = "api"
    api_provider: str = "deepseek"
    deepseek_model: str = "deepseek-chat"
    deepseek_base_url: str = "https://api.deepseek.com"
    anthropic_model: str = "claude-haiku-4-5"
    openai_model: str = "gpt-4o-mini"
    num_sub_queries: int = 3
    local_model: str = "Qwen/Qwen3-7B-Instruct"
    local_device: str = "cuda"

# Context Compression 配置
@dataclass
class CompressionConfig:
    mode: str = "rule"
    max_tokens_per_chunk: int = 400
    local_model: str = "Qwen/Qwen3-7B-Instruct"

# LLM 生成配置
@dataclass
class GeneratorConfig:
    mode: str = "api"
    api_provider: str = "deepseek"
    deepseek_model: str = "deepseek-chat"
    deepseek_base_url: str = "https://api.deepseek.com"
    anthropic_model: str = "claude-sonnet-4-5"
    openai_model: str = "gpt-4o"
    max_tokens: int = 1024
    temperature: float = 0.1
    local_model: str = "Qwen/Qwen3-30B-A3B"
    local_device: str = "cuda"
    load_in_4bit: bool = True

# 全局配置汇总
@dataclass
class RAGConfig:
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    bm25: BM25Config = field(default_factory=BM25Config)
    search: SearchConfig = field(default_factory=SearchConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    query_rewriter: QueryRewriterConfig = field(default_factory=QueryRewriterConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    cohere_api_key: str = field(default_factory=lambda: os.getenv("COHERE_API_KEY", ""))
    deepseek_api_key: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    siliconflow_api_key: str = field(default_factory=lambda: os.getenv("SILICONFLOW_API_KEY", ""))
    storage_dir: str = "./storage"

    def validate(self):
        errors = []
        if self.embedding.api_provider == "bge" and not self.siliconflow_api_key:
            errors.append("缺少 SILICONFLOW_API_KEY(BGE Embedding 需要)")
        if self.embedding.api_provider == "openai" and not self.openai_api_key:
            errors.append("缺少 OPENAI_API_KEY(OpenAI Embedding 需要)")
        if self.reranker.api_provider == "bge" and not self.siliconflow_api_key:
            errors.append("缺少 SILICONFLOW_API_KEY(BGE Reranker 需要)")
        if self.reranker.api_provider == "cohere" and not self.cohere_api_key:
            errors.append("缺少 COHERE_API_KEY(Cohere Reranker 需要)")
        if self.query_rewriter.api_provider == "deepseek" and not self.deepseek_api_key:
            errors.append("缺少 DEEPSEEK_API_KEY(Query Rewrite 需要)")
        if self.query_rewriter.api_provider == "anthropic" and not self.anthropic_api_key:
            errors.append("缺少 ANTHROPIC_API_KEY(Query Rewrite 需要)")
        if self.generator.api_provider == "deepseek" and not self.deepseek_api_key:
            errors.append("缺少 DEEPSEEK_API_KEY(LLM 生成需要)")
        if self.generator.api_provider == "anthropic" and not self.anthropic_api_key:
            errors.append("缺少 ANTHROPIC_API_KEY(LLM 生成需要)")
        
        if errors:
            print("\n".join(errors))
            print("\n请在 .env 文件中配置上述 Key, 参考 .env.example")
            raise SystemExit(1)

# 必须在所有 class 定义之后
DEFAULT_CONFIG = RAGConfig()