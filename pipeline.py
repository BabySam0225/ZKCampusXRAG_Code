"""
pipeline.py — RAG 主流水线
将所有模块串联起来，对外提供两个核心方法：
  - build(docs)     建立索引
  - query(question) 端到端问答
"""

import os
import time
from typing import List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rag_config import RAGConfig, DEFAULT_CONFIG
from document_processor import Document, DocumentProcessor
from hybrid_search import HybridRetriever
from query_rewriter import QueryRewriter
from reranker import Reranker
from context_compressor import ContextCompressor
from generator import Generator, RAGAnswer

console = Console()


class RAGPipeline:
    """
    完整 RAG 流水线
    ┌──────────────────────────────────────────┐
    │  用户问题                                │
    │     ↓ Query Rewrite                      │
    │  改写查询（main + sub_queries）          │
    │     ↓ Hybrid Search                      │
    │  向量检索 + BM25 → RRF 合并 Top 20      │
    │     ↓ Rerank                             │
    │  BGE Cross-Encoder 精排 → Top 5         │
    │     ↓ Context Compression                │
    │  规则/模型压缩 → 减少噪声               │
    │     ↓ LLM 生成                           │
    │  基于上下文生成最终答案                 │
    └──────────────────────────────────────────┘
    """

    def __init__(self, config: RAGConfig = DEFAULT_CONFIG):
        self.config = config
        os.makedirs(config.storage_dir, exist_ok=True)

        # 初始化各模块（懒加载：模型在首次使用时才加载）
        self.processor = DocumentProcessor(config)
        self.retriever = HybridRetriever(config)
        self.query_rewriter = QueryRewriter(config)
        self.reranker = Reranker(config)
        self.compressor = ContextCompressor(config)
        self.generator = Generator(config)

        self._docs: List[Document] = []
        self._index_built = False
        self._graph_retriever = None  # 图谱检索器（可选）

    # ─────────────────────────────────────────
    # 构建索引
    # ─────────────────────────────────────────
    def build(self, docs: List[Document]):
        """从已处理的 Document 列表建立全部索引"""
        console.print(Panel("[bold green]🔨 开始构建 RAG 索引[/bold green]"))
        t0 = time.time()
        self._docs = docs
        self.retriever.build(docs)
        self.retriever.save()
        console.print(f"[green]✓ 索引构建完成，耗时 {time.time()-t0:.1f}s[/green]")
        self._index_built = True

    def build_from_directory(self, dir_path: str):
        """从目录加载文档并建立索引"""
        docs = self.processor.process_directory(dir_path)
        self.build(docs)
        # 同时保存 docs 便于下次直接加载
        docs_path = os.path.join(self.config.storage_dir, "docs.json")
        self.processor.save_docs(docs, docs_path)

    def build_from_file(self, file_path: str):
        """从单个文件建立索引"""
        docs = self.processor.process_file(file_path)
        self.build(docs)

    def build_from_text(self, text: str, source: str = "inline"):
        """从纯文本建立索引（快速测试用）"""
        docs = self.processor.process_text(text, source=source)
        self.build(docs)

    def load_index(self):
        """加载已有索引（跳过重建）"""
        docs_path = os.path.join(self.config.storage_dir, "docs.json")
        if not os.path.exists(docs_path):
            raise FileNotFoundError(f"未找到文档缓存: {docs_path}，请先调用 build()")
        self._docs = self.processor.load_docs(docs_path)
        self.retriever.load(self._docs)
        self._index_built = True
        console.print("[green]✓ 索引加载完成[/green]")

        # 尝试加载图谱（可选，图谱不存在则跳过）
        graph_dir = os.path.join(self.config.storage_dir, "graph")
        if os.path.exists(os.path.join(graph_dir, "graph.json")):
            try:
                from graph_store import KnowledgeGraph
                from graph_retriever import GraphRetriever
                kg = KnowledgeGraph(storage_dir=graph_dir)
                kg.load()
                self._graph_retriever = GraphRetriever(
                    graph=kg, doc_map=self.retriever._doc_map,
                )
                console.print("[green]✓ 知识图谱已加载（GraphRAG 增强已启用）[/green]")
            except Exception as e:
                console.print(f"[dim]图谱加载失败，使用标准检索：{e}[/dim]")
        else:
            console.print("[dim]未找到图谱，使用标准检索（可运行 build_graph.py 构建图谱）[/dim]")

    # ─────────────────────────────────────────
    # 核心问答流程
    # ─────────────────────────────────────────
    def query(self, question: str, history: list = None,
              enable_thinking: bool = False) -> RAGAnswer:
        """
        端到端 RAG 问答
        enable_thinking: 是否开启 Qwen3 深度思考模式（仅 qwen3 provider 有效）
                         前端可通过按钮控制此参数
        """
        if not self._index_built:
            raise RuntimeError("请先调用 build() 或 load_index() 建立索引")

        console.print(Panel(f"[bold white]❓ 用户问题[/bold white]\n{question}"))
        t0 = time.time()

        # ── Step 1: Query Rewrite（传入历史，让模糊问题得到补全）──
        rewrite_result = self.query_rewriter.rewrite(question, history=history or [])
        all_queries = rewrite_result.all_queries()

        # ── Step 2: Hybrid Search ──────────────
        console.print(f"\n[bold cyan]🔍 Hybrid Search[/bold cyan]: {len(all_queries)} 个查询")
        candidates = self.retriever.multi_query_search(all_queries)
        console.print(f"  → 检索到 [cyan]{len(candidates)}[/cyan] 个候选块")

        # ── Step 2.5: Graph RAG 增强（可选）────
        if self._graph_retriever is not None:
            try:
                from graph_retriever import rrf_merge_with_graph
                graph_docs = self._graph_retriever.retrieve(question)
                if graph_docs:
                    candidates = rrf_merge_with_graph(candidates, graph_docs)
                    console.print(
                        f"  → 图谱融合后 [cyan]{len(candidates)}[/cyan] 个候选块"
                    )
            except Exception as e:
                console.print(f"[dim]图谱检索异常，跳过：{e}[/dim]")

        # ── Step 3: Rerank ─────────────────────
        ranked_docs = self.reranker.get_top_docs(rewrite_result.main_query, candidates)

        # ── Step 4: Context Compression ────────
        compressed = self.compressor.compress(rewrite_result.main_query, ranked_docs)
        context = self.compressor.format_context(compressed)

        # ── Step 5: LLM Generation（带历史记忆）──
        answer = self.generator.generate(
            question, context, compressed,
            history=history or [],
            enable_thinking=enable_thinking,
        )

        elapsed = time.time() - t0
        console.print(f"\n[bold green]✓ 完成，总耗时 {elapsed:.1f}s[/bold green]")
        self._print_answer(answer)
        return answer


    # ─────────────────────────────────────────
    # 流式问答（生成器版，供后端 SSE 使用）
    # ─────────────────────────────────────────
    def stream_query(self, question: str, history: list = None,
                     enable_thinking: bool = False):
        """
        生成器函数，逐 chunk yield 内容，供后端 SSE 边生成边推送。
        yield 格式：{"type": "thinking"|"answer"|"done"|"error", "content": str}
          - thinking: R1 思考过程（enable_thinking=True 时才有）
          - answer:   正式回答内容（逐字）
          - done:     流结束信号，携带 sources 和耗时
          - error:    出错信息
        """
        if not self._index_built:
            yield {"type": "error", "content": "索引未加载"}
            return

        import time
        t0 = time.time()

        try:
            # 检索阶段（非流式，快速完成）
            rewrite_result = self.query_rewriter.rewrite(question)
            candidates = self.retriever.multi_query_search(rewrite_result.all_queries())
            ranked_docs = self.reranker.get_top_docs(rewrite_result.main_query, candidates)
            compressed = self.compressor.compress(rewrite_result.main_query, ranked_docs)
            context = self.compressor.format_context(compressed)
            sources = [{"source": c.get("source",""), "content": c.get("content","")[:100]}
                       for c in compressed]
        except Exception as e:
            yield {"type": "error", "content": f"检索失败: {e}"}
            return

        # LLM 流式生成
        try:
            model = (self.generator.cfg.deepseek_reasoner_model
                     if enable_thinking else self.generator.cfg.deepseek_model)
            messages = self.generator._backend._build_messages_or_direct(
                question, context, history or []
            ) if hasattr(self.generator._backend, '_build_messages_or_direct') else None

            # 直接调用底层 client 流式 API
            from generator import _build_messages
            messages = _build_messages(question, context, history or [])
            client = self.generator._backend.client

            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=self.generator.cfg.max_tokens,
                temperature=self.generator.cfg.temperature,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                thinking = getattr(delta, "reasoning_content", None) or ""
                content_text = delta.content or ""
                if thinking:
                    yield {"type": "thinking", "content": thinking}
                if content_text:
                    yield {"type": "answer", "content": content_text}

        except Exception as e:
            yield {"type": "error", "content": f"生成失败: {e}"}
            return

        elapsed = time.time() - t0
        yield {"type": "done", "content": "", "sources": sources, "latency": round(elapsed, 2)}

    # ─────────────────────────────────────────
    # 美化输出
    # ─────────────────────────────────────────
    @staticmethod
    def _print_answer(answer: RAGAnswer):
        # 答案面板
        console.print(Panel(
            f"[bold white]{answer.answer}[/bold white]",
            title="[green]💡 RAG 回答[/green]",
            border_style="green",
        ))
        # 来源 + 图片表格
        table = Table(title="📚 引用来源", show_lines=True)
        table.add_column("编号", style="cyan", width=6)
        table.add_column("来源文件", style="yellow")
        table.add_column("内容摘要", style="white")
        table.add_column("关联图片", style="magenta")
        for chunk in answer.sources:
            preview = chunk["content"][:80].replace("\n", " ") + "..."
            images = chunk.get("images", [])
            img_names = "\n".join(os.path.basename(p) for p in images) if images else "-"
            table.add_row(str(chunk["index"]), os.path.basename(chunk["source"]), preview, img_names)
        console.print(table)

        # 如果有关联图片，列出完整路径方便打开
        all_images = []
        for chunk in answer.sources:
            all_images.extend(chunk.get("images", []))
        if all_images:
            console.print("\n[bold yellow]🖼 关联图片路径（可直接打开查看）：[/bold yellow]")
            for p in all_images:
                console.print(f"  {p}")

    def chat(self):
        """交互式多轮问答，自动维护对话历史"""
        console.print(Panel(
            "[bold green]🚀 RAG 系统已就绪[/bold green]\n"
            "输入问题开始问答，输入 [yellow]clear[/yellow] 清除历史，输入 [yellow]exit[/yellow] 退出"
        ))
        history = []   # [{"role": "user/assistant", "content": "..."}, ...]
        while True:
            try:
                question = input("\n> ").strip()
                if question.lower() in {"exit", "quit", "q"}:
                    console.print("[yellow]再见！[/yellow]")
                    break
                if question.lower() == "clear":
                    history.clear()
                    console.print("[cyan]✓ 对话历史已清除[/cyan]")
                    continue
                if not question:
                    continue
                # 传入历史，获取答案
                # enable_thinking 由外部传入（默认 False），chat 模式固定为 False
                answer = self.query(question, history=history, enable_thinking=False)
                # 把这轮问答存入历史（只存纯文字，不含上下文，避免 token 暴涨）
                history.append({"role": "user", "content": question})
                history.append({"role": "assistant", "content": answer.answer})
                round_num = len(history) // 2
                console.print(f"[dim]（已记忆 {round_num} 轮对话，输入 clear 可清除）[/dim]")
            except KeyboardInterrupt:
                console.print("\n[yellow]已中断[/yellow]")
                break
            except Exception as e:
                console.print(f"[red]错误: {e}[/red]")
