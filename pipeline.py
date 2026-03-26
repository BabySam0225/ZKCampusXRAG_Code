"""
pipeline.py — RAG 主流⽔线
将所有模块串联起来，  对外提供两个核⼼⽅法：
- build(docs)建⽴索引
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
    完整 RAG 流⽔线
    ┌──────────────────────────────────────────┐
    │ ⽤户问题│
    │↓ Query Rewrite│
    │ 改写查询（ main + sub_queries）│
    │↓ Hybrid Search│
    │ 向量检索 + BM25 → RRF 合并 Top 20│
    │↓ Rerank│
    │ BGE Cross-Encoder 精排 → Top 5│
    │↓ Context Compression│
    │ 规则/模型压缩 → 减少噪声│
    │↓ LLM ⽣成│
    │ 基于上下⽂⽣成最终答案│
    └──────────────────────────────────────────┘
    """
    def __init__(self, config: RAGConfig = DEFAULT_CONFIG):
        self.config = config
        os.makedirs(config.storage_dir, exist_ok=True)

        # 初始化各模块（ 懒加载： 模型在⾸次使⽤时才加载）
        self.processor = DocumentProcessor(config)
        self.retriever = HybridRetriever(config)
        self.query_rewriter = QueryRewriter(config)
        self.reranker = Reranker(config)
        self.compressor = ContextCompressor(config)
        self.generator = Generator(config)

        self._docs: List[Document] = []
        self._index_built = False

    # ─────────────────────────────────────────
    # 构建索引
    # ─────────────────────────────────────────
    def build(self, docs: List[Document]):
        """从已处理的 Document 列表建⽴全部索引"""
        console.print(Panel("[bold green]![img](tmp/409949714946_docxword_media_image1.png) 开始构建 RAG 索引[/bold green]"))
        t0 = time.time()
        self._docs = docs
        self.retriever.build(docs)
        self.retriever.save()
        console.print(f"[green]✓ 索引构建完成， 耗时 {time.time()-t0:.1f}s[/green]")
        self._index_built = True

    def build_from_directory(self, dir_path: str):
        """从⽬录加载⽂档并建⽴索引"""
        docs = self.processor.process_directory(dir_path)
        self.build(docs)
        # 同时保存 docs 便于下次直接加载
        docs_path = os.path.join(self.config.storage_dir, "docs.json")
        self.processor.save_docs(docs, docs_path)

    def build_from_file(self, file_path: str):
        """从单个⽂件建⽴索引"""
        docs = self.processor.process_file(file_path)
        self.build(docs)

    def build_from_text(self, text: str, source: str = "inline"):
        """从纯⽂本建⽴索引（ 快速测试⽤） """
        docs = self.processor.process_text(text, source=source)
        self.build(docs)

    def load_index(self):
        """加载已有索引（ 跳过重建） """
        docs_path = os.path.join(self.config.storage_dir, "docs.json")
        if not os.path.exists(docs_path):
            raise FileNotFoundError(f"未找到⽂档缓存: {docs_path}， 请先调⽤ build()")
        self._docs = self.processor.load_docs(docs_path)
        self.retriever.load(self._docs)
        self._index_built = True
        console.print("[green]✓ 索引加载完成[/green]")

    # ─────────────────────────────────────────
    # 核⼼问答流程
    # ─────────────────────────────────────────
    def query(self, question: str, history: list = None) -> RAGAnswer:
        """端到端 RAG 问答， history 为多轮对话历史"""
        if not self._index_built:
            raise RuntimeError("请先调⽤ build() 或 load_index() 建⽴索引")

        console.print(Panel(f"[bold white]![img](?) ⽤户问题[/bold white]\n{question}"))
        t0 = time.time()

        # ── Step 1: Query Rewrite ──────────────
        rewrite_result = self.query_rewriter.rewrite(question)
        all_queries = rewrite_result.all_queries()

        # ── Step 2: Hybrid Search ──────────────
        console.print(f"\n[bold cyan]![img](tmp/409949714946_docxword_media_image3.png) Hybrid Search[/bold cyan]: {len(all_queries)} 个查询")
        candidates = self.retriever.multi_query_search(all_queries)
        console.print(f" → 检索到 [cyan]{len(candidates)}[/cyan] 个候选块")

        # ── Step 3: Rerank ─────────────────────
        ranked_docs = self.reranker.get_top_docs(rewrite_result.main_query, candidates)

        # ── Step 4: Context Compression ────────
        compressed = self.compressor.compress(rewrite_result.main_query, ranked_docs)
        context = self.compressor.format_context(compressed)

        # ── Step 5: LLM Generation（ 带历史记忆） ──
        answer = self.generator.generate(question, context, compressed, history=history)

        elapsed = time.time() - t0
        console.print(f"\n[bold green]✓ 完成， 总耗时 {elapsed:.1f}s[/bold green]")
        self._print_answer(answer)
        return answer

    # ─────────────────────────────────────────
    # 美化输出
    # ─────────────────────────────────────────
    @staticmethod
    def _print_answer(answer: RAGAnswer):
        # 答案⾯板
        console.print(Panel(
            f"[bold white]{answer.answer}[/bold white]",
            title="[green]![img](tmp/409949714946_docxword_media_image4.png) RAG 回答[/green]",
            border_style="green",
        ))
        # 来源 + 图⽚表格
        table = Table(title="![img](tmp/409949714946_docxword_media_image5.png) 引⽤来源", show_lines=True)
        table.add_column("编号", style="cyan", width=6)
        table.add_column("来源⽂件", style="yellow")
        table.add_column("内容摘要", style="white")
        table.add_column("关联图⽚", style="magenta")

        for chunk in answer.sources:
            preview = chunk["content"][:80].replace("\n", " ") + "..."
            images = chunk.get("images", [])
            img_names = "\n".join(os.path.basename(p) for p in images) if images else "-"
            table.add_row(str(chunk["index"]), os.path.basename(chunk["source"]), preview, img_names)
        console.print(table)

        # 如果有关联图⽚， 列出完整路径⽅便打开
        all_images = []
        for chunk in answer.sources:
            all_images.extend(chunk.get("images", []))
        if all_images:
            console.print("\n[bold yellow]![img](tmp/409949714946_docxword_media_image6.png) 关联图⽚路径（ 可直接打开查看） ： [/bold yellow]")
            for p in all_images:
                console.print(f" {p}")

    def chat(self):
        """交互式多轮问答， ⾃动维护对话历史"""
        console.print(Panel(
            "[bold green]![img](tmp/409949714946_docxword_media_image7.png) RAG 系统已就绪[/bold green]\n"
            "输⼊问题开始问答， 输⼊ [yellow]clear[/yellow] 清除历史， 输⼊ [yellow]exit/quit/q[/yellow] 退出"
        ))
        history = []  # [{"role": "user/assistant", "content": "..."}, ...]
        while True:
            try:
                question = input("\n> ").strip()
                if question.lower() in {"exit", "quit", "q"}:
                    console.print("[yellow]再⻅！ [/yellow]")
                    break
                if question.lower() == "clear":
                    history.clear()
                    console.print("[cyan]✓ 对话历史已清除[/cyan]")
                    continue
                if not question:
                    continue
                # 传⼊历史， 获取答案
                answer = self.query(question, history=history)
                # 把这轮问答存⼊历史（ 只存纯⽂字， 不含上下⽂， 避免 token 暴涨）
                history.append({"role": "user", "content": question})
                history.append({"role": "assistant", "content": answer.answer})
                round_num = len(history) // 2
                console.print(f"[dim]（ 已记忆 {round_num} 轮对话， 输⼊ clear 可清除） [/dim]")
            except KeyboardInterrupt:
                console.print("\n[yellow]已中断[/yellow]")
                break
            except Exception as e:
                console.print(f"[red]错误: {e}[/red]")