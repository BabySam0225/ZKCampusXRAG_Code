"""
mental_pipeline.py — 心理辅导 RAG 流水线
在标准 RAG 流程基础上增加：
  1. 情绪识别（每次提问前先判断用户状态）
  2. 危机干预（检测到极端情绪直接走安全回复）
  3. 温暖化生成（使用专属 Prompt，temperature 更高）
  4. 多轮记忆（同 main RAG）
"""

import os
import json
import time
from typing import List, Optional
from dataclasses import dataclass

from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from mental_config import MENTAL_CONFIG, MentalRAGConfig
from mental_prompt import (
    MENTAL_SYSTEM_PROMPT, MENTAL_ANSWER_PROMPT,
    EMOTION_DETECT_PROMPT, CRISIS_RESPONSE
)
from document_processor import DocumentProcessor, Document
from hybrid_search import HybridRetriever
from reranker import Reranker
from context_compressor import ContextCompressor
from query_rewriter import QueryRewriter

console = Console()


# ─────────────────────────────────────────────
# 情绪识别模块
# ─────────────────────────────────────────────
class EmotionDetector:
    """
    用 DeepSeek 快速判断用户情绪状态
    识别结果影响后续处理：危机状态 → 直接干预，普通状态 → 正常 RAG
    """

    def __init__(self, config: MentalRAGConfig):
        import httpx
        self.client = OpenAI(
            api_key=config.deepseek_api_key,
            base_url=config.generator.deepseek_base_url,
            http_client=httpx.Client(verify=False),
        )
        self.model = config.generator.deepseek_model

    def detect(self, text: str) -> dict:
        """返回 {"emotion": ..., "severity": ..., "crisis": bool, "keywords": [...]}"""
        try:
            prompt = EMOTION_DETECT_PROMPT.format(text=text)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content)
            return result
        except Exception:
            # 识别失败时返回默认值，不影响主流程
            return {"emotion": "未知", "severity": "轻微", "crisis": False, "keywords": []}


# ─────────────────────────────────────────────
# 心理辅导专属生成器
# ─────────────────────────────────────────────
class MentalGenerator:
    """使用专属 Prompt 和更高 temperature 生成有温度的回复"""

    def __init__(self, config: MentalRAGConfig):
        import httpx
        self.client = OpenAI(
            api_key=config.deepseek_api_key,
            base_url=config.generator.deepseek_base_url,
            http_client=httpx.Client(verify=False),
        )
        self.cfg = config.generator
        MAX_HISTORY = 8  # 心理对话保留更多历史（4轮）

    def generate(self, question: str, context: str, history: List[dict],
                 stream: bool = True, enable_thinking: bool = False) -> str:
        MAX_HISTORY = 8
        messages = [{"role": "system", "content": MENTAL_SYSTEM_PROMPT}]
        messages.extend(history[-MAX_HISTORY:])

        if context.strip():
            prompt = MENTAL_ANSWER_PROMPT.format(context=context, question=question)
        else:
            prompt = f"用户说：{question}\n\n请用温暖的方式回应，以情感支持为主。"

        messages.append({"role": "user", "content": prompt})

        if stream:
            response = self.client.chat.completions.create(
                model=self.cfg.deepseek_reasoner_model if enable_thinking else self.cfg.deepseek_model,
                messages=messages,
                max_tokens=self.cfg.max_tokens,
                temperature=self.cfg.temperature,
                stream=True,
                extra_body={"enable_thinking": enable_thinking},
            )
            full = ""
            in_thinking = False
            print("\n💙 暖心：", end="", flush=True)
            for chunk in response:
                delta = chunk.choices[0].delta
                thinking = getattr(delta, "reasoning_content", None) or ""
                content_text = delta.content or ""
                if thinking and enable_thinking:
                    if not in_thinking:
                        print("\n\033[2m[深度思考] ", end="", flush=True)
                        in_thinking = True
                    print(thinking, end="", flush=True)
                if content_text:
                    if in_thinking:
                        print("\033[0m\n💙 暖心：", end="", flush=True)
                        in_thinking = False
                    print(content_text, end="", flush=True)
                    full += content_text
            print()
            return full
        else:
            response = self.client.chat.completions.create(
                model=self.cfg.deepseek_reasoner_model if enable_thinking else self.cfg.deepseek_model,
                messages=messages,
                max_tokens=self.cfg.max_tokens,
                temperature=self.cfg.temperature,
                extra_body={"enable_thinking": enable_thinking},
            )
            return response.choices[0].message.content


# ─────────────────────────────────────────────
# 心理辅导主流水线
# ─────────────────────────────────────────────
class MentalPipeline:

    def __init__(self, config: MentalRAGConfig = MENTAL_CONFIG):
        self.config = config
        self.processor = DocumentProcessor(config)
        self.retriever = HybridRetriever(config)
        self.reranker = Reranker(config)
        self.compressor = ContextCompressor(config)
        self.query_rewriter = QueryRewriter(config)
        self.emotion_detector = EmotionDetector(config)
        self.generator = MentalGenerator(config)
        self._index_built = False

    def load_index(self):
        docs_path = os.path.join(self.config.storage_dir, "docs.json")
        if not os.path.exists(docs_path):
            console.print(Panel(
                "[bold red]❌ 心理知识库不存在！[/bold red]\n\n"
                "请先运行：\n"
                "  [bold yellow]python build_mental_kb.py build --dir ./mental_docs[/bold yellow]",
                border_style="red"
            ))
            raise FileNotFoundError("心理知识库未建立")
        docs = self.processor.load_docs(docs_path)
        self.retriever.load(docs)
        self._index_built = True
        console.print("[green]✓ 心理知识库加载完成[/green]")

    def query(self, question: str, history: List[dict] = None,
              enable_thinking: bool = False) -> str:
        """
        完整流程：
          1. 情绪识别
          2. 危机检测 → 直接干预
          3. Query Rewrite + Hybrid Search + Rerank
          4. 有温度的生成
        """
        history = history or []

        # ── Step 1: 情绪识别 ──────────────────
        emotion = self.emotion_detector.detect(question)
        emotion_label = emotion.get("emotion", "未知")
        severity = emotion.get("severity", "轻微")
        is_crisis = emotion.get("crisis", False)

        console.print(
            f"\n[dim]情绪识别：{emotion_label}（{severity}）"
            + ("  [bold red]⚠ 危机信号[/bold red]" if is_crisis else "") + "[/dim]"
        )

        # ── Step 2: 危机干预 ──────────────────
        if is_crisis:
            console.print(Panel(
                CRISIS_RESPONSE,
                title="[bold red]💙 暖心[/bold red]",
                border_style="red",
            ))
            return CRISIS_RESPONSE

        # ── Step 3: 检索相关知识 ──────────────
        context = ""
        try:
            rewrite_result = self.query_rewriter.rewrite(question, history=history or [])
            all_queries = rewrite_result.all_queries()
            candidates = self.retriever.multi_query_search(all_queries)
            if candidates:
                ranked_docs = self.reranker.get_top_docs(rewrite_result.main_query, candidates)
                compressed = self.compressor.compress(rewrite_result.main_query, ranked_docs)
                context = self.compressor.format_context(compressed)
        except Exception as e:
            console.print(f"[dim]检索时遇到问题，将纯用情感支持回复: {e}[/dim]")

        # ── Step 4: 生成有温度的回复 ──────────
        answer = self.generator.generate(question, context, history)

        # 流式模式已在生成时实时打印，非流式时才用 Panel
        if not getattr(self.generator, '_last_streamed', True):
            console.print(Panel(answer, title="[bold green]💙 暖心[/bold green]", border_style="green"))
        return answer

    # ─────────────────────────────────────────
    # 对话主循环
    # ─────────────────────────────────────────
    def chat(self):
        console.print(Panel(
            "[bold green]💙 暖心心理辅导助手已就绪[/bold green]\n\n"
            "你好，我是暖心，很高兴认识你 😊\n"
            "不管你现在是什么心情，都可以跟我说说。\n\n"
            "输入 [yellow]clear[/yellow] 开始新话题，输入 [yellow]exit[/yellow] 退出",
            border_style="green",
        ))

        history = []
        while True:
            try:
                user_input = input("\n你：").strip()
                if not user_input:
                    continue
                if user_input.lower() in {"exit", "quit", "q"}:
                    console.print("\n[green]暖心：保重，有需要随时来找我 💙[/green]")
                    break
                if user_input.lower() == "clear":
                    history.clear()
                    console.print("[cyan]✓ 已开始新话题[/cyan]")
                    continue

                answer = self.query(user_input, history=history)
                # 存入历史
                history.append({"role": "user", "content": user_input})
                history.append({"role": "assistant", "content": answer})
                round_num = len(history) // 2
                console.print(f"[dim]（已记忆 {round_num} 轮对话）[/dim]")

            except KeyboardInterrupt:
                console.print("\n[green]暖心：保重，有需要随时来找我 💙[/green]")
                break
            except Exception as e:
                console.print(f"[red]出错了: {e}[/red]")
