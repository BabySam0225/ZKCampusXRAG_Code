"""
query_rewriter.py — 查询改写层
支持：本地 Qwen3-7B / Anthropic API / OpenAI API
输出：main_query + sub_queries（JSON 格式）
"""

import json
import re
from dataclasses import dataclass
from typing import List

from rich.console import Console

from rag_config import RAGConfig, DEFAULT_CONFIG

console = Console()


REWRITE_PROMPT = """你是一个专业的问题改写助手，负责将用户问题改写为适合知识库检索的独立查询。

【对话历史】（最近几轮，帮助理解用户意图）
{history}

【当前问题】
{question}

改写规则：
1. 如果当前问题是模糊引用（"那个"、"刚才说的"、"这个呢"、"还有呢"），
   必须结合历史还原成完整独立的问题，让它脱离历史也能被检索
2. main_query：改写后的主查询，必须是完整独立的句子，包含所有必要的实体和主题
3. sub_queries：生成 {num} 个表达方式不同的查询（换角度、换词、细化问题）
4. 所有查询都不能依赖对话历史，必须自包含

请严格输出 JSON，不要输出任何其他内容：
{{
  "main_query": "改写后的完整主查询",
  "sub_queries": ["子查询1", "子查询2"]
}}"""


@dataclass
class RewriteResult:
    main_query: str
    sub_queries: List[str]

    def all_queries(self) -> List[str]:
        return [self.main_query] + self.sub_queries


# ─────────────────────────────────────────────
# 解析 LLM 输出为结构化结果
# ─────────────────────────────────────────────
def _parse_rewrite_output(text: str, original: str, num: int) -> RewriteResult:
    """健壮解析 LLM JSON 输出，失败时返回原始查询"""

    # 第一步：清理文本
    # 去掉 markdown 代码块标记（```json ... ``` 或 ``` ... ```）
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    # 去掉开头结尾的换行和空白
    text = text.strip()

    # 第二步：如果文本不以 { 开头，尝试补全（处理模型漏掉开头 { 的情况）
    if text and not text.startswith("{"):
        # 找到第一个 { 的位置
        brace_idx = text.find("{")
        if brace_idx >= 0:
            text = text[brace_idx:]
        else:
            # 没有 {，尝试把内容包装成 JSON
            text = "{" + text

    # 第三步：尝试提取最外层 JSON 对象
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            main_q = data.get("main_query", original).strip()
            sub_qs = data.get("sub_queries", [])[:num]
            while len(sub_qs) < num:
                sub_qs.append(original)
            return RewriteResult(main_query=main_q, sub_queries=sub_qs)
        except json.JSONDecodeError:
            pass

    # 第四步：尝试直接解析整个文本
    try:
        data = json.loads(text)
        main_q = data.get("main_query", original).strip()
        sub_qs = data.get("sub_queries", [])[:num]
        while len(sub_qs) < num:
            sub_qs.append(original)
        return RewriteResult(main_query=main_q, sub_queries=sub_qs)
    except json.JSONDecodeError:
        pass

    # 第五步：实在解析不了，静默回退到原始查询（不报错打断流程）
    console.print(f"[dim]⚠ Query Rewrite 解析失败，使用原始查询[/dim]")
    return RewriteResult(main_query=original, sub_queries=[original] * num)


# ─────────────────────────────────────────────
# DeepSeek API 改写（推荐，中文理解最强）
# ─────────────────────────────────────────────
class DeepSeekQueryRewriter:
    """
    DeepSeek API 完全兼容 OpenAI SDK
    只需设置 base_url 和 api_key，其余与 OpenAI 调用方式完全相同
    价格约为 GPT-4o 的 1/30，中文 Query Rewrite 效果优秀
    """

    def __init__(self, config: RAGConfig = DEFAULT_CONFIG):
        import httpx
        from openai import OpenAI
        self.client = OpenAI(
            api_key=config.deepseek_api_key,
            base_url=config.query_rewriter.deepseek_base_url,
            http_client=httpx.Client(verify=False),
        )
        self.cfg = config.query_rewriter

    def rewrite(self, question: str,
                history: list = None) -> RewriteResult:
        # 把历史最近4条（2轮）格式化成文本，帮助模型理解上下文
        history_text = "无"
        if history:
            recent = history[-4:]
            lines = []
            for msg in recent:
                role = "用户" if msg["role"] == "user" else "助手"
                lines.append(f"{role}：{msg['content'][:80]}")
            history_text = "\n".join(lines)

        prompt = REWRITE_PROMPT.format(
            question=question,
            history=history_text,
            num=self.cfg.num_sub_queries,
        )
        try:
            response = self.client.chat.completions.create(
                model=self.cfg.deepseek_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.3,
                response_format={"type": "json_object"},
                extra_body={"enable_thinking": False},
            )
            text = response.choices[0].message.content or ""
        except Exception as e:
            console.print(f"[dim]⚠ Query Rewrite API 失败: {e}，使用原始查询[/dim]")
            return RewriteResult(
                main_query=question,
                sub_queries=[question] * self.cfg.num_sub_queries
            )
        return _parse_rewrite_output(text, question, self.cfg.num_sub_queries)
class LocalQueryRewriter:
    def __init__(self, config: RAGConfig = DEFAULT_CONFIG):
        self.cfg = config.query_rewriter
        self.tokenizer = None
        self.model = None

    def _load(self):
        if self.model is not None:
            return
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        console.print(f"[cyan]加载 Query Rewriter 模型: {self.cfg.local_model}[/cyan]")
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.local_model, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.local_model,
            device_map=self.cfg.local_device,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

    def rewrite(self, question: str) -> RewriteResult:
        self._load()
        import torch
        prompt = REWRITE_PROMPT.format(question=question, num=self.cfg.num_sub_queries)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.cfg.local_device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.3,
                do_sample=True,
            )
        text = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return _parse_rewrite_output(text, question, self.cfg.num_sub_queries)


# ─────────────────────────────────────────────
# Anthropic API 改写
# ─────────────────────────────────────────────
class AnthropicQueryRewriter:
    def __init__(self, config: RAGConfig = DEFAULT_CONFIG):
        self.cfg = config.query_rewriter
        self.api_key = config.anthropic_api_key

    def rewrite(self, question: str) -> RewriteResult:
        import anthropic
        client = anthropic.Anthropic(api_key=self.api_key)
        prompt = REWRITE_PROMPT.format(question=question, num=self.cfg.num_sub_queries)
        # 使用 Haiku：改写任务简单，Haiku 速度快、成本低
        response = client.messages.create(
            model=self.cfg.anthropic_model,   # claude-haiku-4-5
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text
        return _parse_rewrite_output(text, question, self.cfg.num_sub_queries)


# ─────────────────────────────────────────────
# OpenAI API 改写
# ─────────────────────────────────────────────
class OpenAIQueryRewriter:
    def __init__(self, config: RAGConfig = DEFAULT_CONFIG):
        self.cfg = config.query_rewriter
        self.api_key = config.openai_api_key

    def rewrite(self, question: str) -> RewriteResult:
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key)
        prompt = REWRITE_PROMPT.format(question=question, num=self.cfg.num_sub_queries)
        response = client.chat.completions.create(
            model=self.cfg.openai_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content
        return _parse_rewrite_output(text, question, self.cfg.num_sub_queries)


# ─────────────────────────────────────────────
# 工厂函数（统一入口）
# ─────────────────────────────────────────────
class QueryRewriter:
    """
    统一接口，根据 config.query_rewriter.api_provider 自动路由：
      "deepseek"  → DeepSeek API（默认）
      "anthropic" → Claude Haiku
      "openai"    → GPT-4o-mini
      "local"     → 本地 Qwen3-7B
    """

    def __init__(self, config: RAGConfig = DEFAULT_CONFIG):
        self.cfg = config.query_rewriter
        provider = self.cfg.api_provider if self.cfg.mode == "api" else "local"
        if provider == "deepseek":
            self._backend = DeepSeekQueryRewriter(config)
        elif provider == "anthropic":
            self._backend = AnthropicQueryRewriter(config)
        elif provider == "openai":
            self._backend = OpenAIQueryRewriter(config)
        else:
            self._backend = LocalQueryRewriter(config)

    def rewrite(self, question: str,
                history: list = None) -> RewriteResult:
        console.print(f"\n[bold cyan]🔄 Query Rewrite[/bold cyan]: {question}")
        result = self._backend.rewrite(question, history=history or [])
        console.print(f"  main_query: [green]{result.main_query}[/green]")
        for i, q in enumerate(result.sub_queries):
            console.print(f"  sub_query [{i+1}]: {q}")
        return result
