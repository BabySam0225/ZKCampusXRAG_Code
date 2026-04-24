"""
generator.py — LLM 生成层
支持多轮对话记忆：history 参数传入历史消息列表
"""

from dataclasses import dataclass, field
from typing import List, Optional

from rich.console import Console

from rag_config import RAGConfig, DEFAULT_CONFIG

console = Console()


# ─────────────────────────────────────────────
# Prompt 模板
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """你是一个严格基于知识库回答问题的助手。

核心规则：
1. 你必须仅根据提供的【参考上下文】回答问题
2. 如果上下文中没有足够信息，必须回答："未在知识库中找到相关信息"
3. 禁止使用上下文以外的知识
4. 回答时必须引用来源编号，格式：[来源 N]
5. 可以结合对话历史理解用户的追问意图，但答案内容必须来自上下文
6. 回答要简洁、准确、有结构"""

ANSWER_PROMPT = """参考上下文：
{context}

用户问题：{question}

请基于以上上下文回答问题，并在回答中标注引用的来源编号（如 [来源 1]）。"""


# ─────────────────────────────────────────────
# 对话历史条目
# ─────────────────────────────────────────────
@dataclass
class RAGAnswer:
    question: str
    answer: str
    sources: List[dict]
    model: str


def _build_messages(
    question: str,
    context: str,
    history: List[dict],  # [{"role": "user/assistant", "content": "..."}, ...]
) -> List[dict]:
    """
    组装发给 LLM 的完整消息列表：
      system
      [历史 user/assistant 轮次]   ← 让 LLM 知道之前聊了什么
      当前 user（含检索到的上下文）
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # 加入历史（最多保留最近 MAX_HISTORY 轮，避免 token 爆炸）
    MAX_HISTORY = 12  # 3轮问答 = 6条消息
    messages.extend(history[-MAX_HISTORY:])
    # 当前问题（带上下文）
    prompt = ANSWER_PROMPT.format(context=context, question=question)
    messages.append({"role": "user", "content": prompt})
    return messages


# ─────────────────────────────────────────────
# DeepSeek API 生成
# ─────────────────────────────────────────────
class DeepSeekGenerator:
    def __init__(self, config: RAGConfig):
        import httpx
        from openai import OpenAI
        self.client = OpenAI(
            api_key=config.deepseek_api_key,
            base_url=config.generator.deepseek_base_url,
            http_client=httpx.Client(verify=False),
        )
        self.cfg = config.generator

    def generate(self, question: str, context: str, history: List[dict],
                 stream: bool = True, enable_thinking: bool = False) -> str:
        
        model = self.cfg.deepseek_reasoner_model if enable_thinking else self.cfg.deepseek_model
        messages = _build_messages(question, context, history)

        # 新版 DeepSeek 模型默认开启思考，必须显式传 enable_thinking 才能控制
        extra = {"extra_body": {"enable_thinking": enable_thinking}}

        if stream:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=self.cfg.max_tokens,
                temperature=self.cfg.temperature,
                stream=True,
                **extra,
            )
            full = ""
            in_thinking = False
            print("\n", end="", flush=True)
            for chunk in response:
                delta = chunk.choices[0].delta
                thinking = getattr(delta, "reasoning_content", None) or ""
                content_text = delta.content or ""
                # enable_thinking=False 时直接忽略 reasoning_content，不输出
                if thinking and enable_thinking:
                    if not in_thinking:
                        print("\n\033[2m[深度思考] ", end="", flush=True)
                        in_thinking = True
                    print(thinking, end="", flush=True)
                if content_text:
                    if in_thinking:
                        print("\033[0m\n", end="", flush=True)
                        in_thinking = False
                    print(content_text, end="", flush=True)
                    full += content_text
            print()
            return full
        else:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=self.cfg.max_tokens,
                temperature=self.cfg.temperature,
                **extra,
            )
            return response.choices[0].message.content


# ─────────────────────────────────────────────
# Anthropic API 生成
# ─────────────────────────────────────────────
class AnthropicGenerator:
    def __init__(self, config: RAGConfig):
        import anthropic
        self.client = anthropic.Anthropic(api_key=config.anthropic_api_key)
        self.cfg = config.generator

    def generate(self, question: str, context: str, history: List[dict], stream: bool = False) -> str:
        messages = _build_messages(question, context, history)
        system = messages[0]["content"]
        user_messages = messages[1:]
        if stream:
            full = ""
            print("\n", end="", flush=True)
            with self.client.messages.stream(
                model=self.cfg.anthropic_model,
                max_tokens=self.cfg.max_tokens,
                system=system,
                messages=user_messages,
            ) as s:
                for delta in s.text_stream:
                    print(delta, end="", flush=True)
                    full += delta
            print()
            return full
        else:
            response = self.client.messages.create(
                model=self.cfg.anthropic_model,
                max_tokens=self.cfg.max_tokens,
                system=system,
                messages=user_messages,
            )
            return response.content[0].text


# ─────────────────────────────────────────────
# OpenAI API 生成
# ─────────────────────────────────────────────
class OpenAIGenerator:
    def __init__(self, config: RAGConfig):
        import httpx
        from openai import OpenAI
        self.client = OpenAI(
            api_key=config.openai_api_key,
            http_client=httpx.Client(verify=False),
        )
        self.cfg = config.generator

    def generate(self, question: str, context: str, history: List[dict], stream: bool = False) -> str:
        messages = _build_messages(question, context, history)
        if stream:
            response = self.client.chat.completions.create(
                model=self.cfg.openai_model,
                messages=messages,
                max_tokens=self.cfg.max_tokens,
                temperature=self.cfg.temperature,
                stream=True,
            )
            full = ""
            print("\n", end="", flush=True)
            for chunk in response:
                delta = chunk.choices[0].delta.content or ""
                print(delta, end="", flush=True)
                full += delta
            print()
            return full
        else:
            response = self.client.chat.completions.create(
                model=self.cfg.openai_model,
                messages=messages,
                max_tokens=self.cfg.max_tokens,
                temperature=self.cfg.temperature,
            )
            return response.choices[0].message.content


# ─────────────────────────────────────────────
# 本地模型生成
# ─────────────────────────────────────────────
class LocalGenerator:
    def __init__(self, config: RAGConfig):
        self.cfg = config.generator
        self.tokenizer = None
        self.model = None

    def _load(self):
        if self.model is not None:
            return
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        import torch
        console.print(f"[cyan]加载生成模型: {self.cfg.local_model}[/cyan]")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.local_model, trust_remote_code=True
        )
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16) \
            if self.cfg.load_in_4bit else None
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.local_model,
            device_map="auto",
            quantization_config=quant_config,
            trust_remote_code=True,
        )

    def generate(self, question: str, context: str, history: List[dict]) -> str:
        self._load()
        import torch
        messages = _build_messages(question, context, history)
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_tokens,
                temperature=self.cfg.temperature,
                do_sample=(self.cfg.temperature > 0),
            )
        return self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()


# ─────────────────────────────────────────────
# 统一生成接口
# ─────────────────────────────────────────────
class Generator:
    def __init__(self, config: RAGConfig = DEFAULT_CONFIG):
        self.cfg = config.generator
        provider = self.cfg.api_provider if self.cfg.mode == "api" else "local"
        if provider == "deepseek":
            self._backend = DeepSeekGenerator(config)
        elif provider == "anthropic":
            self._backend = AnthropicGenerator(config)
        elif provider == "openai":
            self._backend = OpenAIGenerator(config)
        else:
            self._backend = LocalGenerator(config)

    def generate(
        self,
        question: str,
        context: str,
        compressed_chunks: List[dict],
        history: List[dict] = None,
        stream: bool = True,
        enable_thinking: bool = False,   # True 时切换到 deepseek-reasoner（R1）
    ) -> RAGAnswer:
        console.print(
            f"\n[bold cyan]🤖 LLM 生成[/bold cyan]"
            + (" [yellow]（深度思考 R1）[/yellow]" if enable_thinking else "") + "..."
        )
        answer = self._backend.generate(
            question, context, history or [],
            stream=stream,
            enable_thinking=enable_thinking,
        )
        model_name = (
            self.cfg.deepseek_model if self.cfg.api_provider == "deepseek"
            else self.cfg.local_model if self.cfg.mode == "local"
            else self.cfg.anthropic_model if self.cfg.api_provider == "anthropic"
            else self.cfg.openai_model
        )
        return RAGAnswer(
            question=question,
            answer=answer,
            sources=compressed_chunks,
            model=model_name,
        )
