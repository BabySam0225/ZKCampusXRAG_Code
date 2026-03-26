"""
career_pipeline.py  —  就业指导 RAG 流⽔线
"""
import os
import json
from typing import List
import httpx
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from career_config import CAREER_CONFIG, CareerRAGConfig, MAJOR_MAP
from document_processor import DocumentProcessor
from hybrid_search import HybridRetriever
from reranker import Reranker
from context_compressor import ContextCompressor
from query_rewriter import QueryRewriter

console = Console()

CAREER_SYSTEM_PROMPT = """你是一位专业、亲切的⼤学⽣就业指导⽼师，专门帮助应届毕业⽣规划就业
【你的两类知识】
1. 本校真实就业数据：各专业就业去向、薪酬⽔平、热⻔企业、升学率等（来⾃知识库）
2. 通⽤职场知识：岗位技能要求、学习路线、求职技巧、⾏业发展趋势等（来⾃你⾃身知识）
【回答原则】
1. 就业数据类问题（去哪⾥、薪酬多少、什么公司）→ 优先引⽤知识库真实数据，给出具体数字
2. 岗位技能类问题（需要学什么、怎么准备）→ 直接⽤你的专业知识回答，结合该岗位在本校的真实招录情况
3. 两类问题结合时 → 先给数据，再给技能建议，形成完整回答
例如："据数据，软件开发岗是⽹络专业最热⻔去向，平均薪酬 6000 元。要拿到这类岗位，建议重点掌握..."
4. 数据要具体，技能建议要可操作（列出具体技术/⼯具名称，不要泛泛⽽谈）
5. 语⽓友善，像学⻓/学姐⼀样分享经验，不要说教
6. 对升学 vs 就业的问题，客观呈现数据，不主观评判
【专业简称对应】
⽹络=网络工程技术，电子=电子信息技术，通信=通信技术，
⼤数据=数据科学与大数据技术，计算机=计算机科学与技术，物联⽹=物联⽹工程技术，信管=信息管理
"""

CAREER_ANSWER_PROMPT = """【就业数据参考】
{context}
【同学的问题】
{question}
请结合以上数据，给出具体、有参考价值的就业指导建议。
"""

class CareerGenerator:
    def __init__(self, config: CareerRAGConfig):
        self.client = OpenAI(
            api_key=config.deepseek_api_key,
            base_url=config.generator.deepseek_base_url,
            http_client=httpx.Client(verify=False),
        )
        self.cfg = config.generator
        self.MAX_HISTORY = 6

    def generate(self, question: str, context: str, history: List[dict]) -> str:
        messages = [{"role": "system", "content": CAREER_SYSTEM_PROMPT}]
        messages.extend(history[-self.MAX_HISTORY:])
        
        if context.strip():
            prompt = CAREER_ANSWER_PROMPT.format(context=context, question=question)
        else:
            prompt = f"同学的问题：{question}\n\n请根据就业指导经验给出建议。"
            
        messages.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(
            model=self.cfg.deepseek_model,
            messages=messages,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
        )
        return response.choices[0].message.content

class CareerPipeline:
    def __init__(self, config: CareerRAGConfig = CAREER_CONFIG):
        self.config = config
        self.processor = DocumentProcessor(config)
        self.retriever = HybridRetriever(config)
        self.reranker = Reranker(config)
        self.compressor = ContextCompressor(config)
        self.query_rewriter = QueryRewriter(config)
        self.generator = CareerGenerator(config)
        self._index_built = False

    def load_index(self):
        docs_path = os.path.join(self.config.storage_dir, "docs.json")
        if not os.path.exists(docs_path):
            console.print(Panel(
                "[bold red]\n就业知识库不存在！[/bold red]\n\n"
                "请先运⾏：\n"
                "[bold yellow]python build_career_kb.py build --files 就业信息下载 2025 年.xlsx[/bold yellow]",
                border_style="red"
            ))
            raise FileNotFoundError("就业知识库未建⽴")
        
        docs = self.processor.load_docs(docs_path)
        self.retriever.load(docs)
        self._index_built = True
        console.print("[green] ✓ 就业知识库加载完成 [/green]")

    def query(self, question: str, history: List[dict] = None) -> str:
        history = history or []
        context = ""
        try:
            rewrite_result = self.query_rewriter.rewrite(question)
            all_queries = rewrite_result.all_queries()
            candidates = self.retriever.multi_query_search(all_queries)
            if candidates:
                ranked_docs = self.reranker.get_top_docs(rewrite_result.main_query, candidates)
                compressed = self.compressor.compress(rewrite_result.main_query, ranked_docs)
                context = self.compressor.format_context(compressed)
        except Exception as e:
            console.print(f"[dim]检索时遇到问题：{e}[/dim]")
        
        answer = self.generator.generate(question, context, history)
        console.print(Panel(answer, title="[bold green]就业指导[/bold green]", border_style="green"))
        return answer

    def chat(self):
        console.print(Panel(
            "[bold green]\n就业指导助⼿已就绪 [/bold green]\n\n"
            "你好！我是你的就业指导助⼿，掌握本校近⼏年真实就业数据。\n"
            "你可以问我：某专业就业⽅向、薪酬⽔平、推荐公司、要不要考研...\n\n"
            "输⼊ [yellow]clear[/yellow] 开始新话题，输⼊ [yellow]exit[/yellow] 退出",
            border_style="green"
        ))
        history = []
        while True:
            try:
                user_input = input("\n你：").strip()
                if not user_input:
                    continue
                if user_input.lower() in {"exit", "quit", "q"}:
                    console.print("\n[green]祝你求职顺利！\n[/green]")
                    break
                if user_input.lower() == "clear":
                    history.clear()
                    console.print("[cyan] ✓ 已开始新话题 [/cyan]")
                    continue
                
                answer = self.query(user_input, history=history)
                history.append({"role": "user", "content": user_input})
                history.append({"role": "assistant", "content": answer})
            except KeyboardInterrupt:
                console.print("\n[green]祝你求职顺利！\n[/green]")
                break
            except Exception as e:
                console.print(f"[red]出错了：{e}[/red]")