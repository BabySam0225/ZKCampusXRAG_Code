"""
eval_autogen.py — 从知识库自动生成测试用例
=============================================
流程：
1. 从知识库随机抽取 chunk
2. 让 DeepSeek 根据 chunk 内容生成问题 + 参考答案
3. 对生成的问题跑一遍 RAG，自动评测回答准确度
4. 把结果存入测试集

用法：
python eval_autogen.py generate --system main --n 20
python eval_autogen.py generate --system mental --n 10
python eval_autogen.py generate --system career --n 15
python eval_autogen.py generate --all  # 三个系统各生成默认数量
python eval_autogen.py evaluate --system main  # 评测已生成的用例
"""

import os
import json
import random
import argparse
import hashlib
import httpx
from openai import OpenAI
from typing import List, Tuple
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from eval_dataset import EvalDataset, EvalCase, EVAL_DIR
from eval_metrics import LLMJudge, SYSTEM_METRICS
from document_processor import Document
from dotenv import load_dotenv

load_dotenv()
console = Console()

# ─────────────────────────────────────────────
# 问题生成 Prompt（按系统区分）
# ─────────────────────────────────────────────
GEN_PROMPTS = {
    "main": """你是一个测试用例生成专家。 根据以下知识库内容， 生成一个大学生会向辅导员助手提问的问题和参考答案。
【知识库内容】
{chunk}

要求：
1. 问题要像真实学生提问， 口语化， 不要太正式
2. 问题必须能从上面的内容中找到答案
3. 参考答案要准确、 完整， 直接基于内容回答
4. 难度选择： easy（直接查询）/ medium（需要理解）/ hard（需要综合推理）
只返回 JSON：
{{"question": "问题内容", "reference": "参考答案", "difficulty": "easy/medium/hard", "tags": ["相关标签1", "相关标签2"]}}""",

    "mental": """你是一个测试用例生成专家。 根据以下心理知识内容， 生成一个大学生可能向心理辅导助手倾诉的内容和理想回应要点。
【心理知识内容】
{chunk}

要求：
1. 问题要模拟真实学生的情绪表达， 有具体情境
2. 参考答案描述理想回应的要点（ 共情方式、 建议方向）
3. 不要生成与自伤相关的极端内容
只返回 JSON：
{{"question": "学生的倾诉内容", "reference": "理想回应要点", "difficulty": "easy/medium/hard", "tags": ["相关标签1", "相关标签2"]}}""",

    "career": """你是一个测试用例生成专家。 根据以下就业数据内容， 生成一个毕业生会向就业指导助手提问的问题和参考答案。
【就业数据内容】
{chunk}

要求：
1. 问题要像真实毕业生在求职时的困惑， 具体而实际
2. 参考答案必须引用内容中的具体数据（薪酬、比例、公司名等）
3. 难度： easy（查单一数据）/ medium（需比较）/ hard（需综合判断）
只返回 JSON：
{{"question": "毕业生的问题", "reference": "基于数据的参考答案", "difficulty": "easy/medium/hard", "tags": ["相关标签1", "相关标签2"]}}"""
}

# 各系统的知识库路径
STORAGE_PATHS = {
    "main": "./storage/docs.json",
    "mental": "./mental_storage/docs.json",
    "career": "./career_storage/docs.json",
}

# ─────────────────────────────────────────────
# 从知识库随机抽取 chunk
# ─────────────────────────────────────────────
def sample_chunks(system: str, n: int, min_len: int = 80) -> List[Document]:
    """从对应知识库随机抽取 n 个有效 chunk"""
    path = STORAGE_PATHS.get(system)
    if not path or not os.path.exists(path):
        console.print(f"[red]❌ {system} 知识库不存在： {path}[/red]")
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_docs = data if isinstance(data, list) else data.get("docs", data)
    # 过滤太短的 chunk（内容不够生成问题）
    valid = [d for d in all_docs if len(d.get("content", "")) >= min_len]

    if not valid:
        console.print(f"[yellow]⚠ {system} 知识库无有效 chunk[/yellow]")
        return []

    sampled = random.sample(valid, min(n, len(valid)))
    console.print(f" 从 {len(valid)} 个 chunk 中抽取 {len(sampled)} 个")
    return [Document(
        doc_id=d.get("doc_id", ""),
        content=d.get("content", ""),
        source=d.get("source", ""),
        chunk_index=d.get("chunk_index", 0),
        metadata=d.get("metadata", {}),
    ) for d in sampled]

# ─────────────────────────────────────────────
# 调用 LLM 生成问题
# ─────────────────────────────────────────────
def get_llm_client():
    import os
    # 优先用 siliconflow qwen3，其次 deepseek
    sf_key = os.getenv("SILICONFLOW_API_KEY", "")
    ds_key = os.getenv("DEEPSEEK_API_KEY", "")
    if sf_key:
        return OpenAI(
            api_key=sf_key,
            base_url="https://api.deepseek.com",
            http_client=httpx.Client(verify=False),
        ), "deepseek-chat"
    return OpenAI(
        api_key=ds_key,
        base_url="https://api.siliconflow.cn/v1",
        http_client=httpx.Client(verify=False),
    ), "Qwen/Qwen3-235B-A22B"

def generate_qa_from_chunk(chunk: Document, system: str,
                           client: OpenAI, model: str) -> dict | None:
    """用 LLM 根据一个 chunk 生成一条问答对"""
    prompt_template = GEN_PROMPTS.get(system, GEN_PROMPTS["main"])
    # 截取 chunk 前 600 字，避免 prompt 过长
    chunk_text = chunk.content[:600]
    prompt = prompt_template.format(chunk=chunk_text)

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.7,  # 适当提高创造性， 让问题更多样
            response_format={"type": "json_object"},
            extra_body={"enable_thinking": False} if "Qwen" in model else {},
        )
        text = resp.choices[0].message.content or ""
        data = json.loads(text)
        # 验证必要字段
        if not data.get("question") or not data.get("reference"):
            return None
        # 去重检查（用问题内容哈希）
        q_hash = hashlib.md5(data["question"].encode()).hexdigest()[:8]
        return {
            "question": data["question"].strip(),
            "reference": data["reference"].strip(),
            "difficulty": data.get("difficulty", "medium"),
            "tags": data.get("tags", []),
            "source_chunk": chunk.doc_id,
            "gen_hash": q_hash,
        }
    except Exception as e:
        console.print(f"[dim] 生成失败: {e}[/dim]")
        return None

# ─────────────────────────────────────────────
# 主生成流程
# ─────────────────────────────────────────────
def generate_cases(system: str, n: int = 20, dataset_name: str = "auto"):
    """为指定系统生成 n 条测试用例并保存到测试集"""
    console.print(Panel(
        f"[bold green]✅ 自动生成测试用例[/bold green]\n"
        f"系统： {system} 目标数量： {n}",
        border_style="green"
    ))

    # 1. 抽取 chunk（多抽一些，预留失败余量）
    chunks = sample_chunks(system, n=int(n * 1.5))
    if not chunks:
        return []

    # 2. 初始化 LLM 客户端
    client, model = get_llm_client()
    console.print(f" 使⽤模型： {model}")

    # 3. 生成问答对
    generated = []
    seen_hashes = set()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"生成 {system} 测试用例...", total=n)

        for chunk in chunks:
            if len(generated) >= n:
                break
            progress.update(task, description=f"处理 chunk {chunk.doc_id[:8]}...")
            qa = generate_qa_from_chunk(chunk, system, client, model)
            if qa and qa["gen_hash"] not in seen_hashes:
                seen_hashes.add(qa["gen_hash"])
                generated.append(qa)
                progress.advance(task)

    # 4. 保存到测试集
    ds = EvalDataset(dataset_name).load()
    new_cases = []
    for qa in generated:
        case = EvalCase(
            question=qa["question"],
            reference=qa["reference"],
            system=system,
            difficulty=qa["difficulty"],
            tags=qa["tags"],
            source="auto",
            notes=f"来源chunk: {qa['source_chunk']}",
        )
        ds.add(case)
        new_cases.append(case)

    ds.save()

    # 5. 展示结果
    t = Table(title=f"生成完成： {len(generated)} 条", show_lines=True)
    t.add_column("难度", width=8)
    t.add_column("标签", width=15)
    t.add_column("问题预览", width=50)
    for case in new_cases[:10]:
        t.add_row(case.difficulty, ", ".join(case.tags[:2]), case.question[:48])
    if len(new_cases) > 10:
        t.add_row("...", "...", f"（ 共 {len(new_cases)} 条） ")
    console.print(t)
    return new_cases

# ─────────────────────────────────────────────
# 评测已生成用例的准确度
# ─────────────────────────────────────────────
def evaluate_auto_cases(system: str, dataset_name: str = "auto", limit: int = None):
    """
    对自动生成的测试用例跑完整 RAG 流程并评测
    相当于： 生成问题 → 跑 RAG → 用 LLM 裁判打分 → 看准确度
    """
    console.print(Panel(
        f"[bold cyan]📊 评测自动生成用例[/bold cyan]\n系统： {system}",
        border_style="cyan"
    ))

    ds = EvalDataset(dataset_name).load()
    # source_filter 不是内置参数，手动过滤
    cases = [c for c in ds.cases if c.system == system and c.source == "auto"]
    if limit:
        cases = cases[:limit]

    if not cases:
        console.print(f"[yellow]没有找到 {system} 的自动生成用例， 先运行 generate[/yellow]")
        return
    console.print(f" 找到 {len(cases)} 条自动生成用例， 开始评测...")

    # 加载对应 pipeline
    try:
        if system == "main":
            from pipeline import RAGPipeline
            from rag_config import DEFAULT_CONFIG
            pipeline = RAGPipeline(DEFAULT_CONFIG)
            pipeline.load_index()
        elif system == "mental":
            from mental_pipeline import MentalPipeline
            pipeline = MentalPipeline()
            pipeline.load_index()
        elif system == "career":
            from career_pipeline import CareerPipeline
            pipeline = CareerPipeline()
            pipeline.load_index()
    except Exception as e:
        console.print(f"[red]❌ 加载 pipeline 失败: {e}[/red]")
        return

    judge = LLMJudge()
    results = []
    passed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("评测中...", total=len(cases))

        for case in cases:
            progress.update(task, description=case.question[:30] + "...")

            # 获取 RAG 回答（非流式）
            try:
                if system == "main":
                    rewrite = pipeline.query_rewriter.rewrite(case.question)
                    candidates = pipeline.retriever.multi_query_search(rewrite.all_queries())
                    ranked = pipeline.reranker.get_top_docs(rewrite.main_query, candidates)
                    compressed = pipeline.compressor.compress(rewrite.main_query, ranked)
                    context = pipeline.compressor.format_context(compressed)
                    rag_result = pipeline.generator.generate(
                        case.question, context, compressed, stream=False
                    )
                    answer = rag_result.answer
                else:
                    answer = pipeline.query(case.question, enable_thinking=False)
                    context = ""
            except Exception as e:
                answer = f"[ERROR] {e}"
                context = ""

            # 自动评分
            eval_result = judge.evaluate(
                case_id=case.case_id,
                question=case.question,
                answer=answer,
                reference=case.reference,
                system=system,
                context=context,
            )
            results.append((case, answer, eval_result))
            if eval_result.passed:
                passed += 1

            status = "✅" if eval_result.passed else "❌"
            console.print(
                f" {status} 综合:{eval_result.overall:.1f} "
                f"{' '.join(f'{k}:{v.score:.0f}' for k,v in eval_result.scores.items())}"
            )
            progress.advance(task)

    # 汇总
    pass_rate = passed / len(results) if results else 0
    avg_score = sum(r.overall for _, _, r in results) / len(results) if results else 0

    # 分指标平均
    metric_avgs = {}
    for _, _, r in results:
        for k, v in r.scores.items():
            metric_avgs.setdefault(k, []).append(v.score)
    metric_avgs = {k: sum(v)/len(v) for k, v in metric_avgs.items()}

    console.print(Panel(
        f"[bold]评测结果： {system} 系统[/bold]\n\n"
        f" 测试用例： {len(results)} 条\n"
        f" 通过率： [{'green' if pass_rate>=0.7 else 'red'}]{pass_rate*100:.1f}%[/{'green' if pass_rate>=0.7 else 'red'}]\n"
        f" 平均综合分： {avg_score:.2f}/5\n\n"
        f" 各指标： " + " ".join(f"{k}:{v:.2f}" for k, v in metric_avgs.items()),
        border_style="green" if pass_rate >= 0.7 else "red"
    ))

    # 找出最差的几条，给出改进方向
    worst = sorted(results, key=lambda x: x[2].overall)[:3]
    if worst and worst[0][2].overall < 3.0:
        console.print("\n[bold yellow]⚠ 最需改进的问题： [/bold yellow]")
        for case, answer, r in worst:
            worst_metric = min(r.scores.items(), key=lambda x: x[1].score)
            console.print(f" [{case.case_id}] {case.question[:40]}...")
            console.print(f"综合:{r.overall:.1f} 最差指标: {worst_metric[0]}={worst_metric[1].score:.1f}")
            console.print(f"原因: {worst_metric[1].reason[:60]}")

    # 保存评测报告
    report = {
        "system": system,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "total": len(results),
        "passed": passed,
        "pass_rate": pass_rate,
        "avg_score": avg_score,
        "metric_avgs": metric_avgs,
        "cases": [
            {
                "case_id": case.case_id,
                "question": case.question,
                "reference": case.reference,
                "answer": answer,
                "overall": r.overall,
                "passed": r.passed,
                "scores": {k: {"score": v.score, "reason": v.reason} for k, v in r.scores.items()},
            }
            for case, answer, r in results
        ]
    }
    os.makedirs(EVAL_DIR, exist_ok=True)
    report_path = os.path.join(EVAL_DIR, f"auto_eval_{system}_{datetime.now().strftime('%Y%m%d_%H%M')}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    console.print(f"[dim]报告已保存： {report_path}[/dim]")

    return pass_rate, avg_score

# ─────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="知识库自动生成测试用例 + 评测")
    sub = parser.add_subparsers(dest="cmd")

    gen_p = sub.add_parser("generate", help="从知识库生成测试用例")
    gen_p.add_argument("--system", choices=["main", "mental", "career"], help="指定系统")
    gen_p.add_argument("--all", action="store_true", help="三个系统全部生成")
    gen_p.add_argument("--n", type=int, default=20, help="每个系统生成数量， 默认20")
    gen_p.add_argument("--dataset", default="auto", help="保存到哪个测试集， 默认 auto")

    eval_p = sub.add_parser("evaluate", help="评测已生成的用例")
    eval_p.add_argument("--system", choices=["main", "mental", "career"], required=True)
    eval_p.add_argument("--dataset", default="auto")
    eval_p.add_argument("--limit", type=int, help="只评测前 N 条")

    both_p = sub.add_parser("run", help="生成 + 立即评测（ 一键完成） ")
    both_p.add_argument("--system", choices=["main", "mental", "career"], required=True)
    both_p.add_argument("--n", type=int, default=20)
    both_p.add_argument("--dataset", default="auto")

    args = parser.parse_args()
    if args.cmd == "generate":
        systems = ["main", "mental", "career"] if args.all else [args.system]
        for sys in systems:
            generate_cases(sys, n=args.n, dataset_name=args.dataset)

    elif args.cmd == "evaluate":
        evaluate_auto_cases(args.system, args.dataset, args.limit)

    elif args.cmd == "run":
        # 生成 + 评测一步完成
        cases = generate_cases(args.system, n=args.n, dataset_name=args.dataset)
        if cases:
            evaluate_auto_cases(args.system, args.dataset)

    else:
        parser.print_help()
