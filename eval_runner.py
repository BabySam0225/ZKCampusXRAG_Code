"""
eval_runner.py — 自动化评测运行器
====================================
对测试集中的每条用例：
1. 调用对应系统（main/mental/career）获取回答
2. 用 LLMJudge 自动评分
3. 保存结果（含版本标签）

用法：
python eval_runner.py run                        # 评测全部用例
python eval_runner.py run --system main          # 只评测辅导员系统
python eval_runner.py run --tag v1.0 --system career
python eval_runner.py list                       # 列出历史评测
"""

import os
import json
import time
import argparse
import uuid
from datetime import datetime
from typing import List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from eval_dataset import EvalDataset, EvalCase, EVAL_DIR
from eval_metrics import LLMJudge, EvalResult

console = Console()
RESULTS_DIR = os.path.join(EVAL_DIR, "results")


def get_pipeline(system: str):
    """按系统名称加载对应 Pipeline"""
    if system == "main":
        from pipeline import RAGPipeline
        from rag_config import DEFAULT_CONFIG
        p = RAGPipeline(DEFAULT_CONFIG)
        p.load_index()
        return p
    elif system == "mental":
        from mental_pipeline import MentalPipeline
        p = MentalPipeline()
        p.load_index()
        return p
    elif system == "career":
        from career_pipeline import CareerPipeline
        p = CareerPipeline()
        p.load_index()
        return p
    else:
        raise ValueError(f"未知系统： {system}")


def get_answer(pipeline, system: str, question: str) -> tuple:
    """获取 AI 回答，返回 (answer, context, latency)"""
    t0 = time.time()
    context = ""
    try:
        if system == "main":
            # 关闭流式，直接返回文本
            rewrite = pipeline.query_rewriter.rewrite(question)
            candidates = pipeline.retriever.multi_query_search(rewrite.all_queries())
            ranked = pipeline.reranker.get_top_docs(rewrite.main_query, candidates)
            compressed = pipeline.compressor.compress(rewrite.main_query, ranked)
            context = pipeline.compressor.format_context(compressed)
            result = pipeline.generator.generate(question, context, compressed, stream=False)
            answer = result.answer
        elif system == "mental":
            answer = pipeline.query(question)
        elif system == "career":
            answer = pipeline.query(question)
        else:
            answer = ""
    except Exception as e:
        answer = f"[ERROR] {e}"
    latency = time.time() - t0
    return answer, context, latency


class EvalRunner:
    def __init__(self, tag: str = ""):
        self.tag = tag or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{self.tag}_{str(uuid.uuid4())[:6]}"
        self.judge = LLMJudge()
        os.makedirs(RESULTS_DIR, exist_ok=True)

    def run(self, cases: List[EvalCase]) -> List[EvalResult]:
        """运行所有测试用例并评分"""
        results = []
        pipelines = {}

        console.print(Panel(
            f"[bold green] 开始评测[/bold green]\n"
            f"运行ID： {self.run_id}\n"
            f"测试用例： {len(cases)} 条",
            border_style="green"
        ))

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("评测中...", total=len(cases))

            for case in cases:
                progress.update(task, description=f"[{case.system}] {case.question[:30]}...")

                # 懒加载 pipeline
                if case.system not in pipelines:
                    try:
                        pipelines[case.system] = get_pipeline(case.system)
                    except Exception as e:
                        console.print(f"[red]加载 {case.system} 失败: {e}[/red]")
                        pipelines[case.system] = None

                pipeline = pipelines.get(case.system)
                if pipeline is None:
                    result = EvalResult(
                        case_id=case.case_id,
                        question=case.question,
                        answer="",
                        reference=case.reference,
                        system=case.system,
                        error="Pipeline 加载失败"
                    )
                    results.append(result)
                    progress.advance(task)
                    continue

                # 获取回答
                answer, context, latency = get_answer(pipeline, case.system, case.question)

                # 自动评分
                result = self.judge.evaluate(
                    case_id=case.case_id,
                    question=case.question,
                    answer=answer,
                    reference=case.reference,
                    system=case.system,
                    context=context,
                    latency=latency,
                )
                results.append(result)

                status = "✅" if result.passed else "❌"
                console.print(f" {status} [{case.case_id}] 综合:{result.overall:.1f} | {case.question[:40]}")
                progress.advance(task)

        return results

    def save(self, results: List[EvalResult], cases: List[EvalCase]):
        """保存评测结果"""
        passed = sum(1 for r in results if r.passed)
        avg_score = sum(r.overall for r in results) / len(results) if results else 0
        avg_latency = sum(r.latency for r in results) / len(results) if results else 0

        # 分指标平均分
        all_metrics = {}
        for r in results:
            for k, v in r.scores.items():
                all_metrics.setdefault(k, []).append(v.score)
        metric_avgs = {k: sum(v)/len(v) for k, v in all_metrics.items()}

        data = {
            "run_id": self.run_id,
            "tag": self.tag,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total": len(results),
            "passed": passed,
            "pass_rate": passed / len(results) if results else 0,
            "avg_score": avg_score,
            "avg_latency": avg_latency,
            "metric_avgs": metric_avgs,
            "results": [r.to_dict() for r in results],
        }
        path = os.path.join(RESULTS_DIR, f"{self.run_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # 打印汇总
        table = Table(title=f"评测汇总： {self.run_id}", show_lines=True)
        table.add_column("指标", style="cyan")
        table.add_column("值", style="green")
        table.add_row("总用例数", str(len(results)))
        table.add_row("通过数", f"{passed}/{len(results)}")
        table.add_row("通过率", f"{passed/len(results)*100:.1f}%" if results else "0%")
        table.add_row("平均综合分", f"{avg_score:.2f}/5")
        table.add_row("平均响应时间", f"{avg_latency:.1f}s")
        for metric, avg in metric_avgs.items():
            table.add_row(f" └ {metric}", f"{avg:.2f}/5")
        console.print(table)
        console.print(f"[dim]结果已保存： {path}[/dim]")
        return path


def list_runs():
    """列出历史评测记录"""
    if not os.path.exists(RESULTS_DIR):
        console.print("[yellow]暂无评测记录[/yellow]")
        return
    files = sorted([f for f in os.listdir(RESULTS_DIR) if f.endswith(".json")], reverse=True)
    if not files:
        console.print("[yellow]暂无评测记录[/yellow]")
        return

    table = Table(title="历史评测记录", show_lines=True)
    table.add_column("运行ID", style="cyan")
    table.add_column("时间")
    table.add_column("用例数")
    table.add_column("通过率")
    table.add_column("平均分")
    table.add_column("平均延迟")

    for f in files[:10]:
        path = os.path.join(RESULTS_DIR, f)
        with open(path, "r", encoding="utf-8") as fp:
            d = json.load(fp)
        table.add_row(
            d.get("run_id", f),
            d.get("created_at", ""),
            str(d.get("total", 0)),
            f"{d.get("pass_rate", 0)*100:.1f}%",
            f"{d.get("avg_score", 0):.2f}",
            f"{d.get("avg_latency", 0):.1f}s",
        )
    console.print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="自动化评测运行器")
    sub = parser.add_subparsers(dest="cmd")

    run_p = sub.add_parser("run", help="运行评测")
    run_p.add_argument("--system", choices=["main", "mental", "career"], help="只评测指定系统，默认全部")
    run_p.add_argument("--tag", default="", help="版本标签，如 v1.0 或 prompt_v2")
    run_p.add_argument("--difficulty", choices=["easy", "medium", "hard"], help="只运行指定难度")
    run_p.add_argument("--dataset", default="default", help="测试集名称")

    sub.add_parser("list", help="列出历史评测")

    args = parser.parse_args()
    if args.cmd == "run":
        ds = EvalDataset(args.dataset).load()
        cases = ds.filter(
            system=args.system,
            difficulty=args.difficulty,
        )
        if not cases:
            console.print("[red]没有符合条件的测试用例[/red]")
        else:
            runner = EvalRunner(tag=args.tag)
            results = runner.run(cases)
            runner.save(results, cases)
    elif args.cmd == "list":
        list_runs()
    else:
        parser.print_help()