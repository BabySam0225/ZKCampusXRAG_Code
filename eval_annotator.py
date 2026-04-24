"""
eval_annotator.py — 人工标注 CLI
==================================
用于：
1. 给已有回答进行人工打分
2. 直接添加新的测试用例
3. 查看和修正已有标注

用法：
python eval_annotator.py annotate --run latest  # 标注最新一次评测结果
python eval_annotator.py add                    # 手动添加测试用例
python eval_annotator.py review                 # 复查已标注结果
"""

import os
import json
import argparse
from datetime import datetime
from typing import List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, IntPrompt, Confirm

from eval_dataset import EvalDataset, EvalCase, EVAL_DIR

console = Console()
ANNOTATION_FILE = os.path.join(EVAL_DIR, "human_annotations.json")


def load_annotations() -> dict:
    if os.path.exists(ANNOTATION_FILE):
        with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_annotations(annotations: dict):
    os.makedirs(EVAL_DIR, exist_ok=True)
    with open(ANNOTATION_FILE, "w", encoding="utf-8") as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)


def annotate_run(run_file: str = None):
    """对一次评测结果做人工标注"""
    # 找到评测结果文件
    if not run_file or run_file == "latest":
        results_dir = os.path.join(EVAL_DIR, "results")
        if not os.path.exists(results_dir):
            console.print("[red] 没有找到评测结果，请先运行 eval_runner.py[/red]")
            return
        files = sorted([f for f in os.listdir(results_dir) if f.endswith(".json")])
        if not files:
            console.print("[red] 结果目录为空[/red]")
            return
        run_file = os.path.join(results_dir, files[-1])

    with open(run_file, "r", encoding="utf-8") as f:
        run_data = json.load(f)

    results = run_data.get("results", [])
    annotations = load_annotations()
    run_id = run_data.get("run_id", "unknown")

    console.print(Panel(
        f"[bold green] 人工标注模式[/bold green]\n"
        f"评测运行： {run_id}\n"
        f"共 {len(results)} 条结果\n\n"
        f"操作说明：\n"
        f" 输入评分 1-5（1=很差 3=及格 5=很好）\n"
        f" 输入 s 跳过，输入 q 退出",
        border_style="green"
    ))

    annotated = 0
    for i, result in enumerate(results):
        case_id = result.get("case_id", "")
        if case_id in annotations.get(run_id, {}):
            console.print(f"[dim]跳过已标注： {case_id}[/dim]")
            continue

        console.print(f"\n[cyan]── 第 {i+1}/{len(results)} 条 [{case_id}] ──[/cyan]")
        console.print(Panel(f"[bold]问题：[/bold]{result['question']}", border_style="blue"))
        console.print(Panel(f"[bold]AI 回答：[/bold]\n{result['answer']}", border_style="yellow"))
        console.print(Panel(f"[bold]参考答案：[/bold]\n{result['reference']}", border_style="green"))

        # 显示自动评分
        scores = result.get("scores", {})
        if scores:
            console.print("[dim]自动评分： " +
                " ".join(f"{k}:{v['score']:.1f}" for k, v in scores.items()) + "[/dim]")

        user_input = Prompt.ask(
            "\n[cyan]你的综合评分 (1-5, s=跳过, q=退出)[/cyan]", default="s"
        )
        if user_input.lower() == "q":
            break
        if user_input.lower() == "s":
            continue

        try:
            human_score = int(user_input)
            if not 1 <= human_score <= 5:
                raise ValueError
        except ValueError:
            console.print("[yellow]无效输入，跳过[/yellow]")
            continue

        comment = Prompt.ask("备注（可选，直接回车跳过）", default="")
        # 保存标注
        if run_id not in annotations:
            annotations[run_id] = {}
        annotations[run_id][case_id] = {
            "human_score": human_score,
            "auto_score": result.get("overall", 0),
            "comment": comment,
            "annotated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        save_annotations(annotations)
        annotated += 1
        console.print(f"[green]✓ 已标注（{annotated} 条）[/green]")

    console.print(f"\n[bold green]标注完成，共标注 {annotated} 条[/bold green]")
    _show_annotation_agreement(annotations.get(run_id, {}))


def _show_annotation_agreement(run_annotations: dict):
    """展示人工评分与自动评分的一致性分析"""
    if not run_annotations:
        return
    pairs = [(v["human_score"], v["auto_score"]) for v in run_annotations.values() if v.get("auto_score")]
    if not pairs:
        return

    human_scores = [p[0] for p in pairs]
    auto_scores = [p[1] for p in pairs]
    avg_human = sum(human_scores) / len(human_scores)
    avg_auto = sum(auto_scores) / len(auto_scores)
    diff = [abs(h - a) for h, a in pairs]
    agreement = sum(1 for d in diff if d <= 1) / len(diff) * 100

    console.print(f"\n[bold cyan] 人机一致性分析[/bold cyan]")
    console.print(f" 人工平均分： {avg_human:.2f}  自动平均分： {avg_auto:.2f}")
    console.print(f" 评分差距在1分以内的比例： {agreement:.1f}%")


def add_case_interactive():
    """交互式添加测试用例"""
    console.print(Panel("[bold green] 添加测试用例[/bold green]", border_style="green"))
    ds = EvalDataset("default").load()

    question = Prompt.ask("问题")
    reference = Prompt.ask("参考答案（金标准）")
    system = Prompt.ask("系统", choices=["main", "mental", "career"], default="main")
    difficulty = Prompt.ask("难度", choices=["easy", "medium", "hard"], default="medium")
    tags_input = Prompt.ask("标签（逗号分隔，如 奖学金,申请）", default="")
    tags = [t.strip() for t in tags_input.split(",") if t.strip()]
    notes = Prompt.ask("备注（可选）", default="")

    case = EvalCase(
        question=question,
        reference=reference,
        system=system,
        difficulty=difficulty,
        tags=tags,
        notes=notes,
        source="manual",
    )

    console.print(Panel(
        f"问题： {case.question}\n"
        f"参考： {case.reference}\n"
        f"系统： {case.system}  难度： {case.difficulty}  标签： {', '.join(case.tags)}",
        title="确认添加",
        border_style="yellow"
    ))

    if Confirm.ask("确认添加？"):
        ds.add(case)
        ds.save()
        console.print(f"[green]✓ 已添加，ID： {case.case_id}[/green]")


def review_annotations():
    """复查已有标注"""
    annotations = load_annotations()
    if not annotations:
        console.print("[yellow]暂无标注记录[/yellow]")
        return

    table = Table(title="标注记录", show_lines=True)
    table.add_column("运行ID", width=20)
    table.add_column("标注数", width=8)
    table.add_column("平均人工分", width=10)
    table.add_column("平均自动分", width=10)

    for run_id, cases in annotations.items():
        human_scores = [v["human_score"] for v in cases.values()]
        auto_scores = [v["auto_score"] for v in cases.values() if v.get("auto_score")]
        avg_h = sum(human_scores) / len(human_scores) if human_scores else 0
        avg_a = sum(auto_scores) / len(auto_scores) if auto_scores else 0
        table.add_row(run_id, str(len(cases)), f"{avg_h:.2f}", f"{avg_a:.2f}")

    console.print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="人工标注工具")
    sub = parser.add_subparsers(dest="cmd")

    ann = sub.add_parser("annotate", help="标注评测结果")
    ann.add_argument("--run", default="latest", help="评测结果文件或 latest")

    sub.add_parser("add", help="添加测试用例")
    sub.add_parser("review", help="复查标注记录")

    args = parser.parse_args()

    if args.cmd == "annotate":
        annotate_run(args.run)
    elif args.cmd == "add":
        add_case_interactive()
    elif args.cmd == "review":
        review_annotations()
    else:
        parser.print_help()