"""
eval_compare.py — Prompt/Version 对比分析
==========================================
对比两次评测结果，找出改进和退步的用例

用法：
python eval_compare.py --a v1.0 --b v2.0
python eval_compare.py --a 20250101_120000 --b 20250102_150000
python eval_compare.py --list-runs  # 列出可对比的运行
"""

import os
import json
import argparse
from typing import List, Dict, Tuple
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from eval_dataset import EVAL_DIR
from eval_metrics import EvalResult

console = Console()
RESULTS_DIR = os.path.join(EVAL_DIR, "results")


def load_run(tag_or_id: str) -> Tuple[dict, List[EvalResult]]:
    """按 tag 或 run_id 加载评测结果"""
    if not os.path.exists(RESULTS_DIR):
        raise FileNotFoundError("没有找到评测结果目录")

    files = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".json")]
    matched = None
    for f in sorted(files):
        with open(os.path.join(RESULTS_DIR, f), "r", encoding="utf-8") as fp:
            data = json.load(fp)
        if tag_or_id in data.get("run_id", "") or tag_or_id in data.get("tag", ""):
            matched = data
            break

    if not matched:
        raise FileNotFoundError(f"未找到匹配的评测结果： {tag_or_id}")

    results = [EvalResult.from_dict(r) for r in matched.get("results", [])]
    return matched, results


def compare_runs(tag_a: str, tag_b: str):
    """对比两次评测结果"""
    try:
        meta_a, results_a = load_run(tag_a)
        meta_b, results_b = load_run(tag_b)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        return

    # 建立 case_id → result 的映射
    map_a = {r.case_id: r for r in results_a}
    map_b = {r.case_id: r for r in results_b}
    common_ids = set(map_a.keys()) & set(map_b.keys())

    if not common_ids:
        console.print("[yellow]两次评测没有共同的测试用例，无法对比[/yellow]")
        return

    # 总体对比
    console.print(Panel(
        f"[bold cyan]对比： {meta_a['run_id']} vs {meta_b['run_id']}[/bold cyan]\n"
        f"{'版本A':12} 通过率:{meta_a['pass_rate']*100:.1f}%  平均分:{meta_a['avg_score']:.2f}  延迟:{meta_a.get('avg_latency',0):.1f}s\n"
        f"{'版本B':12} 通过率:{meta_b['pass_rate']*100:.1f}%  平均分:{meta_b['avg_score']:.2f}  延迟:{meta_b.get('avg_latency',0):.1f}s",
        border_style="cyan"
    ))

    # 分指标对比
    metrics_a = meta_a.get("metric_avgs", {})
    metrics_b = meta_b.get("metric_avgs", {})
    all_metrics = set(list(metrics_a.keys()) + list(metrics_b.keys()))

    mt = Table(title="各指标对比", show_lines=True)
    mt.add_column("指标", style="cyan")
    mt.add_column(f"版本A ({tag_a})", style="yellow")
    mt.add_column(f"版本B ({tag_b})", style="green")
    mt.add_column("变化", style="bold")

    for metric in sorted(all_metrics):
        a_val = metrics_a.get(metric, 0)
        b_val = metrics_b.get(metric, 0)
        diff = b_val - a_val
        if diff > 0.1:
            change = f"[green]▲ +{diff:.2f}[/green]"
        elif diff < -0.1:
            change = f"[red]▼ {diff:.2f}[/red]"
        else:
            change = f"[dim]≈ {diff:+.2f}[/dim]"
        mt.add_row(metric, f"{a_val:.2f}", f"{b_val:.2f}", change)

    console.print(mt)

    # 逐条对比
    improved, regressed, stable = [], [], []
    for cid in common_ids:
        ra = map_a[cid]
        rb = map_b[cid]
        diff = rb.overall - ra.overall
        if diff > 0.5:
            improved.append((cid, ra, rb, diff))
        elif diff < -0.5:
            regressed.append((cid, ra, rb, diff))
        else:
            stable.append((cid, ra, rb, diff))

    # 显示改进的用例
    if improved:
        it = Table(title=f" 改进的用例（{len(improved)}条）", show_lines=True)
        it.add_column("ID", width=8)
        it.add_column("问题", width=40)
        it.add_column("A分", width=6)
        it.add_column("B分", width=6)
        it.add_column("提升")
        for cid, ra, rb, diff in sorted(improved, key=lambda x: -x[3])[:10]:
            it.add_row(cid, ra.question[:38], f"{ra.overall:.1f}", f"{rb.overall:.1f}", f"[green]+{diff:.1f}[/green]")
        console.print(it)

    # 显示退步的用例
    if regressed:
        rt = Table(title=f" 退步的用例（{len(regressed)}条）", show_lines=True)
        rt.add_column("ID", width=8)
        rt.add_column("问题", width=40)
        rt.add_column("A分", width=6)
        rt.add_column("B分", width=6)
        rt.add_column("下降")
        for cid, ra, rb, diff in sorted(regressed, key=lambda x: x[3])[:10]:
            rt.add_row(cid, ra.question[:38], f"{ra.overall:.1f}", f"{rb.overall:.1f}", f"[red]{diff:.1f}[/red]")
        console.print(rt)

    console.print(f"\n 改进：{len(improved)}条  退步：{len(regressed)}条  稳定：{len(stable)}条（共 {len(common_ids)} 条对比）")

    # 保存对比报告
    report = {
        "run_a": meta_a["run_id"],
        "run_b": meta_b["run_id"],
        "overall_delta": meta_b["avg_score"] - meta_a["avg_score"],
        "improved_count": len(improved),
        "regressed_count": len(regressed),
        "stable_count": len(stable),
        "improved_cases": [{"id": c, "delta": d, "question": ra.question} for c, ra, _, d in improved],
        "regressed_cases": [{"id": c, "delta": d, "question": ra.question} for c, ra, _, d in regressed],
    }
    report_path = os.path.join(EVAL_DIR, f"compare_{tag_a}_vs_{tag_b}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    console.print(f"[dim]对比报告已保存： {report_path}[/dim]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评测结果对比")
    parser.add_argument("--a", required=False, help="版本A的 tag 或 run_id")
    parser.add_argument("--b", required=False, help="版本B的 tag 或 run_id")
    parser.add_argument("--list-runs", action="store_true", help="列出所有运行")
    args = parser.parse_args()

    if args.list_runs:
        from eval_runner import list_runs
        list_runs()
    elif args.a and args.b:
        compare_runs(args.a, args.b)
    else:
        parser.print_help()