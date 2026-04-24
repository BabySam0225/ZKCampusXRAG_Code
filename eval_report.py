"""
eval_report.py — 错误分析 + 回归测试
======================================
功能：
1. 错误分析：找出低分用例，按错误类型聚类
2. 回归测试：对比最新结果与 baseline，确保没有退步
3. 生成 HTML 评测报告

用法：
python eval_report.py error --run latest    # 错误分析
python eval_report.py regression --baseline v1  # 回归测试
python eval_report.py html --run latest     # 生成 HTML 报告
"""

import os
import json
import argparse
from collections import Counter, defaultdict
from typing import List, Dict
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from eval_dataset import EVAL_DIR
from eval_metrics import EvalResult

console = Console()
RESULTS_DIR = os.path.join(EVAL_DIR, "results")


def load_latest_run() -> tuple:
    files = sorted([f for f in os.listdir(RESULTS_DIR) if f.endswith(".json")], reverse=True)
    if not files:
        raise FileNotFoundError("没有找到评测结果")
    with open(os.path.join(RESULTS_DIR, files[0]), "r", encoding="utf-8") as f:
        data = json.load(f)
    return data, [EvalResult.from_dict(r) for r in data.get("results", [])]


def load_run_by_tag(tag: str) -> tuple:
    files = sorted([f for f in os.listdir(RESULTS_DIR) if f.endswith(".json")])
    for f in files:
        with open(os.path.join(RESULTS_DIR, f), "r", encoding="utf-8") as fp:
            data = json.load(fp)
        if tag in data.get("run_id", "") or tag in data.get("tag", ""):
            return data, [EvalResult.from_dict(r) for r in data.get("results", [])]
    raise FileNotFoundError(f"未找到： {tag}")


# ── 错误类型分类器 ────────────────────────────
ERROR_PATTERNS = {
    "无法回答": ["未找到", "未在知识库", "无相关信息", "不确定", "无法回答"],
    "回答偏题": ["但是", "另外", "此外"],  # 偏离主题的标志词（粗略）
    "幻觉错误": [],       # 由 faithfulness < 2 判断
    "共情不足": [],       # 由 empathy < 2 判断
    "数据错误": [],       # 由 data_accuracy < 2 判断
    "安全风险": [],       # 由 safety < 2 判断
}


def classify_error(result: EvalResult) -> List[str]:
    """判断一条失败结果的错误类型"""
    errors = []
    answer_lower = result.answer.lower()

    # 基于回答内容
    for err_type, keywords in ERROR_PATTERNS.items():
        if keywords and any(kw in answer_lower for kw in keywords):
            errors.append(err_type)

    # 基于各指标分数
    for metric, score_obj in result.scores.items():
        if score_obj.score < 2:
            if metric == "faithfulness":
                errors.append("幻觉错误")
            elif metric == "empathy":
                errors.append("共情不足")
            elif metric == "data_accuracy":
                errors.append("数据错误")
            elif metric == "safety":
                errors.append("安全风险")
            elif metric == "relevance":
                errors.append("回答偏题")

    if not errors:
        errors.append("综合质量差")
    return list(set(errors))


def error_analysis(run_tag: str = "latest"):
    """错误分析：找出低分用例并聚类"""
    try:
        if run_tag == "latest":
            meta, results = load_latest_run()
        else:
            meta, results = load_run_by_tag(run_tag)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        return

    failed = [r for r in results if not r.passed or r.overall < 3.5]
    console.print(Panel(
        f"[bold red] 错误分析： {meta['run_id']}[/bold red]\n"
        f"总用例： {len(results)}  失败/低分： {len(failed)} "
        f"（占比 {len(failed)/len(results)*100:.1f}%）",
        border_style="red"
    ))

    if not failed:
        console.print("[green] 没有失败或低分用例！[/green]")
        return

    # 错误类型统计
    error_counter = Counter()
    error_cases = defaultdict(list)
    for r in failed:
        errors = classify_error(r)
        for e in errors:
            error_counter[e] += 1
            error_cases[e].append(r)

    et = Table(title="错误类型分布", show_lines=True)
    et.add_column("错误类型", style="red")
    et.add_column("数量", style="yellow")
    et.add_column("占失败比例")
    for err, cnt in error_counter.most_common():
        et.add_row(err, str(cnt), f"{cnt/len(failed)*100:.1f}%")
    console.print(et)

    # 按系统分组
    sys_fails = Counter(r.system for r in failed)
    console.print("\n[bold]各系统失败分布：[/bold] " +
                 " ".join(f"{k}:{v}" for k, v in sys_fails.items()))

    # 低分详情（最差的10条）
    worst = sorted(failed, key=lambda r: r.overall)[:10]
    wt = Table(title="最低分用例 Top 10", show_lines=True)
    wt.add_column("ID", width=8)
    wt.add_column("系统", width=6)
    wt.add_column("综合分", width=6)
    wt.add_column("问题", width=35)
    wt.add_column("错误类型")
    wt.add_column("最低指标")

    for r in worst:
        errors = classify_error(r)
        worst_metric = min(r.scores.items(), key=lambda x: x[1].score) if r.scores else ("none", 0)
        worst_str = f"{worst_metric[0]}:{worst_metric[1].score:.1f}"
        wt.add_row(r.case_id, r.system, f"{r.overall:.1f}",
                   r.question[:33], ", ".join(errors), worst_str)
    console.print(wt)

    # 改进建议
    console.print("\n[bold cyan] 改进建议[/bold cyan]")
    for err_type, cases in sorted(error_cases.items(), key=lambda x: -len(x[1]))[:3]:
        console.print(f"\n [{err_type}]（{len(cases)}条）")
        if err_type == "幻觉错误":
            console.print("→ 检查知识库内容是否覆盖这些问题，或加强 faithfulness 约束")
        elif err_type == "无法回答":
            console.print("→ 这些问题知识库可能没有相关内容，考虑补充知识库文档")
        elif err_type == "共情不足":
            console.print("→ 调整 mental system prompt，加强共情和情感认可指令")
        elif err_type == "数据错误":
            console.print("→ 检查就业 Excel 数据是否准确，或 chunk 生成是否正确")
        elif err_type == "安全风险":
            console.print("→ 立即检查相关用例，考虑加强危机检测逻辑")


def regression_test(baseline_tag: str, current_tag: str = "latest"):
    """回归测试：确保新版本不比 baseline 差"""
    try:
        meta_base, results_base = load_run_by_tag(baseline_tag)
        if current_tag == "latest":
            meta_curr, results_curr = load_latest_run()
        else:
            meta_curr, results_curr = load_run_by_tag(current_tag)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        return

    map_base = {r.case_id: r for r in results_base}
    map_curr = {r.case_id: r for r in results_curr}
    common = set(map_base) & set(map_curr)

    regressions = []
    for cid in common:
        base_r = map_base[cid]
        curr_r = map_curr[cid]
        if curr_r.overall < base_r.overall - 0.5:  # 退步超过 0.5 分
            regressions.append((cid, base_r, curr_r))

        # 安全指标不能退步
        base_safety = base_r.scores.get("safety")
        curr_safety = curr_r.scores.get("safety")
        if base_safety and curr_safety and curr_safety.score < base_safety.score:
            if not any(c == cid for c, _, _ in regressions):
                regressions.append((cid, base_r, curr_r))

    overall_delta = meta_curr["avg_score"] - meta_base["avg_score"]
    passed = len(regressions) == 0 and overall_delta >= -0.2
    status = "[bold green] 回归测试通过[/bold green]" if passed else "[bold red] 回归测试失败[/bold red]"

    console.print(Panel(
        f"{status}\n\n"
        f"Baseline： {meta_base['run_id']}  平均分 {meta_base['avg_score']:.2f}\n"
        f"当前版本： {meta_curr['run_id']}  平均分 {meta_curr['avg_score']:.2f}\n"
        f"分数变化： {overall_delta:+.2f}\n"
        f"退步用例： {len(regressions)} 条（共 {len(common)} 条对比）",
        border_style="green" if passed else "red"
    ))

    if regressions:
        rt = Table(title="退步用例", show_lines=True)
        rt.add_column("ID", width=8)
        rt.add_column("问题", width=40)
        rt.add_column("Baseline", width=8)
        rt.add_column("当前", width=8)
        rt.add_column("退步")
        for cid, base_r, curr_r in regressions:
            diff = curr_r.overall - base_r.overall
            rt.add_row(cid, base_r.question[:38],
                       f"{base_r.overall:.1f}", f"{curr_r.overall:.1f}", f"[red]{diff:.1f}[/red]")
        console.print(rt)
    return passed


def generate_html_report(run_tag: str = "latest"):
    """生成 HTML 评测报告"""
    try:
        if run_tag == "latest":
            meta, results = load_latest_run()
        else:
            meta, results = load_run_by_tag(run_tag)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        return

    # 构建指标行
    metric_rows = ""
    for metric, avg in meta.get("metric_avgs", {}).items():
        color = "#27ae60" if avg >= 4 else "#f39c12" if avg >= 3 else "#e74c3c"
        bar = int(avg / 5 * 100)
        metric_rows += f"""
<tr>
<td>{metric}</td>
<td>
<div style="background:#eee;border-radius:4px;height:16px;width:200px">
<div style="background:{color};width:{bar}%;height:100%;border-radius:4px"></div>
</div>
<span style="margin-left:8px;font-weight:bold">{avg:.2f}/5</span>
</td>
</tr>"""

    # 构建用例行
    case_rows = ""
    for r in sorted(results, key=lambda x: x.overall):
        status = "✅" if r.passed else "❌"
        color = "#d5f5e3" if r.passed else "#fadbd8"
        scores_str = " ".join(f"{k}:{v.score:.1f}" for k, v in r.scores.items())
        case_rows += f"""
<tr style="background:{color}">
<td>{status} {r.case_id}</td>
<td>[{r.system}]</td>
<td>{r.question[:50]}</td>
<td><b>{r.overall:.1f}</b></td>
<td style="font-size:12px">{scores_str}</td>
<td style="font-size:12px">{r.latency:.1f}s</td>
</tr>"""

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>RAG 评测报告 - {meta['run_id']}</title>
<style>
body{{font-family:Arial,sans-serif;max-width:1200px;margin:0 auto;padding:20px;background:#f5f7fa}}
h1{{color:#2c3e50;border-bottom:3px solid #3498db;padding-bottom:10px}}
h2{{color:#34495e;margin-top:30px}}
.summary{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:20px 0}}
.card{{background:white;border-radius:8px;padding:16px;text-align:center;box-shadow:0 2px 4px rgba(0,0,0,0.1)}}
.card .value{{font-size:28px;font-weight:bold;color:#3498db}}
.card .label{{color:#7f8c8d;font-size:13px;margin-top:4px}}
table{{width:100%;border-collapse:collapse;background:white;border-radius:8px;overflow:hidden;box-shadow:0 2px 4px rgba(0,0,0,0.1)}}
th{{background:#3498db;color:white;padding:10px;text-align:left;font-size:13px}}
td{{padding:8px 10px;border-bottom:1px solid #ecf0f1;font-size:13px}}
.pass-rate{{font-size:16px;color:{'#27ae60' if meta['pass_rate']>=0.7 else '#e74c3c'}}}
</style>
</head>
<body>
<h1> RAG 系统评测报告</h1>
<p style="color:#7f8c8d">运行ID： {meta['run_id']} | 生成时间： {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

<div class="summary">
<div class="card"><div class="value">{meta['total']}</div><div class="label">测试用例数</div></div>
<div class="card"><div class="value pass-rate">{meta['pass_rate']*100:.1f}%</div><div class="label">通过率</div></div>
<div class="card"><div class="value">{meta['avg_score']:.2f}</div><div class="label">平均分</div></div>
<div class="card"><div class="value">{meta.get('avg_latency',0):.1f}s</div><div class="label">平均响应</div></div>
</div>

<h2>各指标评分</h2>
<table><tr><th>指标</th><th>平均分</th></tr>{metric_rows}</table>

<h2>用例详情（按分数升序）</h2>
<table>
<tr><th>状态/ID</th><th>系统</th><th>问题</th><th>综合分</th><th>各指标</th><th>延迟</th></tr>
{case_rows}
</table>
</body>
</html>"""

    report_path = os.path.join(EVAL_DIR, f"report_{meta['run_id']}.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    console.print(f"[green] HTML 报告已生成： {report_path}[/green]")
    return report_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="错误分析 + 回归测试")
    sub = parser.add_subparsers(dest="cmd")

    err_p = sub.add_parser("error", help="错误分析")
    err_p.add_argument("--run", default="latest", help="run_id 或 tag，默认最新")

    reg_p = sub.add_parser("regression", help="回归测试")
    reg_p.add_argument("--baseline", required=True, help="baseline 的 tag 或 run_id")
    reg_p.add_argument("--current", default="latest", help="当前版本，默认最新")

    html_p = sub.add_parser("html", help="生成 HTML 报告")
    html_p.add_argument("--run", default="latest")

    args = parser.parse_args()
    if args.cmd == "error":
        error_analysis(args.run)
    elif args.cmd == "regression":
        regression_test(args.baseline, args.current)
    elif args.cmd == "html":
        generate_html_report(args.run)
    else:
        parser.print_help()