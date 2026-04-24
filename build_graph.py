"""
build_graph.py — 知识图谱建库脚本
====================================
从已有的 docs.json（向量知识库）中抽取实体关系，构建知识图谱。
必须先完成对应系统的知识库建库才能运行此脚本。

用法：
  python build_graph.py --system main          # 为辅导员知识库建图谱
  python build_graph.py --system mental
  python build_graph.py --system career
  python build_graph.py --system main --sample 100   # 只抽取前 100 个 chunk（测试用）
  python build_graph.py --system main --stats        # 查看已有图谱统计
"""

import os
import json
import argparse
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

from graph_extractor import GraphExtractor
from graph_store import KnowledgeGraph

console = Console()

# 各系统的 docs.json 和图谱存储路径
SYSTEM_CONFIG = {
    "main":   {"docs": "./storage/docs.json",         "graph": "./storage/graph"},
    "mental": {"docs": "./mental_storage/docs.json",  "graph": "./mental_storage/graph"},
    "career": {"docs": "./career_storage/docs.json",  "graph": "./career_storage/graph"},
}


def load_chunks(docs_path: str) -> list:
    """加载 docs.json，返回 chunk 列表"""
    if not os.path.exists(docs_path):
        raise FileNotFoundError(
            f"知识库不存在：{docs_path}\n请先运行对应的 build_kb.py 建库"
        )
    with open(docs_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("docs", data)


def build_graph(system: str, sample: int = None, interval: float = 1.0):
    cfg = SYSTEM_CONFIG.get(system)
    if not cfg:
        console.print(f"[red]未知系统：{system}[/red]")
        return

    console.print(Panel(
        f"[bold green]🕸 构建知识图谱[/bold green]\n"
        f"系统：{system}\n"
        f"来源：{cfg['docs']}",
        border_style="green"
    ))

    # 加载 chunks
    console.print("\n[cyan]Step 1/3  加载文档块[/cyan]")
    chunks = load_chunks(cfg["docs"])
    if sample:
        chunks = chunks[:sample]
        console.print(f"  [dim]采样模式：只处理前 {sample} 个 chunk[/dim]")
    console.print(f"  共 {len(chunks)} 个 chunk 待处理")

    # 过滤太短的 chunk（内容不足以抽取实体）
    chunks = [c for c in chunks if len(c.get("content", "")) >= 50]
    console.print(f"  有效 chunk（内容≥50字）：{len(chunks)} 个")

    # 抽取实体和关系
    console.print(f"\n[cyan]Step 2/3  实体与关系抽取（间隔 {interval}s）[/cyan]")
    console.print(f"  [dim]预计耗时：约 {len(chunks) * interval / 60:.1f} 分钟[/dim]")
    extractor = GraphExtractor()
    results = extractor.batch_extract(chunks, interval=interval)

    # 统计抽取结果
    total_entities = sum(len(r.get("entities", [])) for r in results)
    total_relations = sum(len(r.get("relations", [])) for r in results)
    console.print(
        f"  抽取完成：{total_entities} 个实体，{total_relations} 条关系"
    )

    # 构建图谱
    console.print("\n[cyan]Step 3/3  构建并保存图谱[/cyan]")
    kg = KnowledgeGraph(storage_dir=cfg["graph"])
    kg.build(results)
    kg.save()
    kg.stats()

    console.print(Panel(
        f"[bold green]✅ 知识图谱构建完成！[/bold green]\n\n"
        f"  图谱目录：{cfg['graph']}/\n"
        f"  下一步：重新运行问答，即可享受图谱增强检索",
        border_style="green"
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="知识图谱建库")
    parser.add_argument("--system", required=True,
                        choices=["main", "mental", "career"])
    parser.add_argument("--sample", type=int, default=None,
                        help="只处理前 N 个 chunk（测试用）")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="抽取请求间隔秒数，默认 1.0")
    parser.add_argument("--stats", action="store_true",
                        help="查看已有图谱统计，不重新构建")
    args = parser.parse_args()

    if args.stats:
        cfg = SYSTEM_CONFIG[args.system]
        kg = KnowledgeGraph(storage_dir=cfg["graph"])
        if kg.load():
            kg.stats()
        else:
            console.print("[yellow]图谱不存在，请先运行建库[/yellow]")
    else:
        build_graph(args.system, sample=args.sample, interval=args.interval)
