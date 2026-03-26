"""
main.py — RAG 问答系统⼊⼝
===========================
两阶段⼯作流：
【 阶段⼀： 建库（ 只需跑⼀次） 】
python build_kb.py build --dir ./docs

【 阶段⼆： 问答（ 每次使⽤） 】
python main.py chat# 交互式对话
python main.py query --q "问题"# 单次问答
"""

import argparse
import os
import sys

from rich.console import Console
from rich.panel import Panel

from rag_config import DEFAULT_CONFIG
console = Console()

def _check_kb_exists():
    """检查知识库是否已建⽴，  没有则提示⽤户先建库"""
    docs_path = os.path.join(DEFAULT_CONFIG.storage_dir, "docs.json")
    if not os.path.exists(docs_path):
        console.print(Panel(
            "[bold red]![img](tmp/409977877762_docxword_media_image1.png) 知识库不存在！ [/bold red]\n\n"
            "请先运⾏建库脚本： \n"
            " [bold yellow]python build_kb.py build --dir ./docs[/bold yellow]\n "
            "把你的 .docx ⽂件放到 ./docs ⽬录下， 再运⾏上⾯的命令。 ",
            border_style="red"
        ))
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="RAG 问答系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使⽤流程：
第⼀步（ 建库） : python build_kb.py build --dir ./docs
第⼆步（ 问答） : python main.py chat
"""
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── chat： 交互式问答 ───────────────────────
    subparsers.add_parser("chat", help="交互式问答（  从已有知识库加载）  ")

    # ── query： 单次问答 ────────────────────────
    query_parser = subparsers.add_parser("query", help="单次问答")
    query_parser.add_argument("--q", type=str, required=True, help="问题内容")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    # 检查知识库是否存在
    _check_kb_exists()

    # 加载 Pipeline
    from pipeline import RAGPipeline
    rag = RAGPipeline(DEFAULT_CONFIG)
    rag.load_index()

    if args.command == "chat":
        rag.chat()
    elif args.command == "query":
        rag.query(args.q)

if __name__ == "__main__":
    main()