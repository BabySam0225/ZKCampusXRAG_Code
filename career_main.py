"""
career_main.py  —  就业指导应⽤⼊⼝
使⽤流程：
第⼀步（建库）: python build_career_kb.py build --files 就业信息下载 2025 年.xlsx
多年数据：python build_career_kb.py build --files 2023.xlsx 2024.xlsx 2025.xlsx
第⼆步（启动）: python career_main.py chat
"""
import sys
import argparse
from rich.console import Console

console = Console()

def main():
    parser = argparse.ArgumentParser(description="就业指导助⼿")
    subparsers = parser.add_subparsers(dest="command")
    
    subparsers.add_parser("chat", help="启动就业指导对话")
    
    query_p = subparsers.add_parser("query", help="单次提问")
    query_p.add_argument("--q", required=True, help="问题内容")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    from career_pipeline import CareerPipeline
    pipeline = CareerPipeline()
    pipeline.load_index()
    
    if args.command == "chat":
        pipeline.chat()
    elif args.command == "query":
        pipeline.query(args.q)

if __name__ == "__main__":
    main()