import sys
import argparse

from rich.console import Console
from rich.panel import Panel

console = Console()


def main():
    parser = argparse.ArgumentParser(description="⼼理辅导助⼿")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("chat", help="启动对话")

    query_p = subparsers.add_parser("query", help="单次提问")
    query_p.add_argument("--q", required=True, help="问题内容")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    from mental_pipeline import MentalPipeline
    pipeline = MentalPipeline()
    pipeline.load_index()

    if args.command == "chat":
        pipeline.chat()
    elif args.command == "query":
        pipeline.query(args.q)


if __name__ == "__main__":
    main()