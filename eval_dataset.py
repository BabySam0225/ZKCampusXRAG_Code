"""
eval_dataset.py — 测试集管理
=============================
负责：创建、加载、版本控制、过滤测试用例

测试用例结构：
question    用户问题
reference   参考答案（人工标注的标准答案）
tags        标签列表，如 ["薪酬","网络专业","困难"]
difficulty  难度：easy / medium / hard
system      所属系统：main / mental / career
source      来源：manual（人工） / auto（自动生成）
"""

import os
import json
import uuid
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()
EVAL_DIR = "./eval_data"


@dataclass
class EvalCase:
    question: str
    reference: str                  # 参考答案（金标准）
    system: str = "main"            # main / mental / career
    tags: List[str] = field(default_factory=list)
    difficulty: str = "medium"      # easy / medium / hard
    source: str = "manual"          # manual / auto
    case_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M"))
    notes: str = ""                 # 备注

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "EvalCase":
        return EvalCase(**{k: v for k, v in d.items() if k in EvalCase.__dataclass_fields__})


class EvalDataset:
    """测试集管理器，支持版本控制和按标签/系统过滤"""
    def __init__(self, name: str = "default"):
        self.name = name
        self.cases: List[EvalCase] = []
        os.makedirs(EVAL_DIR, exist_ok=True)
        self.path = os.path.join(EVAL_DIR, f"{name}.json")

    # ── CRUD ─────────────────────────────────
    def add(self, case: EvalCase):
        self.cases.append(case)

    def remove(self, case_id: str):
        self.cases = [c for c in self.cases if c.case_id != case_id]

    def get(self, case_id: str) -> Optional[EvalCase]:
        return next((c for c in self.cases if c.case_id == case_id), None)

    # ── 过滤 ─────────────────────────────────
    def filter(self, system: str = None, tags: List[str] = None, difficulty: str = None) -> List[EvalCase]:
        result = self.cases
        if system:
            result = [c for c in result if c.system == system]
        if tags:
            result = [c for c in result if any(t in c.tags for t in tags)]
        if difficulty:
            result = [c for c in result if c.difficulty == difficulty]
        return result

    # ── 持久化 ────────────────────────────────
    def save(self):
        data = {
            "name": self.name,
            "version": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "total": len(self.cases),
            "cases": [c.to_dict() for c in self.cases],
        }
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        console.print(f"[green]✓ 测试集已保存： {self.path}（{len(self.cases)} 条）[/green]")

    def load(self) -> "EvalDataset":
        if not os.path.exists(self.path):
            console.print(f"[yellow]⚠ 测试集文件不存在：{self.path}，已创建空集[/yellow]")
            return self
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.cases = [EvalCase.from_dict(c) for c in data.get("cases", [])]
        console.print(f"[green]✓ 加载测试集：{self.name}（{len(self.cases)} 条）[/green]")
        return self

    # ── 版本备份 ──────────────────────────────
    def backup(self):
        """保存带时间戳的备份，用于版本对比"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(EVAL_DIR, f"{self.name}_{ts}.json")
        with open(self.path, "r") as f:
            content = f.read()
        with open(backup_path, "w") as f:
            f.write(content)
        console.print(f"[dim]备份至： {backup_path}[/dim]")
        return backup_path

    # ── 统计 ──────────────────────────────────
    def stats(self):
        from collections import Counter
        table = Table(title=f"测试集统计：{self.name}（共 {len(self.cases)} 条）", show_lines=True)
        table.add_column("维度", style="cyan")
        table.add_column("分布")

        sys_cnt = Counter(c.system for c in self.cases)
        diff_cnt = Counter(c.difficulty for c in self.cases)
        src_cnt = Counter(c.source for c in self.cases)
        all_tags = [t for c in self.cases for t in c.tags]
        tag_cnt = Counter(all_tags).most_common(8)

        table.add_row("系统分布", " ".join(f"{k}:{v}" for k, v in sys_cnt.items()))
        table.add_row("难度分布", " ".join(f"{k}:{v}" for k, v in diff_cnt.items()))
        table.add_row("来源分布", " ".join(f"{k}:{v}" for k, v in src_cnt.items()))
        table.add_row("热门标签", " ".join(f"{k}({v})" for k, v in tag_cnt))
        console.print(table)


# ── 预置测试用例（辅导员系统） ────────────────
BUILTIN_CASES = [
    # 辅导员知识库
    EvalCase("国家奖学金的申请条件是什么？", "参考手册中国家奖学金相关章节", "main", ["奖学金", "申请"], "easy"),
    EvalCase("学生违反校规会受到哪些处分？", "参考手册学生处分相关条例", "main", ["处分", "校规"], "medium"),
    EvalCase("转专业需要满足什么条件？", "参考手册转专业政策", "main", ["转专业", "政策"], "medium"),
    EvalCase("我想休学一年，流程是什么？", "参考手册休学相关规定", "main", ["休学", "流程"], "medium"),

    # 心理辅导
    EvalCase("我最近压力很大，总是睡不着，怎么办？",
             "应先共情，再给出放松技巧建议，如深呼吸、正念冥想等", "mental", ["压力", "睡眠"], "medium"),
    EvalCase("我感觉自己什么都做不好，很沮丧",
             "应认可情绪，避免否定感受，给予鼓励和具体建议", "mental", ["情绪", "自我怀疑"], "hard"),

    # 就业指导
    EvalCase("网络专业毕业生一般去哪些公司？",
             "应引用实际就业数据，列举热门公司和录用人数", "career", ["网络专业", "就业去向"], "easy"),
    EvalCase("软件开发岗位需要掌握哪些技能？",
             "应列举具体技术栈：编程语言、数据库、框架、工具等", "career", ["软件开发", "技能"], "medium"),
    EvalCase("我们学校升学率怎么样，值得考研吗？",
             "应给出升学率数据，客观分析升学vs就业的利弊", "career", ["升学", "考研"], "hard"),
]


def create_default_dataset() -> EvalDataset:
    """创建包含内置测试用例的默认测试集"""
    ds = EvalDataset("default")
    for case in BUILTIN_CASES:
        ds.add(case)
    ds.save()
    return ds


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="测试集管理")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("init", help="创建默认测试集")
    sub.add_parser("stats", help="查看统计")

    lst = sub.add_parser("list", help="列出所有用例")
    lst.add_argument("--system", help="过滤系统")

    args = parser.parse_args()

    if args.cmd == "init":
        ds = create_default_dataset()
        ds.stats()
    elif args.cmd == "stats":
        EvalDataset("default").load().stats()
    elif args.cmd == "list":
        ds = EvalDataset("default").load()
        cases = ds.filter(system=args.system) if args.system else ds.cases
        t = Table(show_lines=True)
        t.add_column("ID", width=8)
        t.add_column("系统", width=8)
        t.add_column("难度", width=6)
        t.add_column("问题", width=50)
        t.add_column("标签")
        for c in cases:
            t.add_row(c.case_id, c.system, c.difficulty, c.question, ", ".join(c.tags))
        console.print(t)
    else:
        parser.print_help()