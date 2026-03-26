"""
build_mental_kb.py  —  ⼼理知识库建库脚本
⽀持格式：
.json   →  对话数据集，格式：{"conversation": [{"system":..,"input":..,"output":..}, ...]}
.docx   →  普通 Word ⽂档（复⽤ build_kb.py 的解析器）

JSON 处理策略：
每个 input+output 对话对 → ⼀个 chunk
chunk 内容格式：
【问题】⽤户说的话
【回答】专家的回复
这样检索时既能匹配⽤户的问题，也能把专家回答⼀起带出来

⽤法：
python build_mental_kb.py build --dir ./mental_docs
python build_mental_kb.py build --dir ./mental_docs --show
python build_mental_kb.py verify
"""
import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from mental_config import MENTAL_CoNFIG
from document_processor import DocumentProcessor, Document
from hybrid_search import HybridRetriever

console = Console()

# ─────────────────────────────────────────────
# JSON 对话数据集解析
# ─────────────────────────────────────────────
def load_json_conversations(path: str) -> List[str]:
    """
    解析⼼理对话 JSON ⽂件，返回 chunk ⽂本列表。
    ⾃动兼容三种格式：
    格式 A（对话数组）：
    [{"conversation": [{"system":..,"input":..,"output":..}, ...]}, ...]
    格式 B（单条问答，每⾏⼀条）：
    {"input": "⽤户说的话", "content": "专家回复"}
    {"input": "...", "content": "..."}
    格式 C（格式 B 的数组版本）：
    [{"input": "...", "content": "..."}, ...]
    """
    chunks = []
    
    # 尝试整体 JSON 解析
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 统⼀转成列表
        if isinstance(data, dict):
            data = [data]
        
        for item in data:
            if not isinstance(item, dict):
                continue
            
            # 格式 A：有 conversation 字段
            if "conversation" in item:
                for turn in item["conversation"]:
                    user_input = turn.get("input", "").strip()
                    # output 或 content 字段都接受
                    expert_output = (turn.get("output") or turn.get("content") or "")
                    if user_input and expert_output:
                        chunks.append(f"【问题】{user_input}\n【回答】{expert_output}")
            
            # 格式 B/C：直接是 input + content/output
            elif "input" in item:
                user_input = item.get("input", "").strip()
                expert_output = (item.get("output") or item.get("content") or "")
                if user_input and expert_output:
                    chunks.append(f"【问题】{user_input}\n【回答】{expert_output}")
        
        return chunks
    
    except json.JSONDecodeError:
        pass
    
    # 整体解析失败 → 尝试按⾏解析（JSONL 格式）
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                user_input = item.get("input", "").strip()
                expert_output = (item.get("output") or item.get("content") or "")
                if user_input and expert_output:
                    chunks.append(f"【问题】{user_input}\n【回答】{expert_output}")
            except json.JSONDecodeError:
                continue  # 跳过⽆法解析的⾏
    
    return chunks

# ─────────────────────────────────────────────
# 把 chunk ⽂本列表转成 Document 列表
# ─────────────────────────────────────────────
def json_chunks_to_docs(chunks: List[str], source: str) -> List[Document]:
    """
    JSON 对话对不需要再⼆次分块（每个对话对本身就是⼀个语义完整的单元）
    直接每条变⼀个 Document
    """
    import hashlib
    docs = []
    for i, text in enumerate(chunks):
        doc_id = hashlib.md5(f"{source}_{i}_{text[:50]}".encode()).hexdigest()
        docs.append(Document(
            doc_id=doc_id,
            content=text,
            source=source,
            chunk_index=i,
            metadata={},
        ))
    return docs

# ─────────────────────────────────────────────
# 扫描⽬录，加载所有⽀持格式
# ─────────────────────────────────────────────
def scan_and_load_mental(dir_path: str) -> Tuple[List, List]:
    """
    返回：
    json_file_docs: [(path, [Document, ...]), ...] 来⾃ JSON
    docx_file_data: [(path, paragraphs, images), ...] 来⾃ Word
    """
    json_results = []
    docx_results = []
    
    files = sorted(Path(dir_path).rglob("*"))
    
    for f in files:
        if not f.is_file():
            continue
        
        ext = f.suffix.lower()
        
        if ext == ".json":
            try:
                chunks = load_json_conversations(str(f))
                if chunks:
                    docs = json_chunks_to_docs(chunks, str(f))
                    json_results.append((str(f), docs))
                    console.print(
                        f" [green]✓ JSON[/green] {f.name} "
                        f"([cyan]{len(docs)}[/cyan] 个对话对)"
                    )
                else:
                    console.print(f" [yellow]⚠ 空⽂件，跳过[/yellow] {f.name}")
            except Exception as e:
                console.print(f" [red]✗ 解析失败[/red] {f.name}: {e}")
        
        elif ext == ".docx":
            try:
                from build_kb import load_docx_with_images
                paragraphs, images = load_docx_with_images(str(f))
                char_count = sum(len(p["text"]) for p in paragraphs)
                
                if paragraphs:
                    docx_results.append((str(f), paragraphs, images))
                    console.print(
                        f" [green]✓ Word[/green] {f.name} "
                        f"({char_count} 字符, {len(images)} 张图⽚)"
                    )
            except Exception as e:
                console.print(f" [red]✗ 读取失败[/red] {f.name}: {e}")
        
        elif ext == ".doc":
            console.print(f" [yellow]⚠ .doc 不⽀持，请另存为 .docx[/yellow] {f.name}")
    
    return json_results, docx_results

# ─────────────────────────────────────────────
# 主建库流程
# ─────────────────────────────────────────────
def build_mental_knowledge_base(dir_path: str, show_chunks: bool = False):
    console.print(Panel(
        f"[bold green]\n开始构建⼼理知识库[/bold green]\n⽬录：{dir_path}"
    ))
    
    # Step 1: 扫描
    console.print("\n[bold cyan]Step 1/4 扫描⽂档[/bold cyan]")
    json_results, docx_results = scan_and_load_mental(dir_path)
    
    if not json_results and not docx_results:
        console.print("[red]\n未找到任何 .json 或 .docx ⽂件[/red]")
        return
    
    console.print(
        f"\nJSON ⽂件：[cyan]{len(json_results)}[/cyan] 个，"
        f"Word ⽂件：[cyan]{len(docx_results)}[/cyan] 个"
    )
    
    # Step 2: 整理所有 docs
    console.print("\n[bold cyan]Step 2/4 整理⽂档块[/bold cyan]")
    all_docs: List[Document] = []
    
    # JSON 对话对直接加⼊（不再分块）
    for path, docs in json_results:
        all_docs.extend(docs)
        console.print(f" {Path(path).name} → [cyan]{len(docs)}[/cyan] 个 chunk")
    
    # Word ⽂档正常分块
    if docx_results:
        from build_kb import build_full_text_with_offsets, bind_images_to_chunks
        processor = DocumentProcessor(MENTAL_CoNFIG)
        
        for path, paragraphs, images in docx_results:
            full_text, para_offsets = build_full_text_with_offsets(paragraphs)
            docs = processor.process_text(full_text, source=path)
            
            if images:
                bind_images_to_chunks(docs, images, paragraphs, path, full_text, para_offsets)
                img_bound = sum(len(d.metadata.get("images", [])) for d in docs)
                console.print(f" {Path(path).name} → [cyan]{len(docs)}[/cyan] 个 chunk, [cyan]{img_bound}[/cyan] 个 chunk 绑定图⽚")
            else:
                console.print(f" {Path(path).name} → [cyan]{len(docs)}[/cyan] 个 chunk")
            
            all_docs.extend(docs)
    
    console.print(f"\n总计 [bold cyan]{len(all_docs)}[/bold cyan] 个 chunk")
    
    # Step 3: chunk 预览
    if show_chunks:
        console.print("\n[bold cyan]⽂档块预览[/bold cyan]")
        table = Table(show_lines=True)
        table.add_column("序号", width=4)
        table.add_column("来源", width=20)
        table.add_column("内容预览", width=70)
        
        for i, doc in enumerate(all_docs[:30]):  # 最多显示 30 条
            preview = doc.content[:80].replace("\n", " ") + "..."
            table.add_row(str(i+1), Path(doc.source).name, preview)
        
        console.print(table)
    
    # Step 4: 建⽴索引
    console.print("\n[bold cyan]Step 3/4 建⽴向量索引 + BM25 索引[/bold cyan]")
    os.makedirs(MENTAL_CoNFIG.storage_dir, exist_ok=True)
    retriever = HybridRetriever(MENTAL_CoNFIG)
    retriever.build(all_docs)
    retriever.save()
    
    # Step 5: 保存⽂档块
    console.print("\n[bold cyan]Step 4/4 保存⽂档块[/bold cyan]")
    docs_path = os.path.join(MENTAL_CoNFIG.storage_dir, "docs.json")
    DocumentProcessor(MENTAL_CoNFIG).save_docs(all_docs, docs_path)
    
    console.print(Panel(
        f"[bold green]\n⼼理知识库构建完成！[/bold green]\n\n"
        f"⽂件数量：{len(json_results)} 个 JSON + {len(docx_results)} 个 Word\n"
        f"总 chunk：{len(all_docs)} 个\n"
        f"存储⽬录：{MENTAL_CoNFIG.storage_dir}/\n\n"
        f"[yellow]下⼀步：[/yellow]\n"
        f"python mental_main.py chat",
        border_style="green"
    ))

# ─────────────────────────────────────────────
# 验证
# ─────────────────────────────────────────────
def verify():
    docs_path = os.path.join(MENTAL_CoNFIG.storage_dir, "docs.json")
    console.print("\n[bold cyan]\n验证⼼理知识库[/bold cyan]")
    
    all_ok = True
    for label, path in [
        ("docs.json", docs_path),
        ("faiss.index", MENTAL_CoNFIG.embedding.index_path),
        ("bm25.pkl", MENTAL_CoNFIG.bm25.index_path),
    ]:
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024
            console.print(f" [green]✓[/green] {label} ({size:.1f} KB)")
        else:
            console.print(f" [red]✗ 缺少 {label}[/red]")
            all_ok = False
    
    if all_ok:
        docs = DocumentProcessor(MENTAL_CoNFIG).load_docs(docs_path)
        console.print(f"\n共 [cyan]{len(docs)}[/cyan] 个 chunk，知识库正常")
    else:
        console.print("\n[red]知识库不完整，请重新运⾏ build[/red]")

# ─────────────────────────────────────────────
# ⼊⼝
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="⼼理知识库建库⼯具")
    subparsers = parser.add_subparsers(dest="command")
    
    build_p = subparsers.add_parser("build", help="建⽴知识库")
    build_p.add_argument("--dir", required=True, help="⽂档⽬录，如 ./mental_docs")
    build_p.add_argument("--show", action="store_true", help="打印 chunk 预览")
    
    subparsers.add_parser("verify", help="验证知识库")
    
    args = parser.parse_args()
    
    if args.command == "build":
        build_mental_knowledge_base(args.dir, args.show)
    elif args.command == "verify":
        verify()
    else:
        parser.print_help()