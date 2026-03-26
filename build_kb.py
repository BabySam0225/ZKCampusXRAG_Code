"""
build_kb.py — 知识库预处理 & 建库脚本
======================================
⽀持格式：
.docx → 提取正⽂ + 表格 + 图⽚（ 绑定到最近 chunk）
.jpg/.png → 跳过独⽴图⽚⽂件

图⽚处理策略（ 不需要多模态 Embedding） ：
1. 从 Word ⾥把图⽚保存到 ./storage/images/
2. 记录每张图⽚出现在第⼏个段落（  para_index）
3. 分块后把图⽚绑定到⽂字最近的 chunk（ metadata["images"]）
4. 检索命中 chunk 时图⽚随之返回

⽤法：
python build_kb.py build --dir ./docs
python build_kb.py build --dir ./docs --show
python build_kb.py verify
"""

import os
import json
import argparse
import hashlib
from pathlib import Path
from typing import List, Tuple, Dict

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from rag_config import DEFAULT_CONFIG
from document_processor import DocumentProcessor, Document
from hybrid_search import HybridRetriever

console = Console()
IMAGE_DIR = os.path.join(DEFAULT_CONFIG.storage_dir, "images")

# ─────────────────────────────────────────────
# Word 解析： 提取段落 + 提取图⽚ + 记录位置
# ─────────────────────────────────────────────
def load_docx_with_images(path: str):
    """
    返回：
    paragraphs: [{"index": int, "text": str}, ...]
    images:  [{"para_index": int, "path": str}, ...]
    """
    from docx import Document as DocxDoc
    from docx.text.paragraph import Paragraph
    from docx.table import Table as DocxTable
    from docx.oxml.ns import qn

    os.makedirs(IMAGE_DIR, exist_ok=True)
    doc = DocxDoc(path)

    paragraphs = []
    images = []
    para_cursor = 0

    def _save_images_in_element(element, para_idx):
        for blip in element.findall('.//' + qn('a:blip')):
            rId = blip.get(qn('r:embed'))
            if not rId or rId not in doc.part.rels:
                continue
            rel = doc.part.rels[rId]
            if "image" not in rel.reltype:
                continue
            img_bytes = rel.target_part.blob
            ext = rel.target_part.content_type.split("/")[-1].replace("jpeg", "jpg")
            img_hash = hashlib.md5(img_bytes).hexdigest()[:12]
            img_filename = f"{Path(path).stem}_{img_hash}.{ext}"
            img_path = os.path.join(IMAGE_DIR, img_filename)
            if not os.path.exists(img_path):
                with open(img_path, "wb") as f:
                    f.write(img_bytes)
            images.append({"para_index": para_idx, "path": img_path})

    for block in doc.element.body:
        tag = block.tag.split("}")[-1]

        if tag == "p":
            para = Paragraph(block, doc)
            text = para.text.strip()
            _save_images_in_element(block, para_cursor)
            if text:
                paragraphs.append({"index": para_cursor, "text": text})
            para_cursor += 1

        elif tag == "tbl":
            table = DocxTable(block, doc)
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    paragraphs.append({"index": para_cursor, "text": " | ".join(cells)})
                    para_cursor += 1

    return paragraphs, images

# ─────────────────────────────────────────────
# 构建带字符位置的全⽂ + 段落偏移表
# ─────────────────────────────────────────────
def build_full_text_with_offsets(paragraphs: List[dict]):
    """
    把段落列表拼成  full_text，  同时记录每个段落的字符起始/结束位置。
    返回：
    full_text: str
    offsets:{para_index: (char_start, char_end)}
    """
    parts = []
    offsets = {}
    cursor = 0
    for p in paragraphs:
        text = p["text"]
        start = cursor
        end = cursor + len(text)
        offsets[p["index"]] = (start, end)
        parts.append(text)
        cursor = end + 1  # +1 是换⾏符 "\n" 的⻓度
    full_text = "\n".join(p["text"] for p in paragraphs)
    return full_text, offsets

# ─────────────────────────────────────────────
# 把图⽚绑定到正确的 chunk（ 基于字符位置）
# ─────────────────────────────────────────────
def bind_images_to_chunks(
    docs: List[Document],
    images: List[dict],
    paragraphs: List[dict],
    source: str,
    full_text: str,
    para_offsets: dict,
) -> List[Document]:
    """
    正确绑定策略：
    1. 每个段落在 full_text ⾥有精确的字符区间 (char_start, char_end)
    2. 每个 chunk 的内容可在 full_text ⾥找到， 从⽽确定它覆盖哪个字符区间
    3. 图⽚的 para_index → 段落字符区间 → 落在哪个 chunk ⾥ → 绑定
    """
    target_docs = [d for d in docs if d.source == source]
    if not target_docs or not images:
        return docs

    # 建⽴每个 chunk 在 full_text 中的字符区间
    # ⽅法： 在 full_text ⾥搜索 chunk 内容（ 去掉换⾏符后匹配）
    chunk_ranges = []  # [(char_start, char_end, doc), ...]
    search_start = 0
    for doc in target_docs:
        # chunk 内容⾥的换⾏可能来⾃段落拼接， ⽤⾸尾各20字符定位
        needle = doc.content[:40].replace("\n", " ")
        pos = full_text.replace("\n", " ").find(needle, search_start)
        if pos == -1:
            pos = search_start  # 找不到时⽤上⼀个结束位置
        c_end = pos + len(doc.content)
        chunk_ranges.append((pos, c_end, doc))
        search_start = max(search_start, pos)

    # 对每张图⽚， 找字符位置最接近的 chunk
    for img in images:
        para_idx = img["para_index"]
        img_path = img["path"]
        # 图⽚的字符位置： 取该段落的起始位置
        # 如果 para_index 不在偏移表（ ⽐如图⽚在表格⾥） ， 取相邻段落
        if para_idx in para_offsets:
            img_char_pos = para_offsets[para_idx][0]
        else:
            # 找最近的段落
            nearest = min(para_offsets.keys(), key=lambda k: abs(k - para_idx))
            img_char_pos = para_offsets[nearest][0]

        # 优先： 找包含图⽚字符位置的 chunk
        bound = False
        for c_start, c_end, doc in chunk_ranges:
            if c_start <= img_char_pos <= c_end:
                doc.metadata.setdefault("images", [])
                if img_path not in doc.metadata["images"]:
                    doc.metadata["images"].append(img_path)
                bound = True
                break

        # 兜底： 找字符距离最近的 chunk
        if not bound and chunk_ranges:
            closest_doc = min(
                chunk_ranges,
                key=lambda t: min(abs(img_char_pos - t[0]), abs(img_char_pos - t[1]))
            )[2]
            closest_doc.metadata.setdefault("images", [])
            if img_path not in closest_doc.metadata["images"]:
                closest_doc.metadata["images"].append(img_path)

    return docs

# ─────────────────────────────────────────────
# 扫描⽬录， 加载所有 .docx
# ─────────────────────────────────────────────
def scan_and_load(dir_path: str):
    """返回 [(⽂件路径, 段落列表, 图⽚列表), ...]"""
    results = []
    files = sorted(Path(dir_path).rglob("*"))

    for f in files:
        if not f.is_file():
            continue
        ext = f.suffix.lower()

        if ext == ".docx":
            try:
                paragraphs, images = load_docx_with_images(str(f))
                char_count = sum(len(p["text"]) for p in paragraphs)
                if paragraphs:
                    results.append((str(f), paragraphs, images))
                    console.print(
                        f" [green]✓ Word[/green] {f.name} "
                        f"({char_count} 字符, {len(images)} 张图⽚)"
                    )
                else:
                    console.print(f" [yellow]⚠ 空⽂档， 跳过[/yellow] {f.name}")
            except Exception as e:
                console.print(f" [red]✗ 读取失败[/red] {f.name}: {e}")

        elif ext in {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}:
            console.print(f" [dim]跳过独⽴图⽚⽂件: {f.name}[/dim]")

        elif ext == ".doc":
            console.print(f" [yellow]⚠ .doc 不⽀持， 请另存为 .docx[/yellow] {f.name}")
    return results

# ─────────────────────────────────────────────
# 主建库流程
# ─────────────────────────────────────────────
def build_knowledge_base(dir_path: str, show_chunks: bool = False):
    console.print(Panel(f"[bold green]![img](tmp/410669243906_docxword_media_image1.png) 开始构建知识库[/bold green]\n⽬录： {dir_path}"))

    # Step 1: 扫描加载
    console.print("\n[bold cyan]Step 1/4 扫描⽂档[/bold cyan]")
    file_triples = scan_and_load(dir_path)
    if not file_triples:
        console.print("[red]![img](tmp/410669243906_docxword_media_image2.png) 未找到任何 .docx ⽂件[/red]")
        return
    console.print(f"\n 共加载 [cyan]{len(file_triples)}[/cyan] 个⽂档")

    # Step 2: 分块 + 图⽚绑定
    console.print("\n[bold cyan]Step 2/4 ⽂档分块 + 图⽚绑定[/bold cyan]")
    processor = DocumentProcessor(DEFAULT_CONFIG)
    all_docs: List[Document] = []
    total_images = 0

    for path, paragraphs, images in file_triples:
        # 拼接全⽂， 同时记录每个段落的字符起始位置
        full_text, para_offsets = build_full_text_with_offsets(paragraphs)
        docs = processor.process_text(full_text, source=path)
        # 把图⽚按字符位置精确绑定到对应 chunk
        if images:
            bind_images_to_chunks(docs, images, paragraphs, path, full_text, para_offsets)
            img_bound = sum(len(d.metadata.get("images", [])) for d in docs)
            total_images += img_bound
            console.print(
                f" {Path(path).name} → [cyan]{len(docs)}[/cyan] 个 chunk， "
                f"[yellow]{img_bound}[/yellow] 张图⽚已绑定"
            )
        else:
            console.print(f" {Path(path).name} → [cyan]{len(docs)}[/cyan] 个 chunk")
        all_docs.extend(docs)
    console.print(f"\n 总计 [bold cyan]{len(all_docs)}[/bold cyan] 个 chunk， "
                  f"[bold yellow]{total_images}[/bold yellow] 张图⽚绑定完成")

    # Step 3: chunk 预览（ 可选）
    if show_chunks:
        console.print("\n[bold cyan]⽂档块预览[/bold cyan]")
        table = Table(show_lines=True)
        table.add_column("序号", width=4)
        table.add_column("来源", width=18)
        table.add_column("图⽚", width=5)
        table.add_column("内容预览", width=65)
        for i, doc in enumerate(all_docs):
            img_count = len(doc.metadata.get("images", []))
            img_mark = f"[yellow]{img_count}张[/yellow]" if img_count else "-"
            preview = doc.content[:80].replace("\n", " ") + "..."
            table.add_row(str(i+1), Path(doc.source).name, img_mark, preview)
        console.print(table)

    # Step 4: 建⽴索引
    console.print("\n[bold cyan]Step 3/4 建⽴向量索引 + BM25 索引[/bold cyan]")
    os.makedirs(DEFAULT_CONFIG.storage_dir, exist_ok=True)
    retriever = HybridRetriever(DEFAULT_CONFIG)
    retriever.build(all_docs)
    retriever.save()

    # Step 5: 保存⽂档块
    console.print("\n[bold cyan]Step 4/4 保存⽂档块[/bold cyan]")
    docs_path = os.path.join(DEFAULT_CONFIG.storage_dir, "docs.json")
    processor.save_docs(all_docs, docs_path)

    console.print(Panel(
        f"[bold green]![img](tmp/410669243906_docxword_media_image3.png) 知识库构建完成！ [/bold green]\n\n"
        f" ⽂档数量： {len(file_triples)} 个⽂件\n"
        f" ⽂档块数： {len(all_docs)} 个 chunk\n"
        f" 图⽚数量： {total_images} 张（ 已绑定到对应 chunk） \n"
        f" 图⽚⽬录： {IMAGE_DIR}\\\n"
        f" 索引⽬录： {DEFAULT_CONFIG.storage_dir}\\\n\n"
        f"[yellow]下⼀步： [/yellow]\n"
        f" python main.py chat",
        border_style="green"
    ))

# ─────────────────────────────────────────────
# 验证知识库
# ─────────────────────────────────────────────
def verify_knowledge_base():
    docs_path = os.path.join(DEFAULT_CONFIG.storage_dir, "docs.json")
    console.print("\n[bold cyan]![img](tmp/410669243906_docxword_media_image4.png) 验证知识库[/bold cyan]")
    all_ok = True
    for label, path in [
        ("docs.json", docs_path),
        ("faiss.index", DEFAULT_CONFIG.embedding.index_path),
        ("bm25.pkl", DEFAULT_CONFIG.bm25.index_path),
    ]:
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024
            console.print(f" [green]✓[/green] {label} ({size:.1f} KB)")
        else:
            console.print(f" [red]✗ 缺少 {label}[/red]")
            all_ok = False

    if all_ok:
        from document_processor import DocumentProcessor
        docs = DocumentProcessor(DEFAULT_CONFIG).load_docs(docs_path)
        img_chunks = [d for d in docs if d.metadata.get("images")]
        total_imgs = sum(len(d.metadata["images"]) for d in img_chunks)
        console.print(f"\n 共 [cyan]{len(docs)}[/cyan] 个 chunk")
        console.print(f"  其中 [yellow]{len(img_chunks)}[/yellow] 个 chunk 绑定了图⽚， 总计 {total_imgs} 张")
        console.print("\n 知识库正常 ![img](tmp/410669243906_docxword_media_image3.png)")
    else:
        console.print("\n [red]知识库不完整， 请重新运⾏ build[/red]")

# ─────────────────────────────────────────────
# ⼊⼝
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="知识库预处理建库⼯具")
    subparsers = parser.add_subparsers(dest="command")

    build_p = subparsers.add_parser("build", help="从⽂档⽬录建⽴知识库")
    build_p.add_argument("--dir", required=True, help="⽂档⽬录， 例如 ./docs")
    build_p.add_argument("--show", action="store_true", help="打印所有 chunk 预览")

    subparsers.add_parser("verify", help="验证已建好的知识库")
    args = parser.parse_args()

    if args.command == "build":
        build_knowledge_base(args.dir, show_chunks=args.show)
    elif args.command == "verify":
        verify_knowledge_base()
    else:
        parser.print_help()