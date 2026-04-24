"""
graph_extractor.py — 实体与关系抽取
=====================================
用 DeepSeek 从每个 chunk 中抽取：
  entities:  [{"name": "国家奖学金", "type": "政策", "desc": "..."}]
  relations: [{"head": "国家奖学金", "rel": "申请条件", "tail": "综合测评前5%"}]
"""

import os
import json
import time
import httpx
from openai import OpenAI
from typing import List, Dict, Tuple
from rich.console import Console
from dotenv import load_dotenv

load_dotenv()
console = Console()

EXTRACT_PROMPT = """从以下文本中抽取实体和关系，用于构建知识图谱。

【文本】
{chunk}

要求：
1. 实体类型：人物、政策、机构、条件、金额、岗位、专业、地区、流程、其他
2. 关系要具体，如"申请条件"、"奖励金额"、"所属机构"、"适用对象"
3. 每个 chunk 最多抽取 8 个实体、10 条关系
4. 实体名称要规范简短，避免重复

只返回 JSON：
{{
  "entities": [
    {{"name": "实体名", "type": "类型", "desc": "一句话描述"}}
  ],
  "relations": [
    {{"head": "实体A", "rel": "关系", "tail": "实体B"}}
  ]
}}"""


class GraphExtractor:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            base_url="https://api.deepseek.com",
            http_client=httpx.Client(verify=False),
        )

    def extract(self, chunk_content: str, chunk_id: str,
                retry: int = 3) -> Dict:
        """从单个 chunk 抽取实体和关系"""
        # 截取前 600 字，避免超出 token 限制
        text = chunk_content[:600]
        prompt = EXTRACT_PROMPT.format(chunk=text)
        wait = 5

        for attempt in range(retry):
            try:
                resp = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=600,
                    temperature=0.1,
                    response_format={"type": "json_object"},
                )
                raw = resp.choices[0].message.content or "{}"
                data = json.loads(raw)
                entities = data.get("entities", [])
                relations = data.get("relations", [])

                # 给每个实体绑定来源 chunk
                for e in entities:
                    e["source_chunks"] = [chunk_id]
                return {"entities": entities, "relations": relations}

            except Exception as e:
                if attempt == retry - 1:
                    console.print(f"[dim]  抽取失败 {chunk_id[:8]}: {e}[/dim]")
                    return {"entities": [], "relations": []}
                time.sleep(wait)
                wait = min(wait * 2, 60)

    def batch_extract(self, chunks: List[Dict],
                      interval: float = 1.0) -> List[Dict]:
        """
        批量抽取，interval 控制请求间隔避免限速
        chunks: [{"doc_id": ..., "content": ...}, ...]
        """
        results = []
        total = len(chunks)
        for i, chunk in enumerate(chunks):
            console.print(
                f"  [dim]抽取 {i+1}/{total}  {chunk['doc_id'][:12]}...[/dim]",
                end="\r"
            )
            result = self.extract(chunk["content"], chunk["doc_id"])
            result["chunk_id"] = chunk["doc_id"]
            results.append(result)
            if i < total - 1:
                time.sleep(interval)
        console.print()
        return results
