"""
graph_store.py — 知识图谱存储与检索
=====================================
基于 NetworkX 构建有向图：
  节点 = 实体（name, type, desc, source_chunks）
  边   = 关系（rel）

核心能力：
  - entity_to_chunks: 实体名 → chunk_id 列表（用于检索）
  - neighbors: 给定实体 → 返回 N 跳内的关联实体
  - query_expand: 从问题实体出发，找到相关 chunk
"""

import os
import json
from typing import List, Set, Dict, Optional
from rich.console import Console

console = Console()

GRAPH_DIR = "./storage/graph"


class KnowledgeGraph:
    def __init__(self, storage_dir: str = GRAPH_DIR):
        try:
            import networkx as nx
            self.G = nx.DiGraph()
        except ImportError:
            raise ImportError("请安装 networkx：pip install networkx")

        self.storage_dir = storage_dir
        # 实体名（小写）→ 原始实体名的映射（用于模糊匹配）
        self._entity_lower_map: Dict[str, str] = {}

    # ─────────────────────────────────────────
    # 构建
    # ─────────────────────────────────────────
    def build(self, extract_results: List[Dict]):
        """
        从抽取结果构建图谱
        extract_results: graph_extractor.batch_extract 的输出
        """
        for result in extract_results:
            chunk_id = result.get("chunk_id", "")

            for entity in result.get("entities", []):
                name = entity.get("name", "").strip()
                if not name:
                    continue
                if self.G.has_node(name):
                    # 合并同名实体的 source_chunks
                    existing = self.G.nodes[name].get("source_chunks", [])
                    new_chunks = entity.get("source_chunks", [chunk_id])
                    self.G.nodes[name]["source_chunks"] = list(
                        set(existing + new_chunks)
                    )
                else:
                    self.G.add_node(
                        name,
                        type=entity.get("type", "其他"),
                        desc=entity.get("desc", ""),
                        source_chunks=entity.get("source_chunks", [chunk_id]),
                    )
                self._entity_lower_map[name.lower()] = name

            for relation in result.get("relations", []):
                head = relation.get("head", "").strip()
                tail = relation.get("tail", "").strip()
                rel = relation.get("rel", "").strip()
                if head and tail and rel:
                    # 如果节点不存在则自动创建（来自关系但未被列为实体）
                    for n in [head, tail]:
                        if not self.G.has_node(n):
                            self.G.add_node(n, type="其他", desc="",
                                            source_chunks=[chunk_id])
                            self._entity_lower_map[n.lower()] = n
                    self.G.add_edge(head, tail, rel=rel)

        console.print(
            f"[green]✓ 知识图谱构建完成：{self.G.number_of_nodes()} 个实体，"
            f"{self.G.number_of_edges()} 条关系[/green]"
        )

    # ─────────────────────────────────────────
    # 检索
    # ─────────────────────────────────────────
    def get_chunks_for_entity(self, entity_name: str) -> List[str]:
        """返回某实体关联的 chunk_id 列表"""
        node = self._match_entity(entity_name)
        if node and self.G.has_node(node):
            return self.G.nodes[node].get("source_chunks", [])
        return []

    def get_neighbors(self, entity_name: str,
                      hops: int = 2) -> List[str]:
        """
        返回 entity_name 在图谱中 N 跳内的所有邻居实体名
        hops=1：直接相连的实体
        hops=2：两步可达的实体（默认，适合大多数问题）
        """
        import networkx as nx
        node = self._match_entity(entity_name)
        if not node or not self.G.has_node(node):
            return []

        visited: Set[str] = {node}
        frontier = {node}
        for _ in range(hops):
            next_frontier = set()
            for n in frontier:
                # 同时考虑出边和入边（无向遍历）
                next_frontier.update(self.G.successors(n))
                next_frontier.update(self.G.predecessors(n))
            frontier = next_frontier - visited
            visited.update(frontier)

        visited.discard(node)
        return list(visited)

    def expand_chunks(self, entity_names: List[str],
                      hops: int = 2, max_chunks: int = 15) -> List[str]:
        """
        从一组实体出发，通过图谱遍历找到相关 chunk_id
        返回去重后的 chunk_id 列表
        """
        chunk_ids: Set[str] = set()
        for name in entity_names:
            # 该实体自身的 chunk
            chunk_ids.update(self.get_chunks_for_entity(name))
            # 邻居实体的 chunk
            for neighbor in self.get_neighbors(name, hops):
                chunk_ids.update(self.get_chunks_for_entity(neighbor))
            if len(chunk_ids) >= max_chunks:
                break
        return list(chunk_ids)[:max_chunks]

    def search_entities_by_keyword(self, keyword: str) -> List[str]:
        """关键词模糊匹配实体名（用于从问题中找实体）"""
        kw_lower = keyword.lower()
        matched = []
        for lower_name, orig_name in self._entity_lower_map.items():
            if kw_lower in lower_name or lower_name in kw_lower:
                matched.append(orig_name)
        return matched[:10]

    def _match_entity(self, name: str) -> Optional[str]:
        """精确或模糊匹配实体名"""
        if self.G.has_node(name):
            return name
        lower = name.lower()
        return self._entity_lower_map.get(lower)

    # ─────────────────────────────────────────
    # 持久化
    # ─────────────────────────────────────────
    def save(self):
        import networkx as nx
        os.makedirs(self.storage_dir, exist_ok=True)
        # 保存图结构
        graph_path = os.path.join(self.storage_dir, "graph.json")
        data = nx.node_link_data(self.G)
        with open(graph_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        # 保存实体小写映射
        map_path = os.path.join(self.storage_dir, "entity_map.json")
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(self._entity_lower_map, f, ensure_ascii=False, indent=2)
        console.print(f"[green]✓ 图谱已保存：{graph_path}[/green]")

    def load(self) -> bool:
        import networkx as nx
        graph_path = os.path.join(self.storage_dir, "graph.json")
        map_path = os.path.join(self.storage_dir, "entity_map.json")
        if not os.path.exists(graph_path):
            return False
        with open(graph_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.G = nx.node_link_graph(data, directed=True)
        if os.path.exists(map_path):
            with open(map_path, "r", encoding="utf-8") as f:
                self._entity_lower_map = json.load(f)
        console.print(
            f"[green]✓ 图谱加载完成：{self.G.number_of_nodes()} 实体，"
            f"{self.G.number_of_edges()} 关系[/green]"
        )
        return True

    def stats(self):
        import networkx as nx
        from collections import Counter
        console.print(f"\n[bold cyan]知识图谱统计[/bold cyan]")
        console.print(f"  实体数：{self.G.number_of_nodes()}")
        console.print(f"  关系数：{self.G.number_of_edges()}")
        types = Counter(
            self.G.nodes[n].get("type", "其他") for n in self.G.nodes
        )
        for t, c in types.most_common(10):
            console.print(f"    {t}: {c}")
        # 度数最高的实体（最重要的节点）
        degree = sorted(self.G.degree(), key=lambda x: x[1], reverse=True)[:5]
        console.print(f"  核心实体（度数最高）：")
        for name, deg in degree:
            console.print(f"    {name}（度={deg}）")
