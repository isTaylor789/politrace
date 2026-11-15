"""Construcción y análisis del grafo de entidades (EI 3.5, visualización).

El grafo tiene dos capas:

* **nodos** = entidades canónicas (`PER:Javier Milei`, `ORG:IMSS`, `LOC:México`,
  `CARGO:secretario`) con atributos: tipo, menciones, documentos donde aparece.
* **aristas** = relaciones tipadas (`coocurre`, `inaugura`, `acusa`,
  `ocupa_cargo`, …) con peso = número de ocurrencias y evidencia textual.

Para las métricas se proyecta a un grafo **no dirigido ponderado** (el peso de la
arista es la suma de ocurrencias de todos los tipos), porque las medidas clásicas
(grado, intermediación, comunidades) están definidas sobre ese objeto. El grafo
tipado se conserva aparte para la exportación.
"""

from collections import Counter, defaultdict

import networkx as nx

from er_scripts import config
from er_scripts.mentions import canonical_label, node_type


class EntityGraph:
    """Grafo tipado con proyección no dirigida para métricas."""

    def __init__(self):
        self.typed = nx.MultiDiGraph()
        self.mentions: Counter = Counter()          # nodo -> nº de menciones
        self.docs: dict[str, set] = defaultdict(set)

    # -- construcción ---------------------------------------------------------
    def add_mention(self, node: str, doc_id: int | None) -> None:
        self.mentions[node] += 1
        if doc_id is not None:
            self.docs[node].add(doc_id)

    def add_edge(self, edge: dict) -> None:
        u, v = edge["source"], edge["target"]
        self.typed.add_edge(u, v, key=edge["type"], type=edge["type"],
                            weight=edge["weight"], docs=edge["docs"],
                            evidence=edge["evidence"])

    def add_edges(self, edges: list[dict]) -> None:
        for e in edges:
            self.add_edge(e)

    # -- estructura -----------------------------------------------------------
    def _ensure_nodes(self) -> None:
        for node in self.mentions:
            if node not in self.typed:
                self.typed.add_node(node)
        for node in self.typed.nodes:
            attrs = self.typed.nodes[node]
            attrs["type"] = node_type(node)
            attrs["label"] = canonical_label(node)
            attrs["mentions"] = self.mentions.get(node, 0)
            attrs["docs"] = len(self.docs.get(node, ()))

    def projection(self) -> nx.Graph:
        """Proyección no dirigida ponderada (suma de pesos de todos los tipos)."""
        self._ensure_nodes()
        g = nx.Graph()
        g.add_nodes_from(self.typed.nodes(data=True))
        for u, v, data in self.typed.edges(data=True):
            w = data.get("weight", 1)
            if g.has_edge(u, v):
                g[u][v]["weight"] += w
                g[u][v]["types"].add(data["type"])
            else:
                g.add_edge(u, v, weight=w, types={data["type"]})
        return g

    # -- métricas -------------------------------------------------------------
    def metrics(self) -> dict:
        full = self.projection()
        if full.number_of_nodes() == 0:
            return {}
        # las métricas de grafo se calculan sobre los nodos conectados: un nodo con
        # una sola mención y ninguna relación es un nodo aislado y distorsiona
        # densidad, modularidad e intermediación.
        aislados = [n for n, d in full.degree() if d == 0]
        g = full.subgraph([n for n, d in full.degree() if d > 0]).copy()
        degree = dict(g.degree())
        wdegree = dict(g.degree(weight="weight"))
        k = min(config.BETWEENNESS_K, g.number_of_nodes())
        between = nx.betweenness_centrality(g, k=k, weight=None, seed=config.__dict__.get("RANDOM_SEED", 0))
        # comunidades (Louvain-like / greedy modularity)
        communities = list(nx.community.greedy_modularity_communities(g, weight="weight"))
        community_of = {node: i for i, block in enumerate(communities) for node in block}

        for node in self.typed.nodes:
            self.typed.nodes[node]["degree"] = degree.get(node, 0)
            self.typed.nodes[node]["weighted_degree"] = wdegree.get(node, 0)
            self.typed.nodes[node]["betweenness"] = round(between.get(node, 0.0), 5)
            self.typed.nodes[node]["community"] = community_of.get(node)

        top_degree = sorted(degree.items(), key=lambda kv: -kv[1])[:15]
        top_between = sorted(between.items(), key=lambda kv: -kv[1])[:15]
        return {
            "nodos_totales": full.number_of_nodes(),
            "nodos_aislados": len(aislados),
            "nodos_conectados": g.number_of_nodes(),
            "nodos": g.number_of_nodes(),
            "aristas_proyectadas": g.number_of_edges(),
            "aristas_tipadas": self.typed.number_of_edges(),
            "densidad": round(nx.density(g), 6),
            "componentes_conexas": nx.number_connected_components(g),
            "componente_gigante": round(
                max((len(c) for c in nx.connected_components(g)), default=0) / max(1, full.number_of_nodes()), 3),
            "grado_medio": round(sum(degree.values()) / max(1, len(degree)), 2),
            "comunidades": len(communities),
            "modularidad": round(nx.algorithms.community.modularity(g, communities, weight="weight"), 4),
            "top_grado": [{"nodo": n, "label": canonical_label(n), "tipo": node_type(n), "grado": d}
                          for n, d in top_degree],
            "top_intermediacion": [{"nodo": n, "label": canonical_label(n), "tipo": node_type(n),
                                    "betweenness": round(b, 5)} for n, b in top_between],
            "comunidades_top": [
                {"id": i, "tamano": len(block),
                 "miembros": [canonical_label(n) for n in sorted(block)[:8]]}
                for i, block in enumerate(sorted(communities, key=len, reverse=True)[:6])
            ],
        }

    # -- exportación ----------------------------------------------------------
    def to_json(self, min_mentions: int = 1) -> dict:
        """Formato nodos/aristas (d3, pyvis, Cytoscape)."""
        self._ensure_nodes()
        nodes = []
        for node, attrs in self.typed.nodes(data=True):
            if attrs.get("mentions", 0) < min_mentions:
                continue
            nodes.append({
                "id": node,
                "label": attrs.get("label", canonical_label(node)),
                "type": attrs.get("type", node_type(node)),
                "mentions": attrs.get("mentions", 0),
                "docs": attrs.get("docs", 0),
                "degree": attrs.get("degree", 0),
                "weighted_degree": attrs.get("weighted_degree", 0),
                "betweenness": attrs.get("betweenness", 0.0),
                "community": attrs.get("community"),
            })
        keep = {n["id"] for n in nodes}
        links = [
            {"source": u, "target": v, "type": d.get("type"), "weight": d.get("weight", 1),
             "docs": d.get("docs", 0)}
            for u, v, d in self.typed.edges(data=True)
            if u in keep and v in keep
        ]
        return {"nodes": nodes, "links": links}

    def save_graphml(self, path) -> None:
        self._ensure_nodes()
        g = nx.Graph()
        for node, attrs in self.typed.nodes(data=True):
            g.add_node(node, label=attrs.get("label", node), type=attrs.get("type", ""),
                       mentions=int(attrs.get("mentions", 0)), docs=int(attrs.get("docs", 0)),
                       degree=int(attrs.get("degree", 0)),
                       betweenness=float(attrs.get("betweenness", 0.0)),
                       community=int(attrs.get("community") or -1))
        for u, v, data in self.typed.edges(data=True):
            if g.has_edge(u, v):
                g[u][v]["weight"] += data.get("weight", 1)
                g[u][v]["rel_types"] = g[u][v]["rel_types"] + "," + data.get("type", "")
            else:
                g.add_edge(u, v, weight=data.get("weight", 1), rel_types=data.get("type", ""))
        nx.write_graphml(g, path)

    # -- estadísticas ---------------------------------------------------------
    def node_stats(self) -> dict:
        by_type = Counter()
        for node, attrs in self.typed.nodes(data=True):
            by_type[attrs.get("type", node_type(node))] += 1
        top_mentions = sorted(self.mentions.items(), key=lambda kv: -kv[1])[:25]
        return {
            "nodos_por_tipo": dict(by_type.most_common()),
            "top_menciones": [
                {"nodo": n, "label": canonical_label(n), "tipo": node_type(n),
                 "menciones": c, "docs": len(self.docs.get(n, ()))}
                for n, c in top_mentions
            ],
            "nodos_con_un_solo_documento": sum(1 for n in self.mentions if len(self.docs.get(n, ())) == 1),
        }
