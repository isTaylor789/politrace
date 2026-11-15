"""Deduplicación de eventos con MinHash + LSH (familias de MinHash, EI/MT).

Dos problemas distintos:

1. **Casi duplicados léxicos**: los presidentes repiten la misma promesa en
   discursos distintos ("vamos a construir el tren maya"). Comparar todos los
   pares es O(n²); con ~10⁵ eventos son 10¹⁰ comparaciones. MinHash + LSH lo baja
   a ~O(n) con una probabilidad de colisión controlada.

2. **Firma MinHash**: con k permutaciones, la probabilidad de que dos conjuntos
   coincidan en una posición es su Jaccard. La estimación es insesgada:

       Ĵ(A,B) = (1/k) Σ_i [ min_{a∈A} h_i(a) == min_{b∈B} h_i(b) ]
       E[Ĵ] = J(A,B) = |A∩B| / |A∪B|

   con h_i(x) = (a_i·x + b_i) mod p, p = 2³¹−1 (primo de Mersenne: el producto
   a_i·x cabe en 64 bits y la aritmética se puede vectorizar).

3. **Banding LSH**: con b bandas de r filas, dos conjuntos con Jaccard s colisionan
   en al menos una banda con probabilidad

       P_colisión(s) = 1 − (1 − s^r)^b

   Con b=12, r=8 y s=0.8 → 1 − (1 − 0.8⁸)¹² ≈ 0.98; con s=0.5 → ≈ 0.32.
   Los candidatos se **verifican** con el Jaccard exacto, así que un falso
   candidato sólo cuesta tiempo, nunca un error de agrupación.
"""

import hashlib

import numpy as np

from ei_scripts import config

_P = 2_147_483_647  # 2^31 - 1


def _rng(seed: int = config.RANDOM_SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


class MinHasher:
    """Firma MinHash vectorizada sobre conjuntos de shingles."""

    def __init__(self, perms: int = config.MINHASH_PERM, shingle_size: int = config.SHINGLE_SIZE,
                 seed: int = config.RANDOM_SEED):
        self.perms = perms
        self.k = shingle_size
        rng = _rng(seed)
        # a_i ∈ [1, p-1], b_i ∈ [0, p-1]: familia universal de hashes lineales
        self.a = rng.integers(1, _P - 1, size=perms, dtype=np.int64)
        self.b = rng.integers(0, _P - 1, size=perms, dtype=np.int64)

    # -- shingles -------------------------------------------------------------
    def shingle_hashes(self, tokens: list[str]) -> np.ndarray:
        """Hashes de los k-shingles de palabras (conjunto, sin orden)."""
        if len(tokens) < self.k:
            seqs = [" ".join(tokens)] if tokens else []
        else:
            seqs = [" ".join(tokens[i:i + self.k]) for i in range(len(tokens) - self.k + 1)]
        if not seqs:
            return np.zeros(0, dtype=np.int64)
        hashes = {
            int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little") % _P
            for s in seqs
        }
        return np.fromiter(sorted(hashes), dtype=np.int64, count=len(hashes))

    # -- firma ----------------------------------------------------------------
    def signature(self, hashes: np.ndarray) -> np.ndarray | None:
        """min_i (a_i·x + b_i) mod p sobre todos los shingles x."""
        if hashes.size == 0:
            return None
        prod = (self.a[:, None] * hashes[None, :] + self.b[:, None]) % _P
        return prod.min(axis=1)

    def jaccard(self, h1: np.ndarray, h2: np.ndarray) -> float:
        """Jaccard exacto entre dos conjuntos de hashes de shingle."""
        if h1.size == 0 or h2.size == 0:
            return 0.0
        s1, s2 = set(h1.tolist()), set(h2.tolist())
        inter = len(s1 & s2)
        return inter / (len(s1) + len(s2) - inter)


class UnionFind:
    """Union-Find con compresión de caminos (para los grupos de duplicados)."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def deduplicate(events: list[dict], tokens_of, speaker_of, hasher: MinHasher | None = None,
                verbose: bool = False) -> dict:
    """Agrupa eventos casi duplicados del mismo hablante.

    `tokens_of(event) -> list[str]` y `speaker_of(event) -> str` desacoplan el
    módulo de la estructura del evento. Devuelve:

        {"grupo": {idx: gid}, "representante": {gid: idx}, "grupos": n,
         "pares_verificados": m, "pares_colision": c}
    """
    hasher = hasher or MinHasher()
    n = len(events)

    sigs: list[np.ndarray | None] = [None] * n
    hashes: list[np.ndarray | None] = [None] * n
    for i, ev in enumerate(events):
        toks = tokens_of(ev)
        if len(toks) < config.DEDUP_MIN_TOKENS:
            continue
        h = hasher.shingle_hashes(toks)
        if h.size == 0:
            continue
        hashes[i] = h
        sigs[i] = hasher.signature(h)

    # -- bandas ---------------------------------------------------------------
    bands = config.LSH_BANDS
    rows = config.LSH_ROWS
    buckets: dict[tuple, list[int]] = {}
    for i, sig in enumerate(sigs):
        if sig is None:
            continue
        key_prefix = speaker_of(events[i])
        for band in range(bands):
            key = (key_prefix, band) + tuple(sig[band * rows:(band + 1) * rows].tolist())
            buckets.setdefault(key, []).append(i)

    uf = UnionFind(n)
    verified = collisions = 0
    for members in buckets.values():
        if len(members) < 2 or len(members) > config.LSH_MAX_BUCKET:
            continue
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                i, j = members[x], members[y]
                collisions += 1
                if uf.find(i) == uf.find(j):
                    continue
                verified += 1
                if hasher.jaccard(hashes[i], hashes[j]) >= config.DEDUP_JACCARD:
                    uf.union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        if sigs[i] is None:
            continue
        groups.setdefault(uf.find(i), []).append(i)

    mapping: dict[int, int] = {}
    representative: dict[int, int] = {}
    gid = 0
    for root, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if len(members) < 2:
            continue
        members.sort(key=lambda idx: (events[idx].get("date") or "", events[idx].get("doc_id") or 0,
                                      events[idx].get("sent_idx") or 0))
        representative[gid] = members[0]
        for idx in members:
            mapping[idx] = gid
        gid += 1

    if verbose:
        print(f"  dedup: {gid} grupos, {collisions} colisiones LSH, {verified} pares verificados")
    return {
        "grupo": mapping,
        "representante": representative,
        "grupos": gid,
        "pares_verificados": verified,
        "pares_colision": collisions,
    }


def collision_probability(s: float, bands: int = config.LSH_BANDS, rows: int = config.LSH_ROWS) -> float:
    """P(al menos una banda en común) para un Jaccard s."""
    return 1.0 - (1.0 - s ** rows) ** bands


def expected_error(k: int = config.MINHASH_PERM, s: float = 0.8) -> float:
    """Desviación típica de la estimación MinHash: sqrt(s(1-s)/k)."""
    return (s * (1 - s) / k) ** 0.5
