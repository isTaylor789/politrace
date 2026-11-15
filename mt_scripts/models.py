"""Modelos de tópicos: LSA, NMF y LDA, con la misma interfaz.

Los tres reciben el `Space` de `matrices.py` y devuelven un `TopicModel` con

- ``topic_word``  (k, n_terms): distribución de términos por tópico (filas suman 1);
- ``doc_topic``   (n_docs, k): proporción de cada tópico en cada documento;
- ``topic_vector`` (k, V_global): el mismo tópico en el vocabulario global, para
  poder alinear tópicos entre modelos (hablante, año, ventana) sin remapear.

Diferencias matemáticas que importan:

- **LSA** (SVD truncada sobre ``W_l2``): componentes ortogonales con signo
  arbitrario. Se toma el valor absoluto de las cargas y se normaliza; es la
  práctica habitual cuando se leen como tópicos ("modelo latente" del temario
  5.3). Aporta la línea base no probabilística y la varianza explicada.
- **NMF** (``W_l2 ≈ H·V`` con ``H,V ≥ 0``): tópicos aditivos y dispersos, que es
  lo que hace legibles las listas de términos. Minimiza la divergencia
  euclidiana; se inicializa con ``nndsvda`` (determinista dado el seed).
- **LDA** (Dirichlet multinomial sobre conteos ``X``): modelo generativo
  probabilístico. La perplejidad se reporta, pero **no** se usa para elegir k
  (está mal alineada con la interpretabilidad: baja al aumentar k).
"""

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.preprocessing import normalize

import mt_scripts.config as config
from mt_scripts.matrices import Space


@dataclass
class TopicModel:
    """Resultado de ajustar un modelo de tópicos sobre un `Space`."""

    family: str
    k: int
    space: Space
    topic_word: np.ndarray            # (k, n_terms) filas normalizadas
    doc_topic: np.ndarray             # (n_docs, k) filas normalizadas
    extras: dict = field(default_factory=dict)

    @property
    def n_docs(self) -> int:
        return self.doc_topic.shape[0]

    @property
    def terms(self) -> list:
        return self.space.terms

    @property
    def terms_ids(self) -> np.ndarray:
        return self.space.terms_ids

    # --- términos ------------------------------------------------------------
    def top_idx(self, topic: int, n: int = config.TOP_TERMS) -> np.ndarray:
        """Posiciones (dentro de `space.terms`) de los n términos del tópico."""
        return np.argsort(-self.topic_word[topic], kind="mergesort")[:n]

    def top_ids(self, topic: int, n: int = config.TOP_TERMS) -> np.ndarray:
        """Ids **globales** de vocabulario de los n términos del tópico."""
        return self.terms_ids[self.top_idx(topic, n)]

    def top_terms(self, topic: int, n: int = config.TOP_TERMS) -> list:
        return [self.space.terms[j] for j in self.top_idx(topic, n)]

    def top_weights(self, topic: int, n: int = config.TOP_TERMS) -> np.ndarray:
        return self.topic_word[topic, self.top_idx(topic, n)]

    def topic_words(self, topic: int, n: int = config.TOP_TERMS_WINDOW) -> dict:
        """``{term_id_global: peso}`` normalizado (vocabulario del tópico)."""
        idx = self.top_idx(topic, n)
        w = self.topic_word[topic, idx]
        w = w / max(w.sum(), 1e-12)
        return {int(self.terms_ids[j]): float(v) for j, v in zip(idx, w)}

    def topic_vector(self, topic: int, n: int = config.TOP_TERMS_WINDOW) -> sp.csr_matrix:
        """Vector del tópico en el vocabulario global (fila CSR, suma 1)."""
        words = self.topic_words(topic, n)
        cols = np.fromiter(words.keys(), dtype=np.int64, count=len(words))
        data = np.fromiter(words.values(), dtype=np.float64, count=len(words))
        return sp.csr_matrix((data, (np.zeros(len(cols), dtype=np.int64), cols)),
                             shape=(1, self.space.global_vocab_size))

    def vectors(self, n: int = config.TOP_TERMS_WINDOW) -> sp.csr_matrix:
        return sp.vstack([self.topic_vector(t, n) for t in range(self.k)]).tocsr()

    def topic_set(self, topic: int, n: int = config.ALIGN_TERMS) -> sp.csr_matrix:
        """Vector **binario** de los n términos del tópico (para alinear)."""
        ids = self.top_ids(topic, n)
        return sp.csr_matrix(
            (np.ones(len(ids)), (np.zeros(len(ids), dtype=np.int64), ids)),
            shape=(1, self.space.global_vocab_size),
        )

    def sets(self, n: int = config.ALIGN_TERMS) -> sp.csr_matrix:
        return sp.vstack([self.topic_set(t, n) for t in range(self.k)]).tocsr()

    # --- documentos ----------------------------------------------------------
    def doc_topics_for(self, doc_ids) -> np.ndarray:
        """θ de los documentos indicados (posiciones dentro de `space.doc_ids`)."""
        pos = {int(d): i for i, d in enumerate(self.space.doc_ids)}
        rows = [pos[int(d)] for d in doc_ids]
        return self.doc_topic[rows]

    def dominant_topic(self, n: int = 1) -> np.ndarray:
        order = np.argsort(-self.doc_topic, axis=1)
        return order[:, :n] if n > 1 else order[:, 0]

    def share(self) -> np.ndarray:
        """Proporción media de cada tópico en el conjunto (filas de θ suman 1)."""
        return self.doc_topic.mean(axis=0)

    def exclusivity(self, n: int = config.TOP_TERMS_DIVERSITY) -> np.ndarray:
        """Exclusividad media de los top-n términos de cada tópico.

        ``excl(t) = media_j w_tj / Σ_{t'} w_t'j``: 1 significa que el término
        sólo aparece en ese tópico (tópico nítido), 1/k que está repartido.
        """
        col_sum = self.topic_word.sum(axis=0)
        col_sum[col_sum == 0] = 1.0
        out = np.zeros(self.k)
        for t in range(self.k):
            idx = self.top_idx(t, n)
            out[t] = float(np.mean(self.topic_word[t, idx] / col_sum[idx]))
        return out


# --- Ajuste ------------------------------------------------------------------
def fit_lsa(space: Space, k: int, *, seed: int = config.SEED,
            n_components: int = None) -> TopicModel:
    n_components = n_components or max(k, min(config.LSA_COMPONENTS, min(space.W_l2.shape) - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=seed, algorithm="randomized")
    theta = svd.fit_transform(space.W_l2)

    # Las k primeras componentes son los "tópicos"; las cargas pueden ser
    # negativas (signo arbitrario), así que se toma |V| y se normaliza.
    V = np.abs(svd.components_[:k])
    V = V / np.maximum(V.sum(axis=1, keepdims=True), 1e-12)

    doc_topic = np.abs(theta[:, :k])
    doc_topic = doc_topic / np.maximum(doc_topic.sum(axis=1, keepdims=True), 1e-12)

    return TopicModel(
        family="lsa", k=k, space=space, topic_word=V, doc_topic=doc_topic,
        extras={
            "explained_variance_ratio": float(svd.explained_variance_ratio_.sum()),
            "n_components": int(n_components),
        },
    )


def fit_nmf(space: Space, k: int, *, seed: int = config.SEED,
            max_iter: int = config.NMF_MAX_ITER) -> TopicModel:
    nmf = NMF(
        n_components=k, init=config.NMF_INIT, max_iter=max_iter,
        tol=config.NMF_TOL, random_state=seed,
    )
    theta = nmf.fit_transform(space.W_l2)
    V = nmf.components_.copy()
    V = V / np.maximum(V.sum(axis=1, keepdims=True), 1e-12)
    doc_topic = theta / np.maximum(theta.sum(axis=1, keepdims=True), 1e-12)
    return TopicModel(
        family="nmf", k=k, space=space, topic_word=V, doc_topic=doc_topic,
        extras={"reconstruction_err": float(nmf.reconstruction_err_),
                "n_iter": int(nmf.n_iter_)},
    )


def fit_lda(space: Space, k: int, *, seed: int = config.SEED,
            max_iter: int = config.LDA_MAX_ITER, compute_perplexity: bool = True) -> TopicModel:
    lda = LatentDirichletAllocation(
        n_components=k, max_iter=max_iter, learning_method=config.LDA_LEARNING_METHOD,
        batch_size=config.LDA_BATCH_SIZE, random_state=seed, n_jobs=config.LDA_JOBS,
        doc_topic_prior=config.LDA_DOC_TOPIC_PRIOR,
        topic_word_prior=config.LDA_TOPIC_WORD_PRIOR,
    )
    theta = lda.fit_transform(space.X)
    V = lda.components_.copy()
    V = V / np.maximum(V.sum(axis=1, keepdims=True), 1e-12)
    doc_topic = theta / np.maximum(theta.sum(axis=1, keepdims=True), 1e-12)
    extras = {"n_iter": int(lda.n_iter_), "log_likelihood": float(lda.bound_)}
    if compute_perplexity:
        extras["perplexity"] = float(lda.perplexity(space.X))
    return TopicModel(
        family="lda", k=k, space=space, topic_word=V, doc_topic=doc_topic, extras=extras,
    )


FAMILIES = {"lsa": fit_lsa, "nmf": fit_nmf, "lda": fit_lda}
FAMILY_LABEL = {"lsa": "LSA (SVD truncada)", "nmf": "NMF", "lda": "LDA"}


def fit(family: str, space: Space, k: int, **kwargs) -> TopicModel:
    """Ajusta una familia concreta; lanza `KeyError` si no existe."""
    return FAMILIES[family](space, k, **kwargs)


# --- Utilidades de ingeniería de tópicos ------------------------------------
def topic_diversity(model: TopicModel, n: int = config.TOP_TERMS_DIVERSITY) -> float:
    """Proporción de términos únicos en el top-n de todos los tópicos.

    ``div = |∪_t top_n(t)| / (k · n)``. Un valor de 1 significa que ningún
    término se repite entre tópicos; valores bajos indican tópicos solapados.
    """
    union = set()
    for t in range(model.k):
        union.update(model.top_ids(t, n).tolist())
    return len(union) / max(1, model.k * n)


def align_topics(a: TopicModel, b: TopicModel, n: int = config.ALIGN_TERMS) -> dict:
    """Empareja los tópicos de dos modelos por coseno de sus conjuntos de términos.

    Los vectores son **binarios** sobre el top-n (``sim = |A∩B| / sqrt(|A||B|)``):
    con pesos continuos el coseno entre dos tópicos del mismo tema cae a 0,01-0,05
    porque las colas de la distribución no se solapan, y el umbral de
    emparejamiento tendría que ser artificialmente bajo. Con binarios, compartir
    20 de 50 términos da 0,40 y compartir 1 da 0,02.

    El emparejamiento es uno a uno (asignación óptima húngara,
    `linear_sum_assignment` sobre ``-similitud``), que es lo que permite seguir un
    tópico a través de ventanas temporales sin que dos tópicos caigan en el mismo.
    """
    from scipy.optimize import linear_sum_assignment

    A = normalize(a.sets(n), norm="l2", copy=False)
    B = normalize(b.sets(n), norm="l2", copy=False)
    S = np.asarray((A @ B.T).todense())
    rows, cols = linear_sum_assignment(-S)
    return {int(r): (int(c), float(S[r, c])) for r, c in zip(rows, cols)}
