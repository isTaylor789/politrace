"""Extracción de eventos de promesa/anuncio sobre una oración.

Pipeline por oración (EI 3.2 + 3.3 + 3.4):

    1. tokenización con offsets                  (segment.py)
    2. Aho-Corasick: disparadores + gazetteers   (automaton.py)
    3. slots por expresiones regulares           (slots.py)
    4. chunker + CFG/CYK                         (grammar.py)
    5. plantilla de evento + confianza           (este módulo)
    6. deduplicación MinHash/LSH global          (dedup.py)

El score de confianza es una suma ponderada de evidencias, normalizada por el peso
total activo (para que las ablaciones sean comparables):

    conf = (Σ w_i · e_i) / (Σ w_i) · φ(polaridad)

con e_i ∈ {0,1} y φ = 1 (afirmado), 0.7 (atenuado), NEGATION_FACTOR (negado).
"""

from dataclasses import dataclass

from ei_scripts import config, lexicon
from ei_scripts.automaton import build_trigger_automaton, resolve_triggers
from ei_scripts.grammar import build_chunks, cyk, event_type, is_action_token, is_content_action
from ei_scripts.segment import segment
from ei_scripts.slots import extract_slots


@dataclass(slots=True)
class Options:
    """Interruptores que permiten las ablaciones del reporte."""

    use_slots: bool = True
    use_cyk: bool = True
    use_negation: bool = True
    use_gazetteer: bool = True
    min_conf: float = config.MIN_CONFIDENCE


_AC = None


def automaton():
    """Autómata compartido (se construye una sola vez)."""
    global _AC
    if _AC is None:
        _AC = build_trigger_automaton()
    return _AC


def _sentence_gazetteer(matches, ac) -> list[tuple[str, int, int]]:
    """Gazetteer (lugares/organizaciones) como anclas de chunk."""
    out, occupied = [], set()
    for m in sorted(matches, key=lambda x: (-x.length, x.start)):
        kind = ac.tags[m.pid][0]
        if kind not in {"place", "org"}:
            continue
        span = set(range(m.start, m.end + 1))
        if span & occupied:
            continue
        occupied |= span
        out.append(("LOC" if kind == "place" else "ORG", m.start, m.end))
    return out


def _confidence(trigger_strength: float, action_quality: float, has_slot: bool,
                parsed: bool, has_place: bool, polarity: str, opts: "Options | None" = None) -> float:
    """Suma ponderada de evidencias, normalizada por los pesos **activos**.

    La normalización hace comparables las ablaciones: en la configuración A (sin
    slots ni CYK) el máximo también es 1.0, en vez de quedar recortado por pesos
    que nadie puede activar.
    """
    opts = opts or Options()
    terms = [(config.W_TRIGGER, trigger_strength), (config.W_ACTION, action_quality)]
    if opts.use_slots:
        terms.append((config.W_SLOT, 1.0 if has_slot else 0.0))
    if opts.use_cyk:
        terms.append((config.W_CYK, 1.0 if parsed else 0.0))
    if opts.use_gazetteer:
        terms.append((config.W_LUGAR, 1.0 if has_place else 0.0))
    total_w = sum(w for w, _ in terms) or 1.0
    score = sum(w * v for w, v in terms) / total_w
    if polarity == "negado":
        score *= config.NEGATION_FACTOR
    elif polarity == "atenuado":
        score *= 0.70
    return round(min(1.0, max(0.0, score)), 4)


def _frame(chunks) -> dict:
    """Slots del marco, tomados de los chunks que el chunker marcó."""
    frame = {"fecha": None, "monto": None, "cantidad": None, "porcentaje": None,
             "lugar": None, "organizacion": None, "beneficiario": None}
    action = []
    for ch in chunks:
        is_slot = ch.payload is not None and hasattr(ch.payload, "kind")
        if ch.label == "PLAZO" and frame["fecha"] is None:
            if is_slot:
                frame["fecha"] = {"texto": ch.payload.surface, "valor": ch.payload.value}
            elif ch.payload is not None:                      # disparador temporal
                frame["fecha"] = {"texto": ch.text, "valor": "rel:" + ch.text}
        elif ch.label == "MONTO" and is_slot:
            key = {"MONEY": "monto", "QTY": "cantidad", "PCT": "porcentaje"}[ch.payload.kind]
            if frame[key] is None:
                frame[key] = {"texto": ch.payload.surface, "valor": ch.payload.value}
        elif ch.label == "LOC" and frame["lugar"] is None:
            frame["lugar"] = ch.text
        elif ch.label == "ORG" and frame["organizacion"] is None:
            frame["organizacion"] = ch.text
        elif ch.label == "BENEF" and frame["beneficiario"] is None:
            frame["beneficiario"] = ch.payload.surface if is_slot else ch.text
        elif ch.label in {"ACC", "VTRIG"}:
            action.append(ch.text)
    return frame, " ".join(action)


def extract_sentence(sentence, opts: Options, ac=None, doc_meta: dict | None = None,
                     stats: dict | None = None) -> dict | None:
    """Intenta convertir una oración en un evento. Devuelve None si no lo es.

    `stats` (opcional) acumula el embudo de filtros, que es lo que se reporta en
    `stats.json`: cuántas oraciones llegan a cada etapa.
    """
    ac = ac or automaton()
    words = [t for t in sentence.tokens if t.is_word]
    if stats is not None:
        stats["oraciones"] = stats.get("oraciones", 0) + 1
    if len(words) < 4:
        if stats is not None:
            stats["descartadas_cortas"] = stats.get("descartadas_cortas", 0) + 1
        return None

    # Una pregunta retórica no es una promesa ("¿cómo va a ser mejor comprar afuera?").
    if "?" in sentence.text:
        if stats is not None:
            stats["rechazadas_pregunta"] = stats.get("rechazadas_pregunta", 0) + 1
        return None

    # Los discursos entre comillas son citas: no son promesas del hablante.
    if "\u201c" in sentence.text or "\u201d" in sentence.text:
        if stats is not None:
            stats["rechazadas_cita"] = stats.get("rechazadas_cita", 0) + 1
        return None

    folded = [w.folded for w in words]
    matches = ac.scan(folded)
    triggers = resolve_triggers(matches, ac, words, sentence.text)
    if not triggers:
        if stats is not None:
            stats["sin_disparador"] = stats.get("sin_disparador", 0) + 1
        return None
    if stats is not None:
        stats["con_disparador"] = stats.get("con_disparador", 0) + 1

    best = max(triggers, key=lambda t: (t.strength, t.end - t.start))

    slots = extract_slots(sentence) if opts.use_slots else []
    gaz = _sentence_gazetteer(matches, ac) if opts.use_gazetteer else []
    chunks = build_chunks(words, triggers, slots, gaz)
    labels = [c.label for c in chunks]

    parsed, tree = cyk(labels)
    if stats is not None and parsed:
        stats["cyk_acepta"] = stats.get("cyk_acepta", 0) + 1
    if opts.use_cyk and not parsed:
        if stats is not None:
            stats["rechazadas_cyk"] = stats.get("rechazadas_cyk", 0) + 1
        return None

    # La acción debe estar cerca del disparador (perífrasis verbal local).
    action_pos = next((i for i, w in enumerate(words) if is_action_token(w)), None)
    near = action_pos is not None and (action_pos - best.end) <= config.MAX_ACTION_DISTANCE.get(best.cls, 6)
    has_action = near or "VTRIG" in labels
    if not has_action:
        if stats is not None:
            stats["rechazadas_sin_accion"] = stats.get("rechazadas_sin_accion", 0) + 1
        return None

    polarity = best.polarity if opts.use_negation else "afirmado"
    frame, action_text = _frame(chunks)
    has_slot = any(frame[k] is not None for k in ("fecha", "monto", "cantidad", "porcentaje"))
    has_place = frame["lugar"] is not None

    if opts.use_negation and polarity == "negado":
        if stats is not None:
            stats["rechazadas_negacion"] = stats.get("rechazadas_negacion", 0) + 1
        return None

    # Calidad de la acción: 1.0 si es de contenido (construir, entregar...),
    # 0.5 si es sólo de discurso (informar, presentar).
    near_content = any(
        is_content_action(w) for w in words[max(0, best.end - 1):best.end + 5]
    )
    action_quality = 1.0 if near_content else 0.5
    conf = _confidence(best.strength, action_quality, has_slot, parsed, has_place, polarity, opts)
    if conf < opts.min_conf:
        if stats is not None:
            stats["rechazadas_confianza"] = stats.get("rechazadas_confianza", 0) + 1
        return None
    if stats is not None:
        stats["aceptadas"] = stats.get("aceptadas", 0) + 1

    etype, evidence = event_type(chunks, words)
    doc_meta = doc_meta or {}
    return {
        "doc_id": doc_meta.get("doc_id"),
        "speech_id": doc_meta.get("speech_id"),
        "speaker": doc_meta.get("speaker"),
        "country": doc_meta.get("country"),
        "date": doc_meta.get("date"),
        "sent_idx": sentence.idx,
        "char_start": sentence.start,
        "char_end": sentence.end,
        "sentence": sentence.text.strip(),
        "trigger": best.seq and " ".join(best.seq),
        "trigger_class": best.cls,
        "trigger_canon": best.canon,
        "strength": round(best.strength, 3),
        "polarity": polarity,
        "action": action_text,
        "event_type": etype,
        "type_evidence": evidence,
        "frame": frame,
        "chunks": [{"label": c.label, "text": c.text} for c in chunks],
        "cyk": parsed,
        "cyk_tree": tree,
        "confidence": conf,
        "method": "+".join(
            ["ac"] + (["regex"] if opts.use_slots else []) + (["cyk"] if opts.use_cyk else [])
        ),
    }


def extract_doc(doc: dict, opts: Options | None = None, ac=None) -> list[dict]:
    """Eventos de un documento del corpus (`corpus.jsonl`)."""
    opts = opts or Options()
    ac = ac or automaton()
    meta = {k: doc.get(k) for k in ("doc_id", "speech_id", "speaker", "country", "date")}
    events = []
    for sentence in segment(doc["text"]):
        ev = extract_sentence(sentence, opts, ac, meta)
        if ev is not None:
            events.append(ev)
    return events


def iter_events(opts: Options | None = None, corpus_path=None):
    """Genera eventos documento a documento (streaming sobre el corpus)."""
    from ri_scripts import config as ri_config
    from ri_scripts.corpus import iter_corpus

    opts = opts or Options()
    ac = automaton()
    for doc in iter_corpus(corpus_path or ri_config.CORPUS_JSONL):
        yield from extract_doc(doc, opts, ac)


def frame_to_columns(event: dict) -> dict:
    """Aplana el marco para tablas: una columna por slot."""
    frame = event.get("frame") or {}
    flat = {}
    for key, value in frame.items():
        flat[key] = value["texto"] if isinstance(value, dict) else value
    flat["fecha_valor"] = (frame.get("fecha") or {}).get("valor") if isinstance(frame.get("fecha"), dict) else None
    flat["monto_valor"] = (frame.get("monto") or {}).get("valor") if isinstance(frame.get("monto"), dict) else None
    flat["cantidad_valor"] = (frame.get("cantidad") or {}).get("valor") if isinstance(frame.get("cantidad"), dict) else None
    return flat


def json_ready(event: dict) -> dict:
    """Copia serializable (el árbol CYK vuelve a ser texto)."""
    from ei_scripts.grammar import render_tree

    out = dict(event)
    tree = out.get("cyk_tree")
    out["cyk_tree"] = render_tree(tree) if tree else None
    return out
