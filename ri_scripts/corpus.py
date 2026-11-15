"""Construcción del corpus: parquet unificado -> JSONL con offsets por línea."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ri_scripts import config

COLUMNS = ["speech_id", "speaker_name", "speech_date", "speech_title", "speech_text", "country_iso"]


def build_corpus(
    parquet: Path = config.PARQUET,
    out_jsonl: Path = config.CORPUS_JSONL,
    out_offsets: Path = config.CORPUS_OFFSETS,
    verbose: bool = True,
) -> int:
    """Escribe un registro JSON por línea (orden estable por hablante y fecha).

    El orden define el `doc_id` (0..N-1) que usan el índice y los buscadores.
    Los documentos de un mismo hablante quedan en un bloque contiguo, lo que
    permite filtrar por hablante con un simple rango de `doc_id`.
    """
    df = pd.read_parquet(parquet, columns=COLUMNS)
    df = df.dropna(subset=["speech_text"])
    df = df[df["speech_text"].astype(str).str.strip() != ""]

    df["speaker_name"] = df["speaker_name"].astype(str)
    df["speech_date"] = df["speech_date"].astype(str)
    df["country_iso"] = df["country_iso"].astype(str)
    df["speech_title"] = df["speech_title"].fillna("").astype(str)
    df["speech_text"] = df["speech_text"].astype(str)

    df = df.sort_values(["speaker_name", "speech_date", "speech_id"], kind="mergesort")
    df = df.reset_index(drop=True)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    offsets = np.zeros(len(df), dtype=np.int64)

    with open(out_jsonl, "w", encoding="utf-8") as fh:
        for i, row in enumerate(df.itertuples(index=False)):
            record = {
                "doc_id": i,
                "speech_id": row.speech_id,
                "speaker": row.speaker_name,
                "country": row.country_iso,
                "date": row.speech_date,
                "title": row.speech_title,
                "text": row.speech_text,
            }
            line = json.dumps(record, ensure_ascii=False) + "\n"
            offsets[i] = fh.tell()
            fh.write(line)

    np.save(out_offsets, offsets)
    if verbose:
        print(f"corpus: {len(df)} documentos -> {out_jsonl}")
        print(f"offsets: {out_offsets}")
        for speaker, n in df["speaker_name"].value_counts().items():
            print(f"  {speaker:32s} {n:5d}")
    return len(df)


def iter_corpus(path: Path = config.CORPUS_JSONL):
    """Itera los registros del corpus sin cargarlo entero en memoria."""
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)


def read_doc(doc_id: int, path: Path = config.CORPUS_JSONL, offsets=None) -> dict:
    """Lee un único documento usando su offset de byte (acceso aleatorio)."""
    if offsets is None:
        offsets = np.load(config.CORPUS_OFFSETS)
    with open(path, "rb") as fh:
        fh.seek(int(offsets[doc_id]))
        return json.loads(fh.readline().decode("utf-8"))
