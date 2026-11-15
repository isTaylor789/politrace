"""Genera todos los artefactos de RI: corpus JSONL + índice + w_global.

    python ri_scripts/build_all.py            # solo si faltan artefactos
    python ri_scripts/build_all.py --force    # reconstruye todo
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ri_scripts import config  # noqa: E402
from ri_scripts.corpus import build_corpus  # noqa: E402
from ri_scripts.index import artifacts_exist, build_index, save_index  # noqa: E402


def main(force: bool = False) -> None:
    if artifacts_exist() and not force:
        print("artefactos ya existentes; usa --force para reconstruir")
        return

    t0 = time.time()
    build_corpus()
    t1 = time.time()

    print("\nconstruyendo índice (tokenización + stemming + TF-IDF)...")
    ix = build_index()
    t2 = time.time()

    save_index(ix)
    t3 = time.time()

    print(f"\ncorpus:  {t1 - t0:6.1f}s")
    print(f"índice:  {t2 - t1:6.1f}s")
    print(f"guardado:{t3 - t2:6.1f}s")
    print(f"artefactos en {config.ARTIFACTS}")
    for path in sorted(config.ARTIFACTS.iterdir()):
        print(f"  {path.name:22s} {path.stat().st_size / 1e6:8.1f} MB")


if __name__ == "__main__":
    main(force="--force" in sys.argv)
