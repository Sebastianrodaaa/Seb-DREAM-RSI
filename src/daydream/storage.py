"""Local run history.

One directory per run. Trees and the summary are JSON. The directory is the
source of truth for that run; nothing is written until a round's tree exists
in memory, and a failed later round does not rewrite earlier tree files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from daydream.dream import DreamOutcome


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_run(out_dir: str, result: Any, dream: DreamOutcome | None) -> None:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    for index, tree in enumerate(result.trees, start=1):
        _write(root / "history" / f"tree_{index:03d}.json", tree.to_dict())
    summary = result.to_dict()
    if dream is not None:
        summary["last_dream"] = {
            "score": dream.score,
            "spec": dream.spec.to_dict(),
            "trace": dream.trace,
        }
    _write(root / "summary.json", summary)
