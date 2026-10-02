from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import config_digest
from .io import atomic_write_json, atomic_write_parquet, sha256_file, text_sha256

GOEMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]

CIVIL_LABELS = ["toxicity", "identity_attack", "obscene", "insult"]
GO_TARGET_LABELS = ["anger", "disgust", "fear", "sadness"]


def prepare_all(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    paths = config["paths"]
    output_dir = Path(paths["prepared_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    civil = prepare_civil_comments(
        paths["civil_input"], output_dir / "civil_comments.parquet", config
    )
    go = prepare_goemotions(
        paths["goemotions_input"], output_dir / "goemotions.parquet", config
    )
    manifest = {"config_sha256": config_digest(config), "civil_comments": civil, "goemotions": go}
    atomic_write_json(manifest, output_dir / "manifest.json")
    return manifest


def prepare_civil_comments(
    input_path: str | Path, output_path: str | Path, config: dict[str, Any]
) -> dict[str, Any]:
    input_path = _resolve_parquet(input_path)
    frame = pd.read_parquet(input_path)
    options = config["data"]["civil_comments"]
    expected = int(options["expected_input_rows"])
    if len(frame) != expected:
        raise ValueError(f"Civil Comments input has {len(frame):,} rows; expected {expected:,}")
    required = {"text", *CIVIL_LABELS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Civil Comments is missing columns: {sorted(missing)}")

    frame = frame.copy()
    frame.insert(0, "source_row", np.arange(len(frame), dtype=np.int64))
    before = len(frame)
    dedup = options["deduplication"]
    mode = dedup["mode"]
    if mode == "exact":
        columns = list(dedup["columns"])
        frame = frame.drop_duplicates(columns, keep=dedup.get("keep", "first"))
    elif mode == "text":
        frame = frame.drop_duplicates(["text"], keep=dedup.get("keep", "first"))
    elif mode == "id":
        if "id" not in frame:
            raise ValueError("deduplication.mode=id requires an id column")
        frame = frame.drop_duplicates(["id"], keep=dedup.get("keep", "first"))
    elif mode != "none":
        raise ValueError(f"Unknown Civil Comments deduplication mode: {mode}")

    frame = frame.reset_index(drop=True)
    threshold = float(options["label_threshold"])
    output = pd.DataFrame(
        {
            "example_id": [f"civil-{i:06d}" for i in range(len(frame))],
            "source_row": frame["source_row"].astype(np.int64),
            "text": frame["text"].astype(str),
        }
    )
    output["text_sha256"] = output["text"].map(text_sha256)
    for label in CIVIL_LABELS:
        output[f"soft_{label}"] = frame[label].astype(float)
        output[f"y_{label}"] = (frame[label].astype(float) >= threshold).astype(np.int8)
    atomic_write_parquet(output, output_path)
    manifest = {
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "input_rows": before,
        "output_rows": len(output),
        "duplicates_removed": before - len(output),
        "deduplication": dedup,
        "label_threshold": threshold,
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
    }
    atomic_write_json(manifest, Path(output_path).with_suffix(".manifest.json"))
    return manifest


def prepare_goemotions(
    input_path: str | Path, output_path: str | Path, config: dict[str, Any]
) -> dict[str, Any]:
    input_path = _resolve_parquet(input_path)
    frame = pd.read_parquet(input_path)
    expected = int(config["data"]["goemotions"]["expected_input_rows"])
    if len(frame) != expected:
        raise ValueError(f"GoEmotions input has {len(frame):,} rows; expected {expected:,}")
    required = {"text", "labels"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"GoEmotions is missing columns: {sorted(missing)}")

    output = pd.DataFrame(
        {
            "example_id": [f"go-{i:05d}" for i in range(len(frame))],
            "source_row": np.arange(len(frame), dtype=np.int64),
            "text": frame["text"].astype(str),
        }
    )
    output["text_sha256"] = output["text"].map(text_sha256)
    label_ids = {name: GOEMOTIONS_LABELS.index(name) for name in GO_TARGET_LABELS}
    normalized = frame["labels"].map(lambda values: {int(value) for value in values})
    for name, label_id in label_ids.items():
        output[f"y_{name}"] = normalized.map(
            lambda values, target=label_id: int(target in values)
        ).astype(np.int8)
    atomic_write_parquet(output, output_path)
    manifest = {
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "input_rows": len(frame),
        "output_rows": len(output),
        "label_ids": label_ids,
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
    }
    atomic_write_json(manifest, Path(output_path).with_suffix(".manifest.json"))
    return manifest


def _resolve_parquet(path: str | Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(path)
    candidates = sorted(path.rglob("*.parquet"))
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one parquet under {path}, found {len(candidates)}")
    return candidates[0]
