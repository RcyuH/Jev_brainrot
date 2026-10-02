from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import config_digest
from .io import atomic_write_json, atomic_write_parquet, sha256_file


async def infer_dataset(dataset: str, config: dict[str, Any]) -> dict[str, Any]:
    if dataset not in {"civil_comments", "goemotions"}:
        raise ValueError(f"Unknown dataset: {dataset}")
    prepared_path = Path(config["paths"]["prepared_dir"]) / f"{dataset}.parquet"
    output_dir = Path(config["paths"]["predictions_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset}.parquet"
    checkpoint_path = output_dir / f"{dataset}.jsonl"
    frame = pd.read_parquet(prepared_path)

    model_config = config["model"]
    endpoints = [value.rstrip("/") for value in model_config["endpoints"]]
    if not endpoints:
        raise ValueError("At least one model endpoint is required")
    prompts = model_config["prompts"][dataset]
    request_signature = _request_signature(
        model_config["revision"], prompts, sha256_file(prepared_path)
    )
    completed = _load_checkpoint(checkpoint_path, request_signature)
    pending = [
        row
        for row in frame.itertuples(index=False)
        if row.example_id not in completed or completed[row.example_id].get("error")
    ]

    clients = [
        httpx.AsyncClient(timeout=float(model_config["timeout_seconds"])) for _ in endpoints
    ]
    semaphores = [
        asyncio.Semaphore(int(model_config["max_in_flight_per_endpoint"]))
        for _ in endpoints
    ]
    versions = await _fetch_versions(clients, endpoints)
    try:
        with checkpoint_path.open("a", encoding="utf-8", buffering=1) as checkpoint:
            progress = tqdm(total=len(pending), desc=f"Infer {dataset}", unit="row")
            batch_size = max(256, len(endpoints) * int(model_config["max_in_flight_per_endpoint"]) * 8)
            for start in range(0, len(pending), batch_size):
                rows = pending[start : start + batch_size]
                tasks = []
                for offset, row in enumerate(rows, start=start):
                    endpoint_index = offset % len(endpoints)
                    tasks.append(
                        _infer_one(
                            row.example_id,
                            row.text,
                            prompts,
                            endpoint_index,
                            clients[endpoint_index],
                            endpoints[endpoint_index],
                            semaphores[endpoint_index],
                            int(model_config["retries"]),
                            request_signature,
                        )
                    )
                results = await asyncio.gather(*tasks)
                for result in results:
                    checkpoint.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                    completed[result["example_id"]] = result
                    progress.update(1)
            progress.close()
    finally:
        await asyncio.gather(*(client.aclose() for client in clients))

    failures = [value for value in completed.values() if value.get("error")]
    if failures:
        sample = failures[:3]
        raise RuntimeError(
            f"Inference has {len(failures)} failed rows. Checkpoint retained at {checkpoint_path}. "
            f"Examples: {sample}"
        )
    prediction_rows = []
    for example_id in frame["example_id"]:
        result = completed.get(example_id)
        if result is None:
            raise RuntimeError(f"Missing prediction for {example_id}")
        prediction_rows.append(
            {"example_id": example_id, **{f"p_{key}": result["probabilities"][key] for key in prompts}}
        )
    predictions = pd.DataFrame(prediction_rows)
    merged = frame.merge(predictions, on="example_id", how="left", validate="one_to_one")
    probability_columns = [f"p_{key}" for key in prompts]
    values = merged[probability_columns].to_numpy(float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Model returned invalid probabilities")
    atomic_write_parquet(merged, output_path)
    manifest = {
        "dataset": dataset,
        "rows": len(merged),
        "prepared_input": str(prepared_path),
        "prepared_sha256": sha256_file(prepared_path),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "checkpoint": str(checkpoint_path),
        "config_sha256": config_digest(config),
        "model_revision": model_config["revision"],
        "endpoints": endpoints,
        "endpoint_versions": versions,
        "prompts": prompts,
    }
    atomic_write_json(manifest, output_path.with_suffix(".manifest.json"))
    return manifest


async def _infer_one(
    example_id: str,
    text: str,
    prompts: dict[str, Any],
    endpoint_index: int,
    client: httpx.AsyncClient,
    endpoint: str,
    semaphore: asyncio.Semaphore,
    retries: int,
    request_signature: str,
) -> dict[str, Any]:
    payload = {"model": "openjev", "state": text, "questions": prompts}
    started = time.perf_counter()
    last_error = "unknown error"
    for attempt in range(retries + 1):
        try:
            async with semaphore:
                response = await client.post(f"{endpoint}/v1/systemone", json=payload)
            response.raise_for_status()
            body = response.json()
            answers = body["answers"]
            probabilities = {}
            for key in prompts:
                value = float(answers[key]["noul"])
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"Probability outside [0, 1] for {key}: {value}")
                probabilities[key] = value
            return {
                "example_id": example_id,
                "probabilities": probabilities,
                "model": body.get("model"),
                "endpoint_index": endpoint_index,
                "elapsed_seconds": time.perf_counter() - started,
                "request_signature": request_signature,
            }
        except (httpx.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            last_error = f"{type(error).__name__}: {error}"
            if attempt < retries:
                await asyncio.sleep(min(16.0, 0.5 * (2**attempt)) + random.random() * 0.2)
    return {
        "example_id": example_id,
        "error": last_error,
        "endpoint_index": endpoint_index,
        "elapsed_seconds": time.perf_counter() - started,
        "request_signature": request_signature,
    }


async def _fetch_versions(
    clients: list[httpx.AsyncClient], endpoints: list[str]
) -> dict[str, Any]:
    versions: dict[str, Any] = {}
    for client, endpoint in zip(clients, endpoints, strict=True):
        response = await client.get(f"{endpoint}/v1/version")
        response.raise_for_status()
        versions[endpoint] = response.json()
    return versions


def _load_checkpoint(path: Path, request_signature: str) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid checkpoint JSON at {path}:{line_number}") from error
            # Ignore stale predictions after the dataset, model, or prompts change.
            if value.get("request_signature") == request_signature:
                # A later retry supersedes an earlier failed row.
                completed[value["example_id"]] = value
    return completed


def _request_signature(model_revision: str, prompts: dict[str, Any], input_sha256: str) -> str:
    payload = json.dumps(
        {"model_revision": model_revision, "prompts": prompts, "input_sha256": input_sha256},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
