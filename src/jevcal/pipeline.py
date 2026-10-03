from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import config_digest
from .io import atomic_write_json, sha256_file
from .plotting import plot_matrix, plot_primary
from .reporting import generate_publication_tables
from .statistics import aggregate_controls, analyze_family, build_experiments


def run_analysis(config: dict[str, Any], quick: bool = False) -> dict[str, Any]:
    paths = config["paths"]
    results_dir = Path(paths["results_dir"])
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"
    bootstrap_dir = results_dir / "bootstrap"
    for directory in (tables_dir, figures_dir, bootstrap_dir):
        directory.mkdir(parents=True, exist_ok=True)

    analysis = dict(config["analysis"])
    if quick:
        analysis["bootstrap_resamples"] = min(50, int(analysis["bootstrap_resamples"]))
        analysis["random_control_subsets"] = min(20, int(analysis["random_control_subsets"]))
    seed = int(config["seed"])

    civil_path = Path(paths["predictions_dir"]) / "civil_comments.parquet"
    go_path = Path(paths["predictions_dir"]) / "goemotions.parquet"
    civil = pd.read_parquet(civil_path)
    go = pd.read_parquet(go_path)

    primary_experiments = build_experiments(
        config["experiments"]["civil_primary"], analysis["primary_rates"]
    )
    matrix_experiments = build_experiments(
        config["experiments"]["civil_matrix"], analysis["matrix_rates"]
    )
    go_experiments = build_experiments(
        config["experiments"]["goemotions"], analysis["primary_rates"]
    )

    primary, primary_bootstrap = analyze_family(
        civil, primary_experiments, analysis, seed + 101
    )
    matrix, matrix_bootstrap = analyze_family(civil, matrix_experiments, analysis, seed + 202)
    go_results, go_bootstrap = analyze_family(go, go_experiments, analysis, seed + 303)
    controls = aggregate_controls(
        civil,
        downstreams=["identity_attack", "obscene"],
        rates=analysis["primary_rates"],
        subsets=int(analysis["random_control_subsets"]),
        seed=seed + 404,
    )

    outputs: dict[str, str] = {}
    for name, table in [
        ("civil_primary", primary),
        ("civil_matrix", matrix),
        ("goemotions", go_results),
        ("aggregate_controls", controls),
    ]:
        csv_path = tables_dir / f"{name}.csv"
        parquet_path = tables_dir / f"{name}.parquet"
        table.to_csv(csv_path, index=False)
        table.to_parquet(parquet_path, index=False)
        outputs[f"{name}_csv"] = str(csv_path)
        outputs[f"{name}_parquet"] = str(parquet_path)

    for name, values in [
        ("civil_primary", primary_bootstrap),
        ("civil_matrix", matrix_bootstrap),
        ("goemotions", go_bootstrap),
    ]:
        path = bootstrap_dir / f"{name}.npz"
        np.savez_compressed(path, **values)
        outputs[f"{name}_bootstrap"] = str(path)

    primary_figure = figures_dir / "figure1_primary.png"
    matrix_figure = figures_dir / "figure2_matrix.png"
    plot_primary(primary, primary_figure)
    plot_matrix(matrix, matrix_figure)
    outputs["figure1"] = str(primary_figure)
    outputs["figure2"] = str(matrix_figure)
    outputs.update(generate_publication_tables(config))

    manifest = {
        "config_sha256": config_digest(config),
        "quick_mode": quick,
        "analysis_protocol": analysis,
        "inputs": {
            "civil_comments": {"path": str(civil_path), "sha256": sha256_file(civil_path)},
            "goemotions": {"path": str(go_path), "sha256": sha256_file(go_path)},
        },
        "outputs": outputs,
    }
    atomic_write_json(manifest, results_dir / "manifest.json")
    return manifest
