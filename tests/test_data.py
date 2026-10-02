from pathlib import Path

import pandas as pd

from jevcal.data import prepare_civil_comments, prepare_goemotions


def _config(civil_rows: int, go_rows: int) -> dict:
    return {
        "data": {
            "civil_comments": {
                "expected_input_rows": civil_rows,
                "label_threshold": 0.5,
                "deduplication": {
                    "mode": "exact",
                    "columns": [
                        "text", "toxicity", "severe_toxicity", "obscene",
                        "threat", "insult", "identity_attack", "sexual_explicit",
                    ],
                    "keep": "first",
                },
            },
            "goemotions": {"expected_input_rows": go_rows},
        }
    }


def test_prepare_civil_binarizes_and_deduplicates(tmp_path: Path):
    row = {
        "text": "same",
        "toxicity": 0.5,
        "severe_toxicity": 0.0,
        "obscene": 0.0,
        "threat": 0.0,
        "insult": 0.49,
        "identity_attack": 0.8,
        "sexual_explicit": 0.0,
    }
    input_path = tmp_path / "civil.parquet"
    output_path = tmp_path / "civil-prepared.parquet"
    pd.DataFrame([row, row]).to_parquet(input_path, index=False)
    manifest = prepare_civil_comments(input_path, output_path, _config(2, 1))
    output = pd.read_parquet(output_path)
    assert manifest["duplicates_removed"] == 1
    assert output["y_toxicity"].tolist() == [1]
    assert output["y_insult"].tolist() == [0]


def test_prepare_goemotions_maps_multilabel_ids(tmp_path: Path):
    input_path = tmp_path / "go.parquet"
    output_path = tmp_path / "go-prepared.parquet"
    pd.DataFrame([{"text": "x", "labels": [2, 11, 25]}]).to_parquet(input_path, index=False)
    prepare_goemotions(input_path, output_path, _config(1, 1))
    output = pd.read_parquet(output_path)
    assert output[["y_anger", "y_disgust", "y_fear", "y_sadness"]].iloc[0].tolist() == [1, 1, 0, 1]

