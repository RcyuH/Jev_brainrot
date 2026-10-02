# Routing-conditional calibration reproduction

This repository reproduces the current draft of *Can We Trust Jev After
Routing?* before adding robustness analyses. It prepares Civil Comments and
GoEmotions, obtains four OpenJev probabilities per example, performs the
score-conditioned audit with paired bootstrap uncertainty, applies BH-FDR,
runs the random/oracle controls, and renders the paper tables and figures.

## Decisions made for underspecified details

All decisions live in `configs/reproduce.yaml`; its SHA-256 is written into
every run manifest.

- Civil Comments soft targets are binarized at `>= 0.5`.
- Routing selects `ceil(r * N)` rows. Scores are descending; `example_id` is
  the deterministic tie-break.
- Score bins are pooled downstream-score quantiles. Routed and non-routed
  examples use the same boundaries. Identical rounded scores are never split,
  so fewer than ten effective bins are possible.
- A bin is supported only with at least 50 routed and 50 non-routed examples.
- Conditional bin differences are weighted by routed count.
- The raw comparison is routed minus the full sample, following Appendix A.1.
- Each paired bootstrap resample re-ranks routing scores and reconstructs bins.
- The two-sided bootstrap sign p-value is
  `2 * (min(n_boot <= 0, n_boot >= 0) + 1) / (B + 1)`, capped at one. With
  2,000 samples its minimum is approximately 0.001, consistent with the
  draft's repeated `q=0.0011` after BH over 16 tests.
- Unsupported planned GoEmotions cells remain in the 12-test BH family with
  `p=1`.
- OpenJev uses the model card's online-FP8 vLLM recipe and calibration
  constants.

### Important data discrepancy

The pinned public `google/civil_comments` test parquet has 97,320 rows. It has
348 duplicate occurrences when all eight published fields are compared, and
458 duplicate texts. Neither matches the draft's statement that 88 rows were
removed. The default uses exact-row deduplication because it is deterministic
and does not selectively inspect target labels. The preparation manifest will
therefore report 96,972 rows. If the original 97,232-row source table becomes
available, point `paths.civil_input` at it and set deduplication to `none`, or
select an explicit `id` rule. The pipeline never silently deletes an arbitrary
88 rows just to match the stated count.

## Internet-connected preparation

Run this on Linux x86_64 with Python 3.12, matching the offline server:

```bash
chmod +x scripts/*.sh
scripts/download_online.sh /transfer/offline_bundle
```

The bundle includes pinned model/data snapshots, Python wheels, and checksums.
The model snapshot is about 55 GB. Verify a clean offline installation before
moving it to the server.

## Offline installation

```bash
scripts/install_offline.sh /transfer/offline_bundle .venv
source .venv/bin/activate

mkdir -p data/raw
ln -s /transfer/offline_bundle/data/civil_comments data/raw/civil_comments
ln -s /transfer/offline_bundle/data/go_emotions data/raw/go_emotions
```

The host needs a sufficiently recent NVIDIA driver for B200/CUDA 13.0. The
Python wheels provide their CUDA runtime; a full host CUDA toolkit is not
required for the prebuilt path.

## Run

Prepare datasets:

```bash
jevcal --config configs/reproduce.yaml prepare
```

In terminal 1, start one OpenJev replica per B200:

```bash
scripts/serve_two_b200.sh /transfer/offline_bundle/models/openjev
```

In terminal 2, run resumable inference and analysis:

```bash
jevcal --config configs/reproduce.yaml infer --dataset all
jevcal --config configs/reproduce.yaml analyze
```

Inference checkpoints each completed example in JSONL. Restarting the same
command resumes without repeating successful rows. For a short end-to-end
analysis smoke test, use `analyze --quick`; this does not produce paper-grade
confidence intervals.

Outputs are placed in:

- `data/prepared/`: normalized examples and provenance manifests;
- `artifacts/predictions/`: labels plus cached OpenJev probabilities;
- `results/tables/`: CSV and Parquet tables;
- `results/bootstrap/`: compressed bootstrap draws;
- `results/figures/`: the two main plots;
- `results/manifest.json`: hashes and the complete analysis protocol.

## Tests

```bash
pytest
```

The tests cover label preparation, deterministic routing, tied-score binning,
conditional weighting, bootstrap p-values, and BH correction without requiring
a GPU or model download.

