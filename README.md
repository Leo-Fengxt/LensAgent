# lensing-public-core

Minimal public code snapshot for reproducing the core `funsearch.orchestrator`
workflow used for the classic `v8ef_batch-final-8` run.

This export is intentionally code-only:

- included: the core `funsearch/` package and its direct runtime dependencies
- excluded: API keys, OAuth tokens, Google Drive auth, logs, run artifacts,
  cached outputs, observation pickles, and catalog data

## Included Files

- `funsearch/`
- `Cutout.py`
- `evaluate.py`
- `kinematic_api.py`
- `observation.py`
- `profiles.py`
- `requirements.txt`

## Required External Data

To run the classic pipeline, provide these items locally next to this repo:

- `observations_v8expertfixed/` at the repo root, containing the observation
  `.pkl` files for `--obs-version v8expertfixed`
- a catalog CSV passed through `--catalog`, or pointed to by
  `LENSING_CATALOG`

The catalog loader expects these columns:

- `SDSS Name`
- `RA`
- `DEC`
- `z_FG`
- `z_BG`
- `Sigma`
- `Sigma_err`
- optional: `background_rms_i`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Classic Run Command

```bash
export OPENROUTER_API_KEY="your_api_key_here"

python -m funsearch.orchestrator \
    --concurrency 4 \
    --campaign-name "v8ef_batch-final-8" \
    --api-key "$OPENROUTER_API_KEY" \
    --catalog "/absolute/path/to/catalog.csv" \
    --obs-version v8expertfixed \
    --auto2 \
    --parallel-per-task 8 \
    --max-tasks 20 \
    --skip-tasks 1 \
    --resume
```

## Drive Uploads

`token.json` is intentionally not included. Without it, Drive upload is
automatically disabled by the orchestrator. If you want to make the public run
path explicit, add `--no-drive`.

If you intentionally enable Drive uploads in your own private environment, you
must provide your own OAuth token and Google client dependencies.

## Public-Snapshot Notes

- This public copy uses a neutral default catalog path:
  `./catalog.csv`, or the `LENSING_CATALOG` environment variable.
- No run outputs, plots, logs, credentials, or dataset files are bundled.
