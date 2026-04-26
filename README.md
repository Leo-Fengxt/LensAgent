# LensAgent: A Self Evolving Agent for Autonomous Physical Inference of Sub-galactic Structure

This is an archival version of our research, and it is not intended nor recommended for production. A production-ready version is on our roadmap and will be coming soon!

```bash
python download_all.py \
    --catalog "catalog.csv" \
    --start 0 \
    --end 117

python regenerate_pkls.py \
    --obs-dir observations \
    --out-dir observations_v8expfixed \
    --fits-dir fits_cache \
    --exp-catalog "catalog_final (1)_with_rms.csv" \
    --start 0 \
    --end 117
```

```bash
python -m lensagent.orchestrator \
    --concurrency 4 \
    --campaign-name "v8ef_batch-final-8" \
    --api-key "$OPENROUTER_API_KEY" \
    --catalog "catalog_final (1).csv" \
    --shuffle \
    --parallel-per-task 8 \
    --max-tasks 20 \
    --skip-tasks 1 \
    --resume
```
