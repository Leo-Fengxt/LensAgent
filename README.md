# LensAgent

```bash
python -m funsearch.orchestrator \
    --concurrency 4 \
    --campaign-name "v8ef_batch-final-8" \
    --api-key "$OPENROUTER_API_KEY" \
    --catalog "/path/to/catalog.csv" \
    --obs-version v8expertfixed \
    --auto2 \
    --parallel-per-task 8 \
    --max-tasks 20 \
    --skip-tasks 1 \
    --resume
```
