# LensAgent

LensAgent fits strong gravitational lenses with a multimodal language model and
`lenstronomy`. It searches physical smooth-lens models, refines the selected
model, and tests the image residuals for dark-matter subhalos.

The repository contains the workflows and prepared data used for:

- the 20-system SDSS sample and the J0946 system;
- 20 mock systems published as families A-D;
- five fixed-count multisubhalo mocks, FD1-FD3 and FR1-FR2.

The SDSS catalog contains 117 systems. Any listed system can be prepared and
run by its catalog identifier.

## Method

Each run has three stages:

1. **AFMS**, Autonomous Fitting-driven Model Selection, searches the allowed
   smooth-lens families with PSO initialization and LensAgent evolution.
2. **PRL**, Parameter Refinement Loop, improves the selected physical model and
   hands off the fit closest to reduced image chi-squared 1.
3. **RSI**, Residual-based Subhalo Inference, searches the PRL residual map.

RSI has two separate pathways. SDSS systems and the A-D mocks use the
single-subhalo pathway: it ranks up to ten pull-map candidates, fits each one
independently with PSO and LensAgent, and uses delta-BIC for the final candidate
comparison.
The FD and FR mocks use fixed-count multisubhalo RSI. It identifies and refines
candidate blobs, searches exact-K supports jointly, applies a tied-mass PSO
polish, and then refines all K subhalos jointly with LensAgent. K is specified
for each benchmark system; it is not inferred by this pathway.

All image likelihoods use

```text
variance = background_rms^2 + max(model, 0) / exposure_time
reduced chi-squared = chi-squared / (N - k)
```

where `N` is the number of fitted pixels and `k` includes nonlinear and solved
linear parameters. SDSS contaminant masks are applied to the likelihood. Images
shown to LensAgent and written to the result directories use asinh
normalization.

## Installation

Python 3.12 is required.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

Set a Requesty API key before running a fit:

```bash
export LENSAGENT_API_KEY="..."
```

The primary model is `vertex/google/gemini-3.1-pro-preview`; image analysis uses
`vertex/gemini-3.1-flash-lite`.

## Data

Prepared observations are in `data/observations`. Mock truth products are in
`data/benchmarks`, and SHA-256 manifests are in `data/manifests`. The packaged
SDSS observations cover the paper sample and J0946.

List a catalog and its local preparation status:

```bash
lensagent catalog --dataset sdss
lensagent catalog --dataset single-mock
lensagent catalog --dataset multisubhalo-mock
```

An SDSS entry not already packaged is prepared automatically when it is first
run. It can also be prepared explicitly:

```bash
lensagent prepare \
  --dataset sdss \
  --system 094656.68+100652.8
```

## Running LensAgent

Run J0946:

```bash
lensagent run \
  --dataset sdss \
  --system 094656.68+100652.8 \
  --output runs/094656.68+100652.8
```

Run the 20-system SDSS paper sample with four systems in parallel:

```bash
lensagent campaign \
  --dataset sdss \
  --paper-sample \
  --concurrency 4 \
  --output runs/sdss-paper
```

Run the A-D mocks or the fixed-count multisubhalo mocks:

```bash
lensagent campaign \
  --dataset single-mock \
  --paper-sample \
  --concurrency 4 \
  --output runs/single-mocks

lensagent campaign \
  --dataset multisubhalo-mock \
  --paper-sample \
  --concurrency 4 \
  --output runs/multisubhalo-mocks
```

Use repeated `--system` options for a subset, or `--all` for an entire catalog.
Running the same command with the same output directory resumes valid completed
stages and PSO archives. An output directory cannot be reused with different
observations, parameters, or workflow settings.

Each system directory contains `afms`, `prl`, and `rsi` artifacts, full model
traces, PSO records, fitted parameters, figures, `status.json`, and
`result.json`. `final_fit.png` is the selected final image model.

## Tests

```bash
python -m pytest
```

The tests verify packaged-data hashes, catalog membership, model-family
selection, the corrected likelihood and parameter count, tied-mass subhalo
evaluation, and the fixed-count candidate-ranking contract.

## Citation

```bibtex
@article{feng2026lensagent,
  title = {LensAgent: A Self Evolving Agent for Autonomous Physical Inference of Sub-galactic Structure},
  author = {Feng, Xiaotang and Wang, Zihan and Shu, Zilang and Kneib, Jean-Paul and Torr, Philip},
  year = {2026},
  eprint = {2604.03691},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2604.03691}
}
```

Machine-readable citation metadata is provided in `CITATION.cff`.
