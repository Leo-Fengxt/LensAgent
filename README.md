# LensAgent

LensAgent fits strong gravitational lenses with a multimodal language model and
`lenstronomy`. It searches physical smooth-lens models, refines the selected
model, and tests the image residuals for dark-matter subhalos.

The repository contains workflows and prepared data for:

- the 20-system SDSS sample and the J0946 system;
- 20 SLACS ACS/F814W observations and J0946;
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

RSI has single-subhalo and fixed-count multisubhalo pathways.

With `observations: sdss`, single-subhalo RSI fits up to ten pull-map candidates
with the smooth model fixed. With `observations: HST`, each candidate has two
stages: jointly fit the macro lens and subhalo with source geometry fixed, then
fix that branch's macro lens and jointly fit source geometry and subhalo.
Foreground-light geometry stays fixed; light amplitudes are solved at every
evaluation. Subhalo centers can move throughout the image footprint.
An independent no-subhalo model goes through both stages. Final selection uses
raw image delta-BIC, with a detection threshold of 6. Benchmark truth is not used
to select single-subhalo candidates.

The A-D mocks use the HST workflow. AFMS permits nine smooth-model families for
real observations and eight for mocks, excluding the MGE mass family for mocks.
The relative macro-strength floor is 0.1 for components parameterized by
Einstein radius. Agent refinement targets reduced chi-squared closest to 1,
including from below, while the RSI evidence comparison uses raw chi-squared.

The FD and FR mocks use fixed-count multisubhalo RSI. It identifies and refines
candidate blobs, searches exact-K supports jointly, applies a tied-mass PSO
polish, and then refines all K subhalos jointly with LensAgent. K is specified
for each benchmark system; it is not inferred by this pathway.

Image likelihoods use

```text
reduced chi-squared = chi-squared / (N - k)
```

where `N` is the number of fitted pixels and `k` includes nonlinear and solved
linear parameters. HST uses its fixed total-noise map, including source noise.
SDSS uses `background_rms^2 + max(model, 0) / exposure_time`. Mock RMS maps contain
all simulated noise terms; their fitting exposure is effectively infinite to
avoid adding source noise twice. Mock generation and all HST fitting use native
pixel sampling, without supersampling. Contaminant masks enter the likelihood. Images
shown to LensAgent and written to the result directories use asinh
normalization.

HST and mock budgets are 800 primary calls for AFMS and 150 for PRL. Each RSI
candidate, and the null model, has six PSO starts of 100 particles and 250 steps
before each of its two agent stages. Each stage permits 150 primary calls.
The NFW mass cap is `1e11` solar masses for HST observations and `1e10` for mocks.

## Installation

Linux and Python 3.12 are required.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

HST and mock workflows use `z-ai/glm-5.3-flash` through OpenRouter for both the
fitting and image-description agents, with high reasoning:

```bash
export OPENROUTER_API_KEY="..."
```

SDSS uses Requesty by default:

```bash
export LENSAGENT_API_KEY="..."
```

For SDSS, the primary model is `vertex/google/gemini-3.1-pro-preview`; image analysis uses
`vertex/gemini-3.1-flash-lite`.

OpenRouter and the native Gemini API are also supported:

```bash
export OPENROUTER_API_KEY="..."
lensagent run \
  --provider openrouter \
  --dataset sdss \
  --system 094656.68+100652.8 \
  --output runs/094656.68+100652.8

export GEMINI_API_KEY="..."
lensagent run \
  --provider gemini \
  --dataset sdss \
  --system 094656.68+100652.8 \
  --output runs/094656.68+100652.8-gemini
```

For SDSS, OpenRouter defaults to `google/gemini-3.1-pro-preview` and
`google/gemini-3.1-flash-lite`. Native Gemini uses the corresponding model names
without the `google/` prefix. Override either model with `--primary-model` or
`--auxiliary-model`. The same options work with `lensagent campaign` and are
passed to every campaign process.

Provider details are documented by
[OpenRouter](https://openrouter.ai/docs/api/api-reference/chat/send-chat-completion-request)
and the [Gemini API](https://ai.google.dev/api/generate-content).

## Data

Prepared observations are in `data/observations`. Mock truth products are in
`data/benchmarks`, and SHA-256 manifests are in `data/manifests`. The packaged
SDSS and HST observations each include 20 SLACS systems and J0946. HST cutouts
are 150 x 150 pixels at 0.05 arcsec per pixel; mocks are 120 x 120 pixels at the
same pixel scale. The catalogs use full system identifiers, not task indices.

List a catalog and its local preparation status:

```bash
lensagent catalog --dataset sdss
lensagent catalog --observations HST
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

To prepare HST inputs from calibrated archive exposures, install the preparation
dependencies and create a separate data root:

```bash
python -m pip install -e ".[hst]"
lensagent prepare --observations HST --system 094656.68+100652.8 --data-root data-prepared
```

Preparation matches science and variance drizzling, builds an effective PSF,
and saves sky, coverage, contaminant-mask, and exposure-consistency diagnostics.
Inspect `preparation.png` and the FITS diagnostics in the system's processed
directory, complete `review_template.json`, then approve the observation:

```bash
lensagent approve \
  --draft data-prepared/hst/processed/094656.68+100652.8/observation.draft.npz \
  --review data-prepared/hst/processed/094656.68+100652.8/review_template.json \
  --output data-prepared/observations/hst/094656.68+100652.8.npz
```

`--regions` supplies explicit lens-centering and mask regions keyed by system
identifier; the catalog's region definitions are used by default. Source products and spectroscopic inputs are listed in
`src/lensagent/resources/manifests/hst_sources.json`. Packaged observations are
ready to load and do not need to be downloaded or prepared again.

## Running LensAgent

Run J0946 with HST observations:

```bash
lensagent run \
  --observations HST \
  --system 094656.68+100652.8 \
  --output runs/hst/094656.68+100652.8
```

Run the 20-system SDSS paper sample with four systems in parallel:

```bash
lensagent campaign \
  --dataset sdss \
  --paper-sample \
  --concurrency 4 \
  --output runs/sdss-paper
```

For the HST sample, use `--observations HST` instead of `--dataset sdss` and a
separate output directory. Both observation modes are recorded in each run's
`configuration.json`.

Run the A-D mocks or the fixed-count multisubhalo mocks:

```bash
lensagent campaign \
  --dataset single-mock \
  --paper-sample \
  --concurrency 8 \
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
HST single-subhalo RSI stores separate `rsi/lens` and `rsi/source` directories,
each with a null branch, candidate branches, PSO chains, and agent evaluations.
Saved observations and configurations guard against incompatible resumes.

## Tests

```bash
python -m pytest
```

Tests cover data hashes, native-pixel mock generation, HST noise and masks,
model families, both RSI stage handoffs, parameter counts, agent refinement,
provider payloads, and fixed-count candidate selection. Tests use no paid APIs.

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
