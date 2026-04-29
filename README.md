# LensAgent: A Self Evolving Agent for Autonomous Physical Inference of Sub-galactic Structure

<p align="center">
  <a href="mailto:xiaotang.feng@stcatz.ox.ac.uk">Xiaotang Feng</a><sup>1,2,*,&dagger;</sup> ·
  <a href="mailto:zihan.wang@queens.ox.ac.uk">Zihan Wang</a><sup>1,*,&dagger;</sup> ·
  <a href="mailto:zilang.shu@pmb.ox.ac.uk">Zilang Shu</a><sup>1</sup> ·
  <a href="mailto:jean-paul.kneib@epfl.ch">Jean-Paul Kneib</a><sup>3</sup> ·
  <a href="mailto:philip.torr@eng.ox.ac.uk">Philip Torr</a><sup>2,&dagger;</sup>
</p>

<p align="center">
  <sup>1</sup> Department of Physics, University of Oxford ·
  <sup>2</sup> Department of Engineering Science, University of Oxford ·
  <sup>3</sup> Institute of Physics, Laboratory of Astrophysics, EPFL
</p>

<p align="center">
  <sup>*</sup> Equal contribution. &nbsp; <sup>&dagger;</sup> Corresponding authors.
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2604.03691"><img src="https://img.shields.io/badge/arXiv-2604.03691-b31b1b.svg" alt="arXiv"></a>
</p>

---

## Abstract

Probing dark matter distribution on sub-galactic scales is essential for testing the Cold Dark Matter ($\Lambda$CDM) paradigm. Strong gravitational lensing, as one of the most powerful approach by far, provides a direct, purely gravitational probe of these substructures. However, extracting cosmological constraints is severely bottlenecked by the mass-sheet degeneracy (MSD) and the unscalable nature of manual and neural-network modeling. Here, we introduce LensAgent, a pioneering training-free, large language model (LLM)-driven agentic framework for the autonomous physical inference of mass distributions. Operating as an autonomous scientific agent, LensAgent couples high-level logical reasoning with deterministic physical modeling tools, demonstarting successful reconstruction of mass distribution in SLACS Grade A strong lensing systems. This self-evolving architecture enables the robust extraction of sub-galactic substructures at scale, unlocking the cosmological potential of upcoming wide-field surveys such as the Rubin Observatory (LSST) and Euclid.

---

LensAgent is the public code release accompanying our manuscript. It provides a reproducible workflow for autonomous strong gravitational lens modeling, kinematic validation, and sub-galactic dark matter substructure inference.

The project explores how a self-evolving, tool-using large language model can work alongside deterministic physics code to fit strong lens systems, use stellar kinematics to help break the mass-sheet degeneracy, and search residual structure for evidence of dark matter subhalos.

Each cycle combines high-level reasoning over lens-model parameters, deterministic image and kinematic evaluation, physicality checks based on the Poisson equation, and an evolving proposal memory that reuses strong fits as context for later search.

This public release is centered on the paper-aligned observation bundles and the current AFMS, PRL, and RSI workflow.

## Pipeline Overview

The public code follows the same three-stage workflow described in the paper:

- **AFMS**: *Autonomous Fitting-driven Model Selection*  
  LensAgent explores candidate macro-model families, seeded by PSO and allocated budget through a bandit-style scheduler.

- **PRL**: *Parameter Refinement Loop*  
  After AFMS identifies the strongest family, the agent refines the fit at higher numerical precision while retaining the accumulated search memory.

- **RSI**: *Residual-based Subhalo Inference*  
  The converged residuals are converted into pull maps, candidate subhalo locations are identified, and LensAgent reruns on the perturbed model to test localized NFW substructure.

Together, these stages provide a training-free workflow for **strong gravitational lensing**, **dark matter subhalo detection**, and **autonomous physical inference**.

## Repository Contents

- `download_all.py`  
  Downloads the FITS frames and PSF data needed to reconstruct the observation bundles.

- `regenerate_pkls.py`  
  Builds the observation bundles used for reproduction.

- `lensagent/`  
  The core agentic pipeline, including orchestration, scoring, prompting, proposal databases, AFMS/PRL/RSI logic, and the OpenAI-compatible chat-completions client.

- `evaluate.py`, `observation.py`, `kinematic_api.py`  
  Deterministic physical modeling and evaluation utilities used by the agent.

## Installation

Create a clean Python environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Prepare the Data

The reproduction workflow assumes a catalog of target systems, downloaded imaging products, and regenerated observation bundles in the paper's format.

```bash
python download_all.py \
    --catalog "catalog_with_rms.csv"

python regenerate_pkls.py \
    --obs-dir observations \
    --out-dir observations_output \
    --fits-dir fits_cache \
    --exp-catalog "catalog_with_rms.csv"
```

After this step, the regenerated observation bundles will live under `observations_output/`.

## Run LensAgent

The command below reproduces the paper-aligned run. It launches the AFMS + PRL + RSI pipeline across a shuffled subset of catalog systems and resumes any previously completed tasks within the same campaign directory.

```bash
python -m lensagent.orchestrator \
    --concurrency 4 \
    --campaign-name "run_batch-final" \
    --api-key "$OPENROUTER_API_KEY" \
    --catalog "catalog_with_rms.csv" \
    --obs-dir observations_output \
    --rsi-mode single \
    --shuffle \
    --parallel-per-task 8 \
    --max-tasks 20 \
    --skip-tasks 1 \
    --resume
```

`--rsi-mode single` (default) fits the strongest residual candidate as a single subhalo, matching the paper's reported runs. Pass `--rsi-mode multi` to let the agent fit several candidates jointly (controlled by `--n-subhalos`).

### Optional: Use a Different OpenAI-Compatible Endpoint

Requesty is the default provider in this public release. If you prefer other OpenAI-compatible chat-completions endpoint, pass `--api-base-url` (or set `LENSAGENT_API_BASE_URL`):

```bash
python -m lensagent.orchestrator \
    --concurrency 4 \
    --campaign-name "run_batch-final" \
    --api-key "$OPENAI_API_KEY" \
    --api-base-url "https://api.openai.com/v1/chat/completions" \
    --model "gpt-5" \
    --catalog "catalog_with_rms.csv" \
    --shuffle \
    --parallel-per-task 8 \
    --max-tasks 20 \
    --skip-tasks 1 \
    --resume
```

## Output Structure

Campaign outputs are written under:

```text
runs/<campaign-name>/
```

Each task stores its intermediate and final artifacts in:

- `afms/`
- `afms/prl/`
- `rsi/`

along with `status.json`, `task.log`, and the campaign-level `orchestrator.log`.

## Important Notes

- This repository is focused on reproducing the paper's public pipeline and results.
- The public code path supports the given observation format only.
- `--shuffle` selects tasks in random order from the catalog-matched observation pool; `--resume` allows interrupted campaigns to continue safely within the same campaign directory.

## Citation


```bibtex
@misc{feng2026lensagentselfevolvingagent,
  title={LensAgent: A Self Evolving Agent for Autonomous Physical Inference of Sub-galactic Structure},
  author={Xiaotang Feng and Zihan Wang and Zilang Shu and Jean-Paul Kneib and Philip Torr},
  year={2026},
  eprint={2604.03691},
  archivePrefix={arXiv},
  primaryClass={astro-ph.GA},
  url={https://arxiv.org/abs/2604.03691},
}
```

## License

This repository is released under the BSD 3-Clause license. See `LICENSE` for details.
