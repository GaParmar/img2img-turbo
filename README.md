# img2img-turbo — IEC Mode (Interactive Evolutionary Computation)

[**Paper**](https://arxiv.org/abs/2403.12036) | [**Original Repo**](https://github.com/GaParmar/img2img-turbo) | Branch: `feature/iec`

This branch extends `img2img-turbo` with an **Interactive Evolutionary Computation (IEC)** interface for sketch-to-image generation. Instead of tweaking parameters manually, you evolve images by selecting favorites and mixing them — like a genetic algorithm driven by human preference.

---

## How It Works

### Workflow
```
Draw sketch → Generate (16 images) → Select favorites → Mix → Repeat
```

1. **Draw** a sketch on the canvas.
2. **Run** — generates 16 images in parallel, each with a randomly sampled seed and sketch-guidance value `r`.
3. **Select** the images you like by clicking them (they appear in the Selected Gallery).
4. **Mix** — evolves a new generation of 16 images from your selection using a genetic-style algorithm.
5. Repeat steps 3–4 until satisfied, then click **Chose** to finalize.

### Genetic Mixing Strategy (`mix()`)

Given `n` selected images, the next 16 are filled as:

| Slot range | Strategy |
|---|---|
| `0 … n-1` | Carry selected params directly |
| `n … n+3` | Mutation: random `r`, seed from selected pool |
| `n+4 … n+7` | Crossover: average `r`, fully random seed |
| `n+8 … 15` | Exploration: fully random `r` and seed |

All 16 slots are shuffled before generation to prevent position bias.

### Sampling

- **Seed**: uniform random integer `[0, MAX_SEED]`
- **Sketch guidance `r`**: sampled from `N(μ=0.1, σ=0.3)`, clipped to `(0.01, 0.99)`

---

## Getting Started

**Environment setup**

```bash
conda env create -f environment.yaml
conda activate img2img-turbo
```

Or with pip:

```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Launch the IEC Gradio demo**

```bash
python gradio_sketch2image_iec.py
```

The demo will start locally and print a public share link (via `share=True`).

---

## UI Overview

| Element | Description |
|---|---|
| Canvas | 512×512 sketch input with pencil / eraser tools |
| Prompt | Text prompt describing the desired output |
| Style | 10 preset styles; defaults to **Anime** |
| Output Gallery | 16 generated images per run |
| Selected Gallery | Images you've clicked to select |
| **Mix** button | Evolve a new generation from selected images |
| **Chose** button | Finalize selection and display result |
| Mix count | Counter showing how many Mix rounds have been run |

---

## Available Styles

`Cinematic`, `3D Model`, `Anime` (default), `Digital Art`, `Photographic`, `Pixel art`, `Fantasy art`, `Neonpunk`, `Manga`, `No Style`

---

## Acknowledgment

Built on top of [img2img-turbo](https://github.com/GaParmar/img2img-turbo) (Parmar et al., arXiv 2403.12036) and [SD-Turbo](https://huggingface.co/stabilityai/sd-turbo) — see its [LICENSE](https://huggingface.co/stabilityai/sd-turbo/blob/main/LICENSE).
