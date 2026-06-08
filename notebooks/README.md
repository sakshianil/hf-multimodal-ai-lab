# Notebooks

This folder organizes the Hugging Face multimodal learning notebooks.

## Notebook sequence

| Order | Notebook | Focus |
|---|---|---|
| 01 | `01_autoclasses.ipynb` | AutoTokenizer, AutoModel, and Hugging Face AutoClasses |
| 02 | `02_pipelines.ipynb` | High-level `pipeline()` API for common NLP and multimodal tasks |
| 03 | `03_multimodal_hf.ipynb` | Multimodal workflows across text, image, and audio |
| 04 | `04_hf_inference_endpoint.ipynb` | Hugging Face inference endpoint/API usage |
| 05 | `05_diffusers.ipynb` | Diffusers and generative image models |
| 06 | `06_image_generation.ipynb` | Text-to-image generation experiments |
| 07 | `07_audio_text_image.ipynb` | Audio transcription followed by image generation |

## Recommended cleanup before committing full notebooks

Before pushing large notebooks, clear heavy cell outputs when possible:

```bash
jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
```

This keeps the repository lightweight and easier to review.

## Current showcase

The first visible demo is documented in the main README:

```text
Hindi singing audio → speech-to-text transcription → generated image output
```

Screenshots and workflow visuals are stored in `screenshots/`.
