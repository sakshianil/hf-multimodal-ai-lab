# HF Multimodal AI Lab

A hands-on Hugging Face multimodal AI lab covering **Transformers pipelines, AutoClasses, image captioning, speech-to-text, visual question answering, Diffusers, image generation, audio-to-text-to-image workflows, and Hugging Face inference endpoints**.

This repository is designed as a portfolio-ready learning lab: notebooks explain the concepts, `app.py` runs as a Hugging Face Space, and `screenshots/` documents working outputs.

## Live demo

- **Hugging Face Space:** https://huggingface.co/spaces/sakshianil/multimodal-ai-playground
- **GitHub repository:** https://github.com/sakshianil/hf-multimodal-ai-lab

Clone the Space:

```bash
git clone https://huggingface.co/spaces/sakshianil/multimodal-ai-playground
```

Clone the GitHub repo:

```bash
git clone https://github.com/sakshianil/hf-multimodal-ai-lab.git
```

## Project scope

This project demonstrates practical AI workflows across text, image, and audio:

- Transformers pipeline API
- AutoTokenizer, AutoModel, and AutoClasses
- Image captioning and multimodal inference
- Speech-to-text from audio
- Text-to-image generation with Diffusers
- Audio → transcription → image generation workflow
- Hugging Face Inference API / endpoint usage
- Hugging Face Spaces deployment with Gradio

## Demo highlight: audio to text to image

The screenshot below shows a multimodal chain using a personal Hindi singing audio sample. The audio is first converted into Hindi text, and then the text is used to generate an image output.

> Audio input → speech transcription → generated image

![Audio to text to image workflow](screenshots/audio-to-text-to-image-workflow.svg)

## Hugging Face Space app

The root `app.py` is Space-ready and contains four tabs:

1. **Image Captioning** — upload an image and generate a caption.
2. **Speech to Text** — upload audio and transcribe it with a lightweight Whisper model.
3. **Text Pipeline** — run a CPU-friendly Transformers text classification pipeline.
4. **Audio → Text → Image concept** — explain the workflow used in the notebook experiments.

The live Space is intentionally CPU-friendly. Diffusers-based image generation can be added later if the Space is upgraded to GPU.

## Repository structure

```text
hf-multimodal-ai-lab/
├── app.py
├── notebooks/
│   └── README.md
├── app/
│   ├── app.py
│   └── requirements.txt
├── docs/
│   ├── project_overview.md
│   └── hf_spaces_plan.md
├── screenshots/
│   ├── audio-to-text-to-image-workflow.svg
│   └── .gitkeep
├── README.md
├── requirements.txt
└── .gitignore
```

## Learning modules

| Module | Focus | Notebook |
|---|---|---|
| Transformers foundations | AutoClasses and model loading | `01_autoclasses.ipynb` |
| Pipelines | High-level Hugging Face pipeline API | `02_pipelines.ipynb` |
| Multimodal inference | Image/audio/text workflows | `03_multimodal_hf.ipynb` |
| Hosted inference | Hugging Face inference endpoint usage | `04_hf_inference_endpoint.ipynb` |
| Diffusers | Generative image models | `05_diffusers.ipynb` |
| Image generation | Text-to-image experiments | `06_image_generation.ipynb` |
| Audio-text-image | Speech-to-text and generated image chaining | `07_audio_text_image.ipynb` |

## Recommended showcase strategy

Use **GitHub** as the main portfolio repository because it can hold notebooks, documentation, screenshots, and the full learning sequence.

Use **Hugging Face Spaces** for the clean live Gradio demo.

## How to sync GitHub code into the Hugging Face Space

After cloning the Hugging Face Space locally, copy these files from the GitHub repo into the Space repo:

```text
app.py
requirements.txt
README.md
screenshots/
docs/
```

Then commit and push to the Space:

```bash
cd multimodal-ai-playground
git add .
git commit -m "Add multimodal AI playground app"
git push
```

## Why this project matters

This lab shows the journey from using simple Hugging Face pipelines to building practical multimodal AI workflows. It is useful for demonstrating applied AI engineering skills across model loading, inference, generative AI, and deployment planning.

## Suggested GitHub topics

```text
huggingface, transformers, diffusers, multimodal-ai, gradio, speech-to-text, image-generation, text-to-image, ai-engineering, notebooks
```

## Next improvements

- Add screenshots for each notebook output
- Push the same `app.py` and `requirements.txt` to the Hugging Face Space
- Add full cleaned notebooks
- Add model cards and workflow diagrams
- Add lightweight CPU-friendly demos and GPU-only notes
