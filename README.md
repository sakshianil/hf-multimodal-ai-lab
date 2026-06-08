# HF Multimodal AI Lab

A portfolio-ready Hugging Face and Gradio project for building practical **multimodal AI workflows** across audio, text, image generation prompts, Transformers, Diffusers, and hosted inference.

This repository currently powers a live Hugging Face Space focused on:

> **Audio upload / recording → local speech-to-text transcription → LLM interpretation → image-generation prompt**

The project is designed as a growing multimodal AI playground. The current version works for audio-to-text-to-image-prompt workflows, and upcoming versions will add richer text workflows, connectors/apps, image upload, audio/image download, and more end-to-end multimodal actions.

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

## Current working app

The live app currently supports an **audio-to-text-to-image-prompt** workflow.

### What it does now

1. User uploads or records audio.
2. The app converts the audio into a clean 16 kHz mono WAV file using `ffmpeg`.
3. Local Whisper Tiny transcribes the audio inside the Space.
4. The user may optionally provide their own OpenAI or OpenRouter key.
5. The transcription is sent to the selected LLM provider for interpretation.
6. The app returns:
   - local Whisper transcription
   - structured LLM analysis
   - cinematic image-generation prompt

### Key handling

No API key is hardcoded or stored in this repository. The app asks the user to paste their own OpenAI or OpenRouter key at runtime. The key is only used for the active request.

OpenRouter keys are detected automatically when they start with `sk-or-`, and the app routes them to the OpenRouter API instead of OpenAI.

### Current default provider setup

```text
Provider: OpenRouter
Model: openai/gpt-4o-mini
Transcription: local Whisper Tiny
```

This avoids sending OpenRouter keys to OpenAI transcription endpoints and keeps audio transcription CPU-friendly.

## Demo highlight: audio to text to image

The screenshot below shows a multimodal chain using a personal Hindi singing audio sample. The audio is first converted into text, and then the text is used as the creative signal for an image-generation prompt/output.

> Audio input → speech transcription → LLM interpretation → image-generation prompt

![Audio to text to image workflow](screenshots/audio-to-text-to-image-workflow.svg)

## Project scope

This project demonstrates practical AI workflows across text, image, and audio:

- Audio upload and recording with Gradio
- Audio normalization with `ffmpeg`
- Local speech-to-text using Whisper Tiny
- User-provided OpenAI/OpenRouter API keys
- OpenRouter-compatible LLM calls through the OpenAI client
- Prompt generation for text-to-image workflows
- Transformers pipeline API
- AutoTokenizer, AutoModel, and AutoClasses
- Image captioning and multimodal inference experiments
- Diffusers and text-to-image generation experiments
- Hugging Face Spaces deployment

## Hugging Face Space app

The root `app.py` is Space-ready and currently uses a stable Gradio Blocks layout.

The interface includes:

- **Upload or record audio** input
- **Instruction** box
- **API settings** accordion
- Provider selection: OpenAI or OpenRouter
- Runtime API key field
- Model field
- Output boxes for transcription, LLM analysis, and image-generation prompt

The live Space is intentionally CPU-friendly. Heavy Diffusers-based image generation can be added later if the Space is upgraded to GPU.

## Planned next features

Upcoming versions may include:

- Text-only prompt workflows
- Image upload and image captioning
- Visual question answering
- Audio upload and audio download
- Image prompt download
- Generated image download
- Connector/app integrations
- RAG over uploaded files
- Multimodal memory and project buckets
- Better UI for creators, researchers, and wellness/brand workflows
- Optional GPU-based image generation

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

## How to sync GitHub code into the Hugging Face Space

The most reliable update method used for this project is `hf upload`:

```bash
curl -L https://raw.githubusercontent.com/sakshianil/hf-multimodal-ai-lab/main/app.py -o app.py
curl -L https://raw.githubusercontent.com/sakshianil/hf-multimodal-ai-lab/main/requirements.txt -o requirements.txt

hf upload sakshianil/multimodal-ai-playground app.py app.py --repo-type space
hf upload sakshianil/multimodal-ai-playground requirements.txt requirements.txt --repo-type space
```

After uploading, restart or factory reboot the Hugging Face Space.

## Why this project matters

This lab shows the journey from simple Hugging Face pipelines to practical multimodal AI engineering. It demonstrates model loading, local transcription, LLM routing, user-key handling, prompt generation, and deployment on Hugging Face Spaces.

It is useful as a portfolio project for applied AI, multimodal product thinking, AI automation, and future creator/research assistant workflows.

## Suggested GitHub topics

```text
huggingface, transformers, diffusers, multimodal-ai, gradio, speech-to-text, whisper, openrouter, audio-processing, image-generation, text-to-image, ai-engineering, notebooks
```

## Next improvements

- Add screenshots of the live Hugging Face Space UI
- Add full cleaned notebooks
- Add downloadable transcription and prompt files
- Add image upload and image captioning
- Add optional generated image output when GPU is available
- Add model cards and workflow diagrams
- Add connector/app integrations
- Add RAG and document upload workflows
