from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import gradio as gr

APP_TITLE = "Multimodal AI Playground"
GITHUB_REPO = "https://github.com/sakshianil/hf-multimodal-ai-lab"
HF_SPACE = "https://huggingface.co/spaces/sakshianil/multimodal-ai-playground"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


@lru_cache(maxsize=1)
def get_image_captioner():
    from transformers import pipeline

    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")


@lru_cache(maxsize=1)
def get_speech_recognizer():
    from transformers import pipeline

    return pipeline("automatic-speech-recognition", model="openai/whisper-tiny")


@lru_cache(maxsize=1)
def get_text_classifier():
    from transformers import pipeline

    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")


def caption_image(image: Any) -> str:
    if image is None:
        return "Please upload an image first."
    try:
        captioner = get_image_captioner()
        result = captioner(image)
        if isinstance(result, list) and result:
            return result[0].get("generated_text", str(result[0]))
        return str(result)
    except Exception as exc:
        return f"Image captioning failed: {exc}"


def transcribe_audio(audio_path: str | None) -> str:
    if not audio_path:
        return "Please upload an audio file first."
    try:
        recognizer = get_speech_recognizer()
        result = recognizer(audio_path)
        if isinstance(result, dict):
            return result.get("text", str(result))
        return str(result)
    except Exception as exc:
        return f"Speech-to-text failed: {exc}"


def classify_text(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "Please enter text first."
    try:
        classifier = get_text_classifier()
        result = classifier(text)
        return str(result)
    except Exception as exc:
        return f"Text classification failed: {exc}"


def explain_audio_to_image_workflow(transcription: str) -> str:
    transcription = (transcription or "").strip()
    if not transcription:
        transcription = "आगो जब तुम साजना आओगा फूल खिलेंगे बरसिया सावन झूम झूम"

    return f"""
## Audio → Text → Image workflow

This project demonstrates a multimodal chain:

1. A Hindi singing/audio input is passed into a speech-to-text model.
2. The model produces a transcription.
3. The transcription can be used as a prompt signal for a text-to-image or generative image model.
4. The generated image is documented in the GitHub repository screenshots.

### Example transcription

{transcription}

### Notes

The live Space keeps the app CPU-friendly. Image generation with Diffusers is documented in the notebooks and can be enabled later with GPU resources.
""".strip()


with gr.Blocks(title=APP_TITLE) as demo:
    gr.Markdown(
        f"""
# {APP_TITLE}

A lightweight Hugging Face Spaces demo for the **HF Multimodal AI Lab**.

This Space is connected to the GitHub portfolio project: [{GITHUB_REPO}]({GITHUB_REPO})

Live Space: [{HF_SPACE}]({HF_SPACE})
"""
    )

    with gr.Tab("Image Captioning"):
        gr.Markdown("Upload an image and generate a caption using a Hugging Face image-to-text pipeline.")
        image_input = gr.Image(label="Upload image", type="pil")
        image_button = gr.Button("Generate caption")
        image_output = gr.Textbox(label="Caption", lines=4)
        image_button.click(caption_image, inputs=image_input, outputs=image_output)

    with gr.Tab("Speech to Text"):
        gr.Markdown("Upload audio and transcribe it with a lightweight Whisper model.")
        audio_input = gr.Audio(label="Upload audio", type="filepath")
        audio_button = gr.Button("Transcribe audio")
        audio_output = gr.Textbox(label="Transcription", lines=6)
        audio_button.click(transcribe_audio, inputs=audio_input, outputs=audio_output)

    with gr.Tab("Text Pipeline"):
        gr.Markdown("A simple Transformers text pipeline example for quick CPU-friendly inference.")
        text_input = gr.Textbox(label="Text", placeholder="Write a sentence to classify...", lines=4)
        text_button = gr.Button("Run text classifier")
        text_output = gr.Textbox(label="Pipeline output", lines=4)
        text_button.click(classify_text, inputs=text_input, outputs=text_output)

    with gr.Tab("Audio → Text → Image concept"):
        gr.Markdown(
            "This tab documents the multimodal chaining idea from the notebooks. "
            "GPU image generation can be added later."
        )
        workflow_text = gr.Textbox(
            label="Optional transcription",
            value="आगो जब तुम साजना आओगा फूल खिलेंगे बरसिया सावन झूम झूम",
            lines=3,
        )
        workflow_button = gr.Button("Explain workflow")
        workflow_output = gr.Markdown()
        workflow_button.click(explain_audio_to_image_workflow, inputs=workflow_text, outputs=workflow_output)

    gr.Markdown(
        f"""
---

## Project links

- GitHub repository: [{GITHUB_REPO}]({GITHUB_REPO})
- Hugging Face Space: [{HF_SPACE}]({HF_SPACE})

For heavier Diffusers image generation, use the notebooks or upgrade the Space to GPU.
"""
    )


if __name__ == "__main__":
    demo.launch()
