from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import gradio as gr
from openai import OpenAI

APP_TITLE = "Multimodal AI Playground"
GITHUB_REPO = "https://github.com/sakshianil/hf-multimodal-ai-lab"
HF_SPACE = "https://huggingface.co/spaces/sakshianil/multimodal-ai-playground"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

Provider = Literal["OpenAI", "OpenRouter"]


@lru_cache(maxsize=1)
def get_image_captioner():
    from transformers import pipeline

    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")


@lru_cache(maxsize=1)
def get_local_speech_recognizer():
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


def transcribe_with_local_whisper(audio_path: str) -> str:
    recognizer = get_local_speech_recognizer()
    result = recognizer(audio_path)
    if isinstance(result, dict):
        return result.get("text", str(result)).strip()
    return str(result).strip()


def transcribe_with_openai(audio_path: str, api_key: str) -> str:
    client = OpenAI(api_key=api_key)
    with open(audio_path, "rb") as audio_file:
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
        )
    return (getattr(result, "text", None) or str(result)).strip()


def build_llm_summary(
    transcription: str,
    provider: Provider,
    api_key: str,
    model: str,
    user_goal: str,
) -> str:
    if not api_key.strip():
        return "No API key provided, so only transcription was generated. Add your own OpenAI or OpenRouter key for LLM analysis."

    base_url = OPENROUTER_BASE_URL if provider == "OpenRouter" else None
    client = OpenAI(api_key=api_key.strip(), base_url=base_url)

    prompt = f"""
You are a helpful multimodal AI assistant.

A user uploaded an audio file. The audio has been transcribed.

User goal:
{user_goal or "Explain the transcription and create a strong image-generation prompt."}

Transcription:
{transcription}

Return a structured response with:
1. Clean transcription
2. Short meaning/interpretation if possible
3. Image-generation prompt inspired by the transcription
4. Notes about limitations if the transcription may be imperfect
""".strip()

    response = client.chat.completions.create(
        model=model.strip(),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response.choices[0].message.content or "No response generated."


def process_audio_workflow(
    audio_path: str | None,
    provider: Provider,
    api_key: str,
    model: str,
    user_goal: str,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> tuple[str, str, str]:
    """Process user audio in the background.

    No API key is stored. The key is only used for the current request.
    OpenAI can be used for both transcription and LLM analysis.
    OpenRouter is used for LLM analysis after local Whisper transcription.
    """
    if not audio_path:
        return "Please upload or record audio first.", "", ""

    api_key = (api_key or "").strip()
    provider = provider or "OpenAI"
    model = (model or "gpt-4o-mini").strip()

    progress(0.1, desc="Preparing audio...")

    try:
        if provider == "OpenAI" and api_key:
            progress(0.25, desc="Transcribing audio with your OpenAI key...")
            transcription = transcribe_with_openai(audio_path, api_key)
        else:
            progress(0.25, desc="Transcribing audio locally with Whisper tiny...")
            transcription = transcribe_with_local_whisper(audio_path)
    except Exception as exc:
        return f"Transcription failed: {exc}", "", ""

    progress(0.65, desc="Building multimodal interpretation...")

    try:
        llm_output = build_llm_summary(
            transcription=transcription,
            provider=provider,
            api_key=api_key,
            model=model,
            user_goal=user_goal,
        )
    except Exception as exc:
        llm_output = f"LLM analysis failed: {exc}"

    image_prompt = f"""
Create a cinematic image inspired by this audio transcription:

{transcription}

Style: expressive, atmospheric, emotionally rich, detailed lighting, high-quality composition.
""".strip()

    progress(1.0, desc="Done")
    return transcription, llm_output, image_prompt


def transcribe_audio(audio_path: str | None) -> str:
    if not audio_path:
        return "Please upload an audio file first."
    try:
        return transcribe_with_local_whisper(audio_path)
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

1. A user uploads or records audio.
2. The app transcribes the audio.
3. The transcription is sent to an LLM using the user's own OpenAI or OpenRouter key.
4. The LLM returns a structured interpretation and an image-generation prompt.
5. The image prompt can be used in a text-to-image model.

### Example transcription

{transcription}

### Privacy and key handling

The app asks the user to paste their own API key. The key is not saved in the repository and is only used for the active request.
""".strip()


with gr.Blocks(title=APP_TITLE) as demo:
    gr.Markdown(
        f"""
# {APP_TITLE}

A lightweight Hugging Face Spaces demo for the **HF Multimodal AI Lab**.

This app accepts user audio and processes it in the background. It asks users to provide their own OpenAI or OpenRouter API key for LLM analysis. No private keys are stored in the repository.

GitHub portfolio project: [{GITHUB_REPO}]({GITHUB_REPO})  
Live Space: [{HF_SPACE}]({HF_SPACE})
"""
    )

    with gr.Tab("Audio workflow"):
        gr.Markdown(
            "Upload or record audio. The app transcribes it, then uses your own OpenAI/OpenRouter key to produce a structured interpretation and an image-generation prompt."
        )
        workflow_audio = gr.Audio(label="Upload or record audio", type="filepath")
        with gr.Row():
            provider = gr.Dropdown(
                choices=["OpenAI", "OpenRouter"],
                value="OpenAI",
                label="Provider for LLM analysis",
            )
            model = gr.Textbox(
                label="Model",
                value="gpt-4o-mini",
                placeholder="OpenAI: gpt-4o-mini | OpenRouter: openai/gpt-4o-mini",
            )
        api_key = gr.Textbox(
            label="Your API key",
            type="password",
            placeholder="Paste your OpenAI or OpenRouter key. It is used only for this request.",
        )
        user_goal = gr.Textbox(
            label="Goal / instruction",
            value="Transcribe this audio, explain the meaning if possible, and create a cinematic image-generation prompt.",
            lines=3,
        )
        workflow_button = gr.Button("Process audio in background")
        transcription_output = gr.Textbox(label="1. Transcription", lines=5)
        analysis_output = gr.Markdown(label="2. LLM analysis")
        image_prompt_output = gr.Textbox(label="3. Image-generation prompt", lines=8)
        workflow_button.click(
            process_audio_workflow,
            inputs=[workflow_audio, provider, api_key, model, user_goal],
            outputs=[transcription_output, analysis_output, image_prompt_output],
        )

    with gr.Tab("Image Captioning"):
        gr.Markdown("Upload an image and generate a caption using a Hugging Face image-to-text pipeline.")
        image_input = gr.Image(label="Upload image", type="pil")
        image_button = gr.Button("Generate caption")
        image_output = gr.Textbox(label="Caption", lines=4)
        image_button.click(caption_image, inputs=image_input, outputs=image_output)

    with gr.Tab("Speech to Text only"):
        gr.Markdown("Upload audio and transcribe it locally with a lightweight Whisper model. This tab does not require an API key.")
        audio_input = gr.Audio(label="Upload audio", type="filepath")
        audio_button = gr.Button("Transcribe audio locally")
        audio_output = gr.Textbox(label="Transcription", lines=6)
        audio_button.click(transcribe_audio, inputs=audio_input, outputs=audio_output)

    with gr.Tab("Text Pipeline"):
        gr.Markdown("A simple Transformers text pipeline example for quick CPU-friendly inference.")
        text_input = gr.Textbox(label="Text", placeholder="Write a sentence to classify...", lines=4)
        text_button = gr.Button("Run text classifier")
        text_output = gr.Textbox(label="Pipeline output", lines=4)
        text_button.click(classify_text, inputs=text_input, outputs=text_output)

    with gr.Tab("Audio → Text → Image concept"):
        workflow_text = gr.Textbox(
            label="Optional transcription",
            value="आगो जब तुम साजना आओगा फूल खिलेंगे बरसिया सावन झूम झूम",
            lines=3,
        )
        concept_button = gr.Button("Explain workflow")
        concept_output = gr.Markdown()
        concept_button.click(explain_audio_to_image_workflow, inputs=workflow_text, outputs=concept_output)

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
    demo.queue().launch()
