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
def get_local_speech_recognizer():
    from transformers import pipeline

    return pipeline("automatic-speech-recognition", model="openai/whisper-tiny")


def get_file_path(file_value: Any) -> str | None:
    """Normalize Gradio multimodal file values into a filepath."""
    if not file_value:
        return None
    if isinstance(file_value, str):
        return file_value
    if isinstance(file_value, dict):
        return file_value.get("path") or file_value.get("name")
    return getattr(file_value, "path", None) or getattr(file_value, "name", None)


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
        return "No API key was provided, so only transcription was generated. Add your own OpenAI or OpenRouter key for LLM analysis."

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
2. Short meaning or interpretation if possible
3. Image-generation prompt inspired by the transcription
4. Notes about limitations if the transcription may be imperfect
""".strip()

    response = client.chat.completions.create(
        model=model.strip(),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response.choices[0].message.content or "No response generated."


def process_chat_message(
    message: dict[str, Any],
    history: list[dict[str, Any]],
    provider: Provider,
    api_key: str,
    model: str,
    default_goal: str,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> str:
    """ChatInterface handler with audio upload inside the message box.

    Users provide their own API key. No key is stored or hardcoded.
    """
    text = (message.get("text") or "").strip()
    files = message.get("files") or []

    audio_path = None
    for file_value in files:
        file_path = get_file_path(file_value)
        if file_path and file_path.lower().endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".mp4")):
            audio_path = file_path
            break

    if not audio_path:
        return (
            "Please attach an audio file using the upload button in the message box. "
            "You can also type an instruction such as: 'Transcribe this and create an image prompt.'"
        )

    api_key = (api_key or "").strip()
    provider = provider or "OpenAI"
    model = (model or "gpt-4o-mini").strip()
    user_goal = text or default_goal or "Transcribe this audio, explain it, and create an image-generation prompt."

    progress(0.15, desc="Audio received. Starting transcription...")

    try:
        if provider == "OpenAI" and api_key:
            transcription = transcribe_with_openai(audio_path, api_key)
        else:
            transcription = transcribe_with_local_whisper(audio_path)
    except Exception as exc:
        return f"Transcription failed: {exc}"

    progress(0.65, desc="Building LLM analysis with user-provided key...")

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

    return f"""
## 1. Transcription

{transcription}

## 2. LLM analysis

{llm_output}

## 3. Image-generation prompt

```text
{image_prompt}
```
""".strip()


provider_input = gr.Dropdown(
    choices=["OpenAI", "OpenRouter"],
    value="OpenAI",
    label="Provider for LLM analysis",
)
api_key_input = gr.Textbox(
    label="Your API key",
    type="password",
    placeholder="Paste your OpenAI or OpenRouter key. It is used only for this request.",
)
model_input = gr.Textbox(
    label="Model",
    value="gpt-4o-mini",
    placeholder="OpenAI: gpt-4o-mini | OpenRouter: openai/gpt-4o-mini",
)
default_goal_input = gr.Textbox(
    label="Default instruction",
    value="Transcribe this audio, explain the meaning if possible, and create a cinematic image-generation prompt.",
    lines=3,
)


demo = gr.ChatInterface(
    fn=process_chat_message,
    multimodal=True,
    type="messages",
    title=APP_TITLE,
    description=(
        "Upload or record audio directly in the chat box. The app transcribes the audio, "
        "then uses your own OpenAI or OpenRouter API key for LLM analysis. No keys are stored.\n\n"
        f"GitHub: {GITHUB_REPO} | Hugging Face Space: {HF_SPACE}"
    ),
    textbox=gr.MultimodalTextbox(
        file_types=["audio"],
        placeholder="Upload audio here and optionally type: Transcribe this and create an image prompt...",
    ),
    additional_inputs=[provider_input, api_key_input, model_input, default_goal_input],
    additional_inputs_accordion=gr.Accordion("API settings", open=True),
)


if __name__ == "__main__":
    demo.queue().launch()
