from __future__ import annotations

import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Literal

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


def convert_audio_to_wav(input_path: str) -> str:
    """Convert uploaded/recorded audio to a clean WAV file for local Whisper.

    Gradio recordings may arrive as webm/mp4/temp files that soundfile cannot
    decode directly. The Hugging Face Space has ffmpeg available in most Gradio
    images, so we normalize everything to 16 kHz mono WAV before transcription.
    """
    source = Path(input_path)
    if not source.exists():
        raise FileNotFoundError(f"Audio file not found: {input_path}")

    if not shutil.which("ffmpeg"):
        return input_path

    temp_dir = Path(tempfile.mkdtemp(prefix="audio_normalized_"))
    wav_path = temp_dir / "audio_16khz_mono.wav"

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(wav_path),
    ]

    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Audio conversion failed: {completed.stderr[-1000:]}")

    return str(wav_path)


def transcribe_with_local_whisper(audio_path: str) -> str:
    normalized_audio = convert_audio_to_wav(audio_path)
    recognizer = get_local_speech_recognizer()
    result = recognizer(normalized_audio)
    if isinstance(result, dict):
        return result.get("text", str(result)).strip()
    return str(result).strip()


def transcribe_with_openai(audio_path: str, api_key: str) -> str:
    client = OpenAI(api_key=api_key)
    with open(audio_path, "rb") as audio_file:
        result = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
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


def process_audio_workflow(
    audio_path: str | None,
    instruction: str,
    provider: Provider,
    api_key: str,
    model: str,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> tuple[str, str, str]:
    if not audio_path:
        return "Please upload or record an audio file first.", "", ""

    api_key = (api_key or "").strip()
    provider = provider or "OpenAI"
    model = (model or "gpt-4o-mini").strip()
    instruction = instruction or "Transcribe this audio, explain it, and create an image-generation prompt."

    progress(0.2, desc="Preparing and transcribing audio...")
    try:
        if provider == "OpenAI" and api_key:
            transcription = transcribe_with_openai(audio_path, api_key)
        else:
            transcription = transcribe_with_local_whisper(audio_path)
    except Exception as exc:
        return f"Transcription failed: {exc}", "", ""

    progress(0.7, desc="Running LLM analysis...")
    try:
        analysis = build_llm_summary(
            transcription=transcription,
            provider=provider,
            api_key=api_key,
            model=model,
            user_goal=instruction,
        )
    except Exception as exc:
        analysis = f"LLM analysis failed: {exc}"

    image_prompt = f"""
Create a cinematic image inspired by this audio transcription:

{transcription}

Style: expressive, atmospheric, emotionally rich, detailed lighting, high-quality composition.
""".strip()

    progress(1.0, desc="Done")
    return transcription, analysis, image_prompt


with gr.Blocks(title=APP_TITLE) as demo:
    gr.Markdown(f"# {APP_TITLE}")
    gr.Markdown(
        "Upload or record audio, then process it with local Whisper or your own OpenAI/OpenRouter key. "
        "No API key is stored or hardcoded."
    )
    gr.Markdown(f"GitHub: {GITHUB_REPO}  \nHugging Face Space: {HF_SPACE}")

    audio_input = gr.Audio(label="Upload or record audio", type="filepath")
    instruction_input = gr.Textbox(
        label="Instruction",
        value="Transcribe this audio, explain the meaning if possible, and create a cinematic image-generation prompt.",
        lines=3,
    )

    with gr.Accordion("API settings", open=True):
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

    run_button = gr.Button("Process audio", variant="primary")
    transcription_output = gr.Textbox(label="1. Transcription", lines=6)
    analysis_output = gr.Markdown(label="2. LLM analysis")
    image_prompt_output = gr.Textbox(label="3. Image-generation prompt", lines=8)

    run_button.click(
        process_audio_workflow,
        inputs=[audio_input, instruction_input, provider_input, api_key_input, model_input],
        outputs=[transcription_output, analysis_output, image_prompt_output],
    )


if __name__ == "__main__":
    demo.queue().launch()
