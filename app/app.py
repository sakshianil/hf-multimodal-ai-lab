from __future__ import annotations

import gradio as gr


def describe_project(image, audio, text_prompt: str) -> str:
    """Placeholder app for the future Hugging Face Space demo.

    The repository notebooks contain the full experiments. This lightweight app
    is intentionally CPU-friendly and can later be connected to image captioning,
    speech-to-text, VQA, or text-to-image pipelines.
    """
    parts = ["HF Multimodal AI Lab demo scaffold"]
    if image is not None:
        parts.append("Image input received. Connect this to image captioning or VQA.")
    if audio is not None:
        parts.append("Audio input received. Connect this to speech-to-text.")
    if text_prompt:
        parts.append(f"Text prompt: {text_prompt}")
    return "\n".join(parts)


with gr.Blocks(title="HF Multimodal AI Lab") as demo:
    gr.Markdown("# HF Multimodal AI Lab")
    gr.Markdown(
        "A lightweight Gradio scaffold for future Hugging Face Spaces deployment. "
        "The full learning workflow is documented in the notebooks."
    )
    with gr.Row():
        image = gr.Image(label="Image input", type="pil")
        audio = gr.Audio(label="Audio input", type="filepath")
    text_prompt = gr.Textbox(label="Text prompt", placeholder="Ask a question or describe the generation task...")
    button = gr.Button("Run demo")
    output = gr.Textbox(label="Output", lines=8)
    button.click(describe_project, inputs=[image, audio, text_prompt], outputs=output)


if __name__ == "__main__":
    demo.launch()
