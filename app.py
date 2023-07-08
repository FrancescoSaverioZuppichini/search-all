from pathlib import Path
from time import perf_counter
from typing import Optional
import torch
import gradio as gr
from dotenv import load_dotenv
from PIL import Image

from logger import logger
from models.model_utils import ModalityType, get_embeddings, get_model

load_dotenv()
from vector_store import VectorStore

vs = VectorStore.from_env()
model = get_model()

BUCKET_LINK = "https://activeloop-sandbox-fdd0.s3.amazonaws.com/"


def search_button_handler(
    text_query: Optional[str],
    image_query: Optional[Image.Image],
    audio_mic_query: Optional[str],
    audio_file_query: Optional[str],
    limit: int = 15,
):  
    print(audio_file_query)
    if not text_query and not image_query and not audio_mic_query and not audio_file_query:
        logger.info("No inputs!")
        return
    # we have to pass a list for each query
    if text_query == "" and len(text_query) <= 0:
        text_query = None
    if text_query is not None:
        text_query = [text_query]
    if image_query is not None:
        image_query = [image_query]
    if audio_mic_query is not None:
        audio_query = [audio_mic_query]
    if audio_file_query is not None:
        audio_query = [audio_file_query]
    start = perf_counter()
    logger.info(f"Searching ...")
    embeddings = get_embeddings(model, text_query, image_query, audio_query).values()
    # if multiple inputs, we sum them
    embedding = torch.stack(list(embeddings), dim=0).sum(0).squeeze()
    logger.info(f"Model took {(perf_counter() - start) * 1000:.2f} for embedding = {embedding.shape}")
    images_paths, query_res = vs.retrieve(embedding.cpu().float(), limit)
    return [f"{BUCKET_LINK}{image_path}" for image_path in images_paths]


def clear_button_handler():
    return [None] * 6
css = """
#image_query { height: auto !important; }
#audio_file_query { height: 100px; }
"""
with gr.Blocks(css=css) as demo:
    # pairs of (input_type, data, +/-)
    inputs = gr.State([])
    with Path("docs/APP_README.md").open() as f:
        gr.Markdown(f.read())
    text_query = gr.Text(label="Text")
    with gr.Row():
        image_query = gr.Image(label="Image", type="pil", elem_id="image_query")
        with gr.Column():
            audio_mic_query = gr.Audio(label="Audio", source="microphone", type="filepath")
            audio_file_query = gr.Audio(label="Audio", type="filepath", elem_id="audio_file_query")
    markdown = gr.Markdown("")
    search_button = gr.Button("Search", label="Search", variant="primary")
    clear_button = gr.Button("Clear", label="Clear", variant="secondary")
    with gr.Accordion("Settings", open=False):
        limit = gr.Slider(
            minimum=1,
            maximum=30,
            value=15,
            step=1,
            label="search limit",
            interactive=True,
        )
    gallery = gr.Gallery().style(columns=[3], object_fit="contain", height="auto")
    clear_button.click(clear_button_handler, [], [text_query, image_query, audio_mic_query, audio_file_query, markdown, gallery])
    search_button.click(
        search_button_handler, [text_query, image_query, audio_mic_query, audio_file_query, limit], [gallery]
    )

demo.queue()
demo.launch()
