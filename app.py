from pathlib import Path
from time import perf_counter
from typing import Optional
import torch
import gradio as gr
import numpy as np
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
    audio_query: Optional[str],
    limit: int = 15,
):
    if not text_query and not image_query and not audio_query:
        logger.info("No inputs!")
        return
    # we have to pass a list for each query
    if text_query == "" and len(text_query) <= 0:
        text_query = None
    if text_query is not None:
        text_query = [text_query]
    if image_query is not None:
        image_query = [image_query]
    if audio_query is not None:
        audio_query = [audio_query]
    start = perf_counter()
    logger.info(f"Searching ...")
    embeddings = get_embeddings(model, text_query, image_query, audio_query).values()
    embeddings = torch.stack(list(embeddings), dim=0).squeeze()
    weights = torch.ones((embeddings.shape[0], 1), device=embeddings.device) / embeddings.shape[0]
    embedding = (embeddings / weights).sum(0).cpu().float()
    print(embedding.shape, embedding.dtype)
    logger.info(f"Model took {(perf_counter() - start) * 1000:.2f}")
    images_paths = vs.retrieve(embedding, limit)
    return [f"{BUCKET_LINK}{image_path}" for image_path in images_paths]


with gr.Blocks() as demo:
    with Path("docs/APP_README.md").open() as f:
        gr.Markdown(f.read())
    text_query = gr.Text(label="Text")
    with gr.Row():
        image_query = gr.Image(label="Image", type="pil")
        with gr.Column():
            audio_query = gr.Audio(label="Audio", source="microphone", type="filepath")
            search_button = gr.Button("Search", label="Search", variant="primary")
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
    search_button.click(
        search_button_handler, [text_query, image_query, audio_query, limit], [gallery]
    )

demo.queue()
demo.launch()
