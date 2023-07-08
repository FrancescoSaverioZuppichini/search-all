from pathlib import Path
from time import perf_counter
from typing import Optional
import torch
import gradio as gr
import numpy as np
from dotenv import load_dotenv
from PIL import Image
from bs4 import BeautifulSoup
from logger import logger
from models.model_utils import ModalityType, get_embeddings, get_model

load_dotenv()
from vector_store import VectorStore
from functools import partial
# vs = VectorStore.from_env()
# model = get_model()

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
    return [None, None, None, None, "<table class='inputs-table'></table>", []]


def handle_query_button(data_type:str, mode: str, data, inputs, html_table: str):
    inputs.append((type, data, mode))

    soup = BeautifulSoup(html_table, 'html.parser')
    table = soup.find('table', {'class': 'inputs-table'})
    # Create a new row
    new_row = soup.new_tag('tr')
    # Create a new cell within the row
    new_cell = soup.new_tag('td')
    if data_type == "image":
        new_td_el = soup.new_tag('img')
        print(data)
        new_td_el['src'] = f"{BUCKET_LINK}/{data.filename}"
    elif data_type == "audio":
        new_td_el = soup.new_tag('source')
        new_td_el['src'] = data
    else:
        new_td_el = soup.new_tag('h4')
        new_td_el.string = data

    new_cell.append(new_td_el)
    new_row.append(new_cell)

    new_cell = soup.new_tag('td')
    new_td_el = soup.new_tag('h4')
    new_td_el.string = mode
    new_cell.append(new_td_el)

    new_row.append(new_cell)
    table.append(new_row)
    html_table = soup.prettify()
    return inputs, html_table

def handle_audio_query_button(data_type:str, mode: str, audio_mic_query, audio_file_query, inputs, html_table):
    audio_query = audio_mic_query or audio_file_query
    return handle_query_button(data_type, mode, audio_query, inputs, html_table)
    
css = """
#image_query { height: auto !important; max-height: 400px !important; }
#audio_file_query { height: 100px; }
"""
with gr.Blocks(css=css) as demo:
    # pairs of (input_type, data, +/-)
    inputs = gr.State([])
    with Path("docs/APP_README.md").open() as f:
        gr.Markdown(f.read())
    with gr.Row():
        with gr.Column():
            with gr.Row():
                text_query = gr.Text(label="Text")
                text_query_plus_button = gr.Button("➕").style(full_width=False)
                text_query_minus_button = gr.Button("➖").style(full_width=False)
            with gr.Row():
                image_query = gr.Image(label="Image", type="pil", elem_id="image_query")
                image_query_plus_button = gr.Button("➕").style(full_width=False)
                image_query_minus_button = gr.Button("➖").style(full_width=False)
            with gr.Row():
                with gr.Column():
                    audio_mic_query = gr.Audio(label="Audio", source="microphone", type="filepath")
                    audio_file_query = gr.Audio(label="Audio", type="filepath", elem_id="audio_file_query")
                audio_query_plus_button = gr.Button("➕").style(full_width=False)
                audio_query_minus_button = gr.Button("➖").style(full_width=False)
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
    html_table = gr.HTML("<table class='inputs-table'><tr><td>Query</td><td>Mode</td></tr></table>")
    text_query_plus_button.click(partial(handle_query_button, "text", "+"), [text_query, inputs, html_table], [inputs, html_table])
    text_query_minus_button.click(partial(handle_query_button, "text", "-"), [text_query, inputs, html_table], [inputs, html_table])

    audio_query_plus_button.click(partial(handle_audio_query_button, "audio", "+"), [audio_mic_query, audio_file_query, inputs, html_table], [inputs, html_table])
    audio_query_minus_button.click(partial(handle_audio_query_button, "audio", "-"), [audio_mic_query, audio_file_query, inputs, html_table], [inputs, html_table])

    image_query_plus_button.click(partial(handle_query_button, "image", "+"), [image_query, inputs, html_table], [inputs, html_table])
    image_query_minus_button.click(partial(handle_query_button, "image", "-"), [image_query, inputs, html_table], [inputs, html_table])

    gallery = gr.Gallery().style(columns=[3], object_fit="contain", height="auto")
    clear_button.click(clear_button_handler, [], [text_query, image_query, audio_mic_query, audio_file_query, html_table, gallery])
    search_button.click(
        search_button_handler, [text_query, image_query, audio_mic_query, audio_file_query, limit], [gallery]
    )

demo.queue()
demo.launch()
